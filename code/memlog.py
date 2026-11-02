"""System memory logger for long GPU jobs (no torch import, ~30 MB RAM).

    python code/memlog.py                  # appends one row per minute to runs/logs/memlog.csv
    python code/memlog.py --summary 30     # worst case over the last 30 min + advice

Columns: time, ram_avail_gib, commit_used_gib, commit_limit_gib, py_procs, py_private_gib,
vram_used_mib, gpu_util, gpu_w, gpu_temp. The 30-minute check reads --summary, which
reports the WORST value inside the window (min free RAM, max VRAM, max commit), not just
the latest sample, so a short spike during validation is not missed.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import time
from pathlib import Path

import psutil

LOG = Path(r"T:\Group_Project\runs\logs\memlog.csv")
COLS = ["time", "ram_avail_gib", "commit_used_gib", "commit_limit_gib", "py_procs", "py_private_gib",
        "vram_used_mib", "gpu_util", "gpu_w", "gpu_temp"]
VRAM_TOTAL_MIB = 12227


def commit_gib() -> tuple[float, float]:
    """True Windows commit charge / limit (GetPerformanceInfo). This, not 'available RAM',
    decides whether an allocation can fail: memmap streaming of the 84 GB pixel cache
    parks clean file pages in worker working sets, which drives 'available' to ~0
    without risk (those pages are dropped on demand)."""
    import ctypes
    from ctypes import wintypes

    class PI(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
            "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + \
            [(n, wintypes.DWORD) for n in ("HandleCount", "ProcessCount", "ThreadCount")]
    pi = PI(); pi.cb = ctypes.sizeof(PI)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    return pi.CommitTotal * pi.PageSize / 2 ** 30, pi.CommitLimit * pi.PageSize / 2 ** 30


def sample() -> dict:
    vm = psutil.virtual_memory()
    cu, cl = commit_gib()
    py = [p for p in psutil.process_iter(["name", "memory_info"]) if (p.info["name"] or "").lower().startswith("python")]
    priv = sum(getattr(p.info["memory_info"], "private", p.info["memory_info"].rss) for p in py if p.info["memory_info"])
    try:
        g = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu",
                                     "--format=csv,noheader,nounits"], text=True, timeout=20).strip().split(", ")
    except Exception:
        g = ["", "", "", ""]
    return {"time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ram_avail_gib": round(vm.available / 2 ** 30, 2),
            "commit_used_gib": round(cu, 2), "commit_limit_gib": round(cl, 2),
            "py_procs": len(py), "py_private_gib": round(priv / 2 ** 30, 2),
            "vram_used_mib": g[0], "gpu_util": g[1], "gpu_w": g[2], "gpu_temp": g[3]}


def summary(minutes: int) -> int:
    rows = list(csv.DictReader(open(LOG, encoding="utf-8")))
    cut = dt.datetime.now() - dt.timedelta(minutes=minutes)
    win = [r for r in rows if dt.datetime.strptime(r["time"], "%Y-%m-%d %H:%M:%S") >= cut] or rows[-1:]
    f = lambda k: [float(r[k]) for r in win if r[k] not in ("", None)]
    ram, vram, com, temp = f("ram_avail_gib"), f("vram_used_mib"), f("commit_used_gib"), f("gpu_temp")
    lim = float(win[-1]["commit_limit_gib"])
    print("[memcheck] last %d min, %d samples (%s .. %s)" % (minutes, len(win), win[0]["time"], win[-1]["time"]))
    print("  RAM available : min %.1f GiB  now %.1f GiB" % (min(ram), ram[-1]))
    print("  commit charge : max %.1f / %.1f GiB" % (max(com), lim))
    if vram:
        print("  VRAM used     : max %.0f MiB  now %.0f MiB  (of %d)" % (max(vram), vram[-1], VRAM_TOTAL_MIB))
    if temp:
        print("  GPU temp      : max %.0f C" % max(temp))
    print("  python procs  : now %s using %s GiB private" % (win[-1]["py_procs"], win[-1]["py_private_gib"]))
    adv = []
    if max(com) > lim - 4:
        adv.append("COMMIT NEAR LIMIT (<4 GiB headroom): allocations can fail; lower DataLoader workers "
                   "(runs/ft_overrides.json: workers 2, eval_workers 1) / close apps")
    elif min(ram) < 2.0:
        print("  note: available RAM low but commit headroom %.1f GiB -> memmap file cache, "
              "reclaimable, no action" % (lim - max(com)))
    if vram and max(vram) > 0.92 * VRAM_TOTAL_MIB:
        adv.append("VRAM HIGH: >92%% of the card; set micro 1 / k 6 in runs/ft_overrides.json")
    if temp and max(temp) >= 90:
        adv.append("GPU HOT (>=90 C): check laptop ventilation / power mode")
    print("  STATUS: " + ("OK" if not adv else "ACTION NEEDED\n   - " + "\n   - ".join(adv)))
    return 1 if adv else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=float, default=60)
    ap.add_argument("--summary", type=int, default=0, help="summarise the last N minutes and exit")
    a = ap.parse_args()
    if a.summary:
        return summary(a.summary)
    new = not LOG.exists()
    while True:
        with open(LOG, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, COLS)
            if new:
                w.writeheader(); new = False
            w.writerow(sample())
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
