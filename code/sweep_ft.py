"""T4.6 + T4.7: fine-tuning LR pilot, LR choice, and the main fine-tuning sweep.

    python code/sweep_ft.py --pilot                    # 2 models x 3 LRs, r0 s0 N=520, val only
    python code/sweep_ft.py --rotations 0              # main sweep (needs the pilot's choice)
    python code/sweep_ft.py --pilot --rotations 0,1    # both, back to back

Each run is its own subprocess (train_ft.py), so all GPU/host memory is returned to the
OS between runs and a crash costs one run, not the sweep. Runs are resumable and skipped
when finished. The LR is chosen per model on the pilot's VALIDATION macro AUC only and
written to results/ft_lr_choice.json; the winning pilot run doubles as the main sweep's
N=520 point (it is only scored on test, never retrained). Within a rotation, sizes go
small -> large with M1/M2 alternating, so a partial sweep still gives matched pairs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from pathlib import Path

# deliberately NOT importing common (which pulls in torch, ~2 GB of commit on Windows):
# this process only launches subprocesses and must stay small for the whole sweep
GP = Path(r"T:\Group_Project")
RESULTS, RUNS = GP / "results", GP / "runs"
SIZES = [65, 130, 260, 520, 1040, 2080, 2600]


def fmt_lr(lr: float) -> str:
    return ("%.0e" % lr).replace("e-0", "e-")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")

PY = sys.executable
PILOT_LRS = [1e-5, 3e-5, 1e-4]
PILOT = dict(rotation=0, seed=0, n=520)
SCRATCH = dict(models=["M5"], rotation=0, sizes=[260, 1040, 2600], lr=1e-3)   # T4.9
OVERRIDES = RUNS / "ft_overrides.json"


def name(m, r, s, n, lr):
    return "ft-%s-r%d-s%d-n%04d-lr%s" % (m, r, s, n, fmt_lr(lr))


def overrides() -> list[str]:
    """Live tuning without stopping the sweep: runs/ft_overrides.json, read before EVERY run,
    e.g. {"workers": 2, "eval_workers": 1, "micro": 1, "k": 6}. Applies to runs not yet started."""
    if not OVERRIDES.exists():
        return []
    try:
        d = json.loads(OVERRIDES.read_text())
    except Exception as e:
        print("[sweep_ft] ignoring unreadable overrides:", e)
        return []
    out = []
    for k in ("workers", "eval_workers", "micro", "k"):
        if k in d:
            out += ["--" + k.replace("_", "-"), str(d[k])]
    return out


def run(m, r, s, n, lr, extra=(), log=None) -> int:
    ov = overrides()
    cmd = [PY, "-X", "utf8", str(GP / "code" / "train_ft.py"), "--model", m, "--rotation", str(r),
           "--seed", str(s), "--n", str(n), "--lr", str(lr), *extra, *ov]
    if ov:
        print("[sweep_ft] overrides:", " ".join(ov), flush=True)
    for attempt in (1, 2):
        print("[sweep_ft] %s  start %s (attempt %d)" % (now(), name(m, r, s, n, lr), attempt), flush=True)
        t0 = time.time()
        with open(RUNS / "logs" / (name(m, r, s, n, lr) + ".log"), "a", encoding="utf-8") as f:
            rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(GP / "code"))
        print("[sweep_ft] %s  end   %s rc=%d  %.2f h" % (now(), name(m, r, s, n, lr), rc, (time.time() - t0) / 3600),
              flush=True)
        if rc == 0:
            return 0
    return rc


def choose(models) -> dict:
    out = {}
    for m in models:
        cand = []
        for lr in PILOT_LRS:
            p = RUNS / name(m, PILOT["rotation"], PILOT["seed"], PILOT["n"], lr) / "metrics.json"
            if p.exists():
                cand.append((json.loads(p.read_text())["val_macro"], lr))
        if len(cand) == len(PILOT_LRS):
            v, lr = max(cand)
            out[m] = {"lr": lr, "val_macro": v, "pilot": {fmt_lr(l): vv for vv, l in cand}}
    (RESULTS / "ft_lr_choice.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="M1,M2")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--rotations", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--scratch", action="store_true", help="T4.9 from-scratch lower bound (M5, r0, 3 sizes)")
    a = ap.parse_args()
    models = a.models.split(",")
    (RUNS / "logs").mkdir(parents=True, exist_ok=True)
    fails = []
    if a.pilot:
        for lr in PILOT_LRS:                     # LR outer: both models get each LR before the next
            for m in models:
                if run(m, PILOT["rotation"], PILOT["seed"], PILOT["n"], lr, ["--val-only"]):
                    fails.append(name(m, 0, 0, 520, lr))
        print("[sweep_ft] LR choice:", json.dumps(choose(models)), flush=True)
    if a.rotations:
        ch = choose(models)
        missing = [m for m in models if m not in ch]
        if missing:
            print("[sweep_ft] no LR choice for %s; run --pilot first" % missing)
            return 2
        rots = [int(x) for x in a.rotations.split(",")]
        # Rotations are completed IN ORDER: an earlier rotation left unfinished (e.g. the r0 sweep
        # was interrupted on 2026-10-26 and the queue moved on) is finished first. Finished runs
        # are skipped instantly.
        rots = sorted(set(range(min(rots))) | set(rots))
        for r in rots:
            for n in [int(x) for x in a.sizes.split(",")]:
                for m in models:
                    if run(m, r, a.seed, n, ch[m]["lr"]):
                        fails.append(name(m, r, a.seed, n, ch[m]["lr"]))
    if a.scratch:
        for n in SCRATCH["sizes"]:
            for m in SCRATCH["models"]:
                if run(m, SCRATCH["rotation"], a.seed, n, SCRATCH["lr"]):
                    fails.append(name(m, SCRATCH["rotation"], a.seed, n, SCRATCH["lr"]))
    print("[sweep_ft] %s finished; %d failed %s" % (now(), len(fails), fails))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
