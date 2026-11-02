"""T4.4: fine-tuning throughput + memory benchmark, data-loader throughput, and the
frozen/FT input-equivalence test.

    python code/bench_ft.py                    # M1 and M2, all configs
    python code/bench_ft.py --models M1 --steps 10

Every training config is run at its WORST case (micro * MAX_SLICES slices per micro-batch,
the largest batch train_ft.py can ever build), so the measured peak is an upper bound for
the real run. A config is "safe" when its peak reserved VRAM leaves >= 1.5 GiB headroom
under the 90 % process cap. Writes results/tables/throughput.csv.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from common import MODELS, RESULTS, gpu_cap, load_rotation, load_series_index, load_subset, load_targets
from ft_data import MAX_SLICES, StudyDataset, ViewSampler, collate, gpu_prep
from ft_model import StudyModel, equivalence_test, param_groups
from train_ft import loader


def bench_train(model_id, micro, ckpt, steps, cap):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    dim = MODELS[model_id]["dim"]
    m = StudyModel(model_id, torch.zeros(dim), torch.ones(dim), ckpt=ckpt).cuda()
    opt = torch.optim.AdamW(param_groups(m, 3e-5))
    M = micro * MAX_SLICES
    px = torch.randint(0, 255, (M, 320, 320), dtype=torch.uint8, device="cuda")
    series = torch.arange(M, device="cuda") // 8
    slot = torch.zeros(micro, MAX_SLICES, dtype=torch.long, device="cuda")
    pos = torch.rand(micro, MAX_SLICES, device="cuda")
    mask = torch.ones(micro, MAX_SLICES, dtype=torch.bool, device="cuda")
    t, w = torch.rand(micro, 12, device="cuda"), torch.ones(micro, 12, device="cuda")
    m.train()
    try:
        for i in range(steps + 3):
            if i == 3:
                torch.cuda.synchronize(); t0 = time.perf_counter()
            x = gpu_prep(px, series, m.enc.mean, m.enc.std, aug=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = m.head_forward(m.enc(x), slot, pos, mask)
            loss = F.binary_cross_entropy_with_logits(lg.float(), t, weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        res = {"sl_per_s": steps * M / el, "oom": False}
    except torch.cuda.OutOfMemoryError:
        res = {"sl_per_s": None, "oom": True}
    res.update({"peak_alloc_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30})
    res["safe"] = (not res["oom"]) and res["peak_reserved_gib"] <= cap - 1.5
    del m, opt
    torch.cuda.empty_cache()
    return res


@torch.no_grad()
def bench_infer(model_id, chunk=128, n=1024):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    dim = MODELS[model_id]["dim"]
    m = StudyModel(model_id, torch.zeros(dim), torch.ones(dim)).cuda().eval()
    px = torch.randint(0, 255, (chunk, 320, 320), dtype=torch.uint8, device="cuda")
    for i in range(n // chunk + 2):
        if i == 2:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.enc(gpu_prep(px, None, m.enc.mean, m.enc.std))
    torch.cuda.synchronize()
    r = {"sl_per_s": n / (time.perf_counter() - t0), "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    del m
    torch.cuda.empty_cache()
    return r


def bench_loader(workers, studies=96):
    tg, se = load_targets(), load_series_index()
    sids = load_subset(0, 0, 520)
    ds = StudyDataset(sids, "train", 8, 0, tg, se)
    dl = loader(ds, 1, workers, ViewSampler(len(sids), studies, 0))
    t0, nsl, first = time.perf_counter(), 0, None
    for b in dl:
        if first is None:
            first = time.perf_counter() - t0; t0 = time.perf_counter()
        nsl += int(b["mask"].sum())
    el = time.perf_counter() - t0
    tds = StudyDataset(load_rotation(0)["test"][:40], "test", 8, 0, tg, se)
    t1, tsl = time.perf_counter(), 0
    for b in loader(tds, 2, 2):
        tsl += int(b["mask"].sum())
    return {"train_sl_per_s": nsl / el, "startup_s": first, "test_sl_per_s": tsl / (time.perf_counter() - t1),
            "mean_train_slices_per_study": nsl / max(1, studies - 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="M1,M2,M5,M6")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--skip-equiv", action="store_true")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    cap = gpu_cap(0.90)
    print("[bench] VRAM cap %.2f GiB" % cap, flush=True)
    rows = []
    for mid in a.models.split(","):
        if not a.skip_equiv and not MODELS[mid].get("scratch"):
            rows.append({"model": mid, "test": "equivalence_cos_min", "value": equivalence_test(mid)})
        r = bench_infer(mid)
        print("[bench] %s inference  %.0f sl/s  peak %.2f GiB" % (mid, r["sl_per_s"], r["peak_alloc_gib"]), flush=True)
        rows.append({"model": mid, "test": "infer_sl_per_s", "value": r["sl_per_s"]})
        for micro in (1, 2):
            for ckpt in (True, False):
                r = bench_train(mid, micro, ckpt, a.steps, cap)
                tag = "train_micro%d_%s" % (micro, "ckpt" if ckpt else "nockpt")
                print("[bench] %s %-22s %s  peak alloc %.2f / reserved %.2f GiB  %s" %
                      (mid, tag, "OOM" if r["oom"] else "%.0f sl/s" % r["sl_per_s"], r["peak_alloc_gib"],
                       r["peak_reserved_gib"], "SAFE" if r["safe"] else "not safe"), flush=True)
                for k, v in r.items():
                    rows.append({"model": mid, "test": tag + "_" + k, "value": float(v) if v is not None else None})
    for wk in (2, 3, 4):
        r = bench_loader(wk)
        print("[bench] loader %d workers: train %.0f sl/s (%.1f slices/study, startup %.0fs), test %.0f sl/s"
              % (wk, r["train_sl_per_s"], r["mean_train_slices_per_study"], r["startup_s"], r["test_sl_per_s"]),
              flush=True)
        for k, v in r.items():
            rows.append({"model": "-", "test": "loader_w%d_%s" % (wk, k), "value": float(v)})
    pl.DataFrame(rows).write_csv(RESULTS / "tables" / "throughput.csv")
    print("[bench] wrote results/tables/throughput.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
