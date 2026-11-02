"""T3.3: the frozen learning-curve sweep, resumable.

    python code/sweep_frozen.py                                  # everything (1,680 runs)
    python code/sweep_frozen.py --models M3,M4 --rotations 0     # a slice of it
    python code/sweep_frozen.py --select-only                    # just rewrite the selection table

Rotation is the OUTER loop and model the inner one, so after rotation 0 finishes every
model has a complete, matched curve. One model's embeddings are resident at a time
(<= 3.4 GiB); runs whose metrics.json exists are skipped, so killing and restarting
loses at most the run in flight. After the sweep, results/frozen_selected.csv keeps the
best hyperparameters per (model, rotation, seed, N) by VALIDATION macro AUC.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
import traceback

import polars as pl
import torch

from common import CACHE, MODELS, RESULTS, ROTATIONS, RUNS, SEEDS, SIZES, gpu_cap, now
from train_frozen_lc import FrozenCtx, run_name, run_one

GRID = [(lr, dr) for lr in (3e-4, 1e-3) for dr in (0.2, 0.5)]


def emb_ready(m: str) -> bool:
    p = CACHE / ("emb_train_%s.json" % MODELS[m]["tag"])
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    return d.get("done", d["n"]) == d["n"]


def select() -> pl.DataFrame:
    rows = []
    for p in RUNS.glob("frozen-*/metrics.json"):
        m = json.loads(p.read_text())
        rows.append({"run": m["run"], "model": m["model"], "rotation": m["rotation"], "seed": m["seed"],
                     "n_train": m["n_train"], "lr": m["hp"]["lr"], "drop": m["hp"]["drop"],
                     "val_macro": m["val_macro"], "test_macro": m["test_macro"],
                     "gold_macro": m["gold_macro"], "wall_s": m["wall_s"]})
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    sel = (df.sort("val_macro", descending=True)
             .group_by(["model", "rotation", "seed", "n_train"], maintain_order=True).first()
             .sort(["model", "rotation", "seed", "n_train"]))
    sel.write_csv(RESULTS / "frozen_selected.csv")
    df.write_csv(RESULTS / "frozen_all_runs.csv")
    return sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="M3,M4,M1,M2")
    ap.add_argument("--rotations", default=",".join(map(str, ROTATIONS)))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--sizes", default=",".join(map(str, SIZES)))
    ap.add_argument("--select-only", action="store_true")
    args = ap.parse_args()
    if args.select_only:
        print(select())
        return 0

    torch.backends.cudnn.benchmark = True
    print("[sweep] VRAM cap %.1f GiB" % gpu_cap())
    models = args.models.split(",")
    rots = [int(x) for x in args.rotations.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    sizes = [int(x) for x in args.sizes.split(",")]
    todo = [(r, m, s, n, lr, dr) for r in rots for m in models
            for s, n, (lr, dr) in itertools.product(seeds, sizes, GRID)
            if not (RUNS / run_name(m, r, s, n, lr, dr) / "metrics.json").exists()]
    print("[sweep] %s  %d runs to do" % (now(), len(todo)))

    ctx, failed, done, t0 = None, [], 0, time.perf_counter()
    for attempt in (1, 2):                       # second pass retries failures once
        queue, failed = (todo if attempt == 1 else failed), []
        for (r, m, s, n, lr, dr) in queue:
            if not emb_ready(m):
                print("[sweep] skip %s: embeddings not finished" % m)
                continue
            if ctx is None or ctx.model_id != m:
                ctx = None; gc.collect(); torch.cuda.empty_cache()
                ctx = FrozenCtx(m)
                print("[sweep] loaded %s embeddings (%d dims)" % (m, ctx.dim), flush=True)
            try:
                res = run_one(ctx, r, s, n, lr, dr)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                failed.append((r, m, s, n, lr, dr)); print("[sweep] OOM on", run_name(m, r, s, n, lr, dr))
                continue
            except Exception:
                traceback.print_exc()
                failed.append((r, m, s, n, lr, dr))
                continue
            done += 1
            if res and (done % 10 == 0 or done <= 3):
                el = time.perf_counter() - t0
                print("[sweep] %d/%d  %-40s val %.4f test %.4f  %.1fs  | %.1f s/run, eta %.1f h"
                      % (done, len(todo), res["run"], res["val_macro"], res["test_macro"], res["wall_s"],
                         el / done, el / done * (len(todo) - done) / 3600), flush=True)
        if not failed:
            break
    sel = select()
    print("[sweep] %s finished %d runs, %d failed; %d selected rows -> results/frozen_selected.csv"
          % (now(), done, len(failed), len(sel)))
    for f in failed:
        print("  FAILED", run_name(f[1], f[0], *f[2:]))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
