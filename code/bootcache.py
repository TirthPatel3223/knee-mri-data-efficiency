"""Study-level bootstrap draws of every selected run's AUCs, computed once and cached.

fit_curves.py (uncertainty of fitted curves) and stats_tests.py (paired model-vs-model
deltas) both need, for each selected run, its macro and per-finding AUC on each bootstrap
resample of its test fold. Computing them is ~1 s per run, so they are cached in
results/cache/boot_<split>.npz as {run: float32 (13, B)} -- rows = 12 findings + macro.

The resample matrices are exactly collect_results.py's (common.boot_counts with seed
1000 + rotation for a test fold, 1000 for gold), so draw b of every run on the same split
uses the same resampled studies. That is what makes cross-run deltas *paired*.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from common import KEYS, LABELS, RESULTS, RUNS, auc_counts, boot_counts, load_rotation, load_targets
from collect_results import split_labels

B = 2000
CACHE = RESULTS / "cache"
ROWS = LABELS + ["macro"]


def _compute(split: str, runs: pl.DataFrame) -> dict[str, np.ndarray]:
    tg = load_targets()
    gold_ids = sorted(tg.filter(pl.col("is_gold"))["study_dir"].to_list())
    info: dict[int, tuple] = {}
    out = {}
    for run, r in runs.select("run", "rotation").unique().sort("run").iter_rows():
        key = -1 if split == "gold" else r
        if key not in info:
            ids = gold_ids if split == "gold" else sorted(load_rotation(r)["test"])
            y, m = split_labels(tg, ids, split)
            info[key] = (ids, {s: i for i, s in enumerate(ids)}, y, m,
                         boot_counts(len(ids), B, seed=1000 + max(key, 0)))
        ids, pos, y, m, cnt = info[key]
        pr = pl.read_parquet(RUNS / run / ("preds_%s.parquet" % split))
        P = np.empty((len(ids), len(KEYS)))
        P[[pos[s] for s in pr["study_dir"].to_list()]] = pr.select(KEYS).to_numpy()
        d = np.stack([auc_counts(y[m[:, j], j], P[m[:, j], j], cnt[:, m[:, j]])
                      for j in range(len(KEYS))])
        out[run] = np.vstack([d, np.nanmean(d, 0)]).astype(np.float32)
    return out


def draws(split: str = "test") -> dict[str, np.ndarray]:
    """{run: (13, B)} for every selected run that was scored on `split`."""
    f = CACHE / ("boot_%s.npz" % split)
    runs = (pl.read_parquet(RESULTS / "runs.parquet")
            .filter(pl.col("selected") & (pl.col("split") == split))
            .select("run", "rotation").unique())
    if f.is_file():
        z = dict(np.load(f))
        if set(z) >= set(runs["run"].to_list()):
            return z
    CACHE.mkdir(parents=True, exist_ok=True)
    z = _compute(split, runs)
    np.savez_compressed(f, **z)
    return z


if __name__ == "__main__":
    import time
    for s in ("test", "gold"):
        t = time.time()
        z = draws(s)
        print("[bootcache] %s: %d runs x %s in %.0f s" % (s, len(z), next(iter(z.values())).shape, time.time() - t))
