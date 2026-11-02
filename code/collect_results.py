"""T5.1: collect every run into one long table, results/runs.parquet.

One row per (run x split x finding), plus a "macro" row per (run x split). Every later
analysis (curves, fits, tests, cost) reads this file, never the run directories, so
they all see the same numbers.

What it adds beyond copying metrics.json:
  * `selected`: frozen runs -> the HP picked on the val fold (frozen_selected.csv);
    FT runs -> the pilot-chosen LR (ft_lr_choice.json); the 6 pilot runs at other LRs
    and the 1,260 unselected frozen HP runs stay in the table with selected=False.
  * a recompute of every selected run's test/gold AUCs from its saved predictions.
    A mismatch with metrics.json means predictions were scored against the wrong
    studies (the 2026-10-26 plan-cache bug) and aborts the collection.
  * study-level bootstrap 95% CIs (2,000 resamples) for selected runs. All runs on a
    split share one resample matrix (common.boot_counts), so the CIs are consistent
    with the paired deltas computed later in stats_tests.py.
  * n_pos / n_neg of the scored split and the training positives per finding.

Regimes: "frozen", "ft" (pretrained, fine-tuned) and "scratch" (M5/M6, random init).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import polars as pl

from common import (LABELS, KEYS, MODELS, RESULTS, RUNS, auc_counts, boot_counts,
                    load_rotation, load_targets, now)

SPLITS = ("test", "gold")


def split_labels(tg: pl.DataFrame, ids: list[str], split: str):
    """(y, mask) per finding, row-aligned with ids. Test = proxy label over studies
    whose report addresses the finding; gold = expert label over all 58."""
    t = pl.DataFrame({"study_dir": ids}).join(tg, on="study_dir", how="left")
    if t["is_gold"].null_count():
        raise SystemExit("[collect] prediction ids missing from targets.parquet")
    if split == "gold":
        y = np.stack([t["g_" + k].to_numpy() == 1 for k in KEYS], 1)
        m = np.ones_like(y, dtype=bool)
    else:
        y = np.stack([(t["s_" + k] == "pos").to_numpy() for k in KEYS], 1)
        m = np.stack([t["s_" + k].is_in(["pos", "neg"]).to_numpy() for k in KEYS], 1)
    return y, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tol", type=float, default=1e-6, help="max |recomputed - metrics.json| AUC")
    a = ap.parse_args()

    tg = load_targets()
    sel_frozen = set(pl.read_csv(RESULTS / "frozen_selected.csv")["run"].to_list())
    ft_lr = {m: v["lr"] for m, v in json.loads((RESULTS / "ft_lr_choice.json").read_text()).items()}

    # Split-level label arrays and bootstrap counts are the same for every run on a
    # rotation's test fold (and for gold), so build them once and key by study order.
    gold_ids = sorted(tg.filter(pl.col("is_gold"))["study_dir"].to_list())
    split_cache: dict[tuple, dict] = {}

    def split_info(split: str, r: int) -> dict:
        key = ("gold", -1) if split == "gold" else ("test", r)
        if key not in split_cache:
            ids = gold_ids if split == "gold" else sorted(load_rotation(r)["test"])
            y, m = split_labels(tg, ids, split)
            split_cache[key] = dict(ids=ids, pos={s: i for i, s in enumerate(ids)}, y=y, m=m,
                                    cnt=boot_counts(len(ids), a.boot, seed=1000 + max(r, 0)))
        return split_cache[key]

    rows, n_runs, n_ci, bad = [], 0, 0, []
    dirs = sorted(d for d in RUNS.iterdir() if (d / "metrics.json").is_file())
    print("[collect] %s  %d run dirs" % (now(), len(dirs)), flush=True)
    for d in dirs:
        mt = json.loads((d / "metrics.json").read_text())
        model, n = mt["model"], mt["n_train"]
        regime = "scratch" if MODELS[model].get("scratch") else mt["regime"]
        if regime == "frozen":
            selected = mt["run"] in sel_frozen
        elif regime == "ft":
            selected = abs(mt["hp"]["lr"] - ft_lr[model]) < 1e-12
        else:
            selected = True
        base = dict(run=mt["run"], regime=regime, model=model, rotation=mt["rotation"],
                    seed=mt["seed"], n_train=n, hp_id=mt["run"].split("-n%04d-" % n)[1],
                    selected=selected, gpu_h=mt.get("gpu_h"), kwh=mt.get("kwh"))
        rows.append(dict(base, split="val", finding="macro", auc=mt["val_macro"]))
        n_runs += 1
        if mt.get("test_macro") is None:            # LR-pilot runs were scored on val only
            continue
        for split in SPLITS:
            si = split_info(split, mt["rotation"])
            per = mt["%s_per_finding" % split]
            lo = hi = {}
            if selected:
                pr = pl.read_parquet(d / ("preds_%s.parquet" % split))
                if sorted(pr["study_dir"].to_list()) != si["ids"]:
                    bad.append((mt["run"], split, "study set differs from the split"))
                    continue
                order = np.array([si["pos"][s] for s in pr["study_dir"].to_list()])
                P = np.empty((len(order), len(KEYS)))
                P[order] = pr.select(KEYS).to_numpy()
                boots, lo, hi = [], {}, {}
                for j, lab in enumerate(LABELS):
                    mk = si["m"][:, j]
                    yy, pp = si["y"][mk, j], P[mk, j]
                    full = auc_counts(yy, pp, np.ones((1, mk.sum())))[0]
                    if np.isfinite(per[lab]) and abs(full - per[lab]) > a.tol:
                        bad.append((mt["run"], split, "%s recomputed %.6f vs %.6f" % (lab, full, per[lab])))
                    bt = auc_counts(yy, pp, si["cnt"][:, mk])
                    boots.append(bt)
                    lo[lab], hi[lab] = np.nanpercentile(bt, 2.5), np.nanpercentile(bt, 97.5)
                macro_bt = np.nanmean(np.stack(boots), 0)
                lo["macro"], hi["macro"] = np.nanpercentile(macro_bt, 2.5), np.nanpercentile(macro_bt, 97.5)
                n_ci += 1
            for j, lab in enumerate(LABELS):
                mk = si["m"][:, j]
                npos = int(si["y"][mk, j].sum())
                rows.append(dict(base, split=split, finding=lab, auc=per[lab],
                                 auc_lo=lo.get(lab), auc_hi=hi.get(lab), n_pos=npos,
                                 n_neg=int(mk.sum()) - npos,
                                 n_train_pos=mt["train_n_pos"][lab]))
            rows.append(dict(base, split=split, finding="macro", auc=mt["%s_macro" % split],
                             auc_lo=lo.get("macro"), auc_hi=hi.get("macro"),
                             n_pos=None, n_neg=len(si["ids"]), n_train_pos=None))
        if n_runs % 250 == 0:
            print("[collect] %s  %d runs" % (now(), n_runs), flush=True)

    if bad:
        print("[collect] %d CONSISTENCY FAILURES (not written):" % len(bad))
        for b in bad[:20]:
            print("   ", b)
        return 1

    df = pl.DataFrame(rows, infer_schema_length=None).with_columns(
        pl.col("auc", "auc_lo", "auc_hi", "gpu_h", "kwh").cast(pl.Float64))
    out = RESULTS / "runs.parquet"
    df.write_parquet(out)
    print("[collect] %s  wrote %s: %d rows, %d runs, %d with CIs; predictions match metrics.json"
          % (now(), out, df.height, n_runs, n_ci))

    # Quick look: selected runs, mean test macro by (regime, model) x N.
    s = (df.filter(pl.col("selected") & (pl.col("split") == "test") & (pl.col("finding") == "macro"))
         .group_by("regime", "model", "n_train").agg(pl.col("auc").mean().round(3), pl.len().alias("k"))
         .sort("n_train").pivot(on="n_train", index=["regime", "model"], values="auc")
         .sort("regime", "model"))
    with pl.Config(tbl_rows=30, tbl_cols=12):
        print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
