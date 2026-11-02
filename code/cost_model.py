"""T5.8: what does reaching a target performance cost a hospital? (RQ5)

Exams needed come from the learning curves (fit_curves.n_to_target, matched design): read off
the observed curve inside 65-2,600 exams, log-linear extrapolation beyond (flagged), and not
reported beyond 10x the data. Three
cost lines per exam count:
  * compute -- MEASURED: GPU wall-hours of our own runs on an RTX 5070 Ti laptop GPU.
      FT: gpu_h of the run at that N (log-log interpolation between the 7 measured sizes).
      frozen: embedding the training + validation studies once (measured 0.40 h for all
      4,407 studies) + the full 4-config head grid (sum of all 4 runs, seconds each).
      Priced at an ASSUMED cloud rate, central $0.59/h (RunPod L4, secure cloud), range
      $0.34-0.89/h (RTX 4090 community / secure), RunPod pricing page updated 2026-10-08.
  * manual expert labels -- ASSUMED: reading time per knee MRI, median 8 min (IQR 4.6-11.4)
      from a survey of 144 radiologists (Kwee et al., Tomography 2024;10(9):1527-33),
      range 5-20 min for 12 structured findings; x2 readers (double read, the host's gold
      protocol also adds an adjudicator), range x1-x3; radiologist wage $170.17/h (BLS
      OEWS May 2023 mean, SOC 29-1224), range $115-250/h.
  * LLM report labels -- MEASURED: local qwen3:14b read 4,407 reports in 8.4 GPU-h
      (6.9 s/report), priced at the same cloud rate. Needs no radiologist time, but gives
      the noisier labels this study trained on (gold-58 macro 0.866).
Every assumption is a named constant below, so the sensitivity ranges are explicit.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import polars as pl

from common import RESULTS

GPU_RATE = {"low": 0.34, "central": 0.59, "high": 0.89}          # $/GPU-h, RunPod 2026-10-08
READ_MIN = {"low": 5.0, "central": 8.0, "high": 20.0}            # min per exam, Kwee 2024
READERS = {"low": 1.0, "central": 2.0, "high": 3.0}              # double read central
WAGE = {"low": 115.0, "central": 170.17, "high": 250.0}          # $/h, BLS OEWS May 2023
EMBED_H_PER_STUDY = (23.9 + 23.6) / 2 / 60 / 4407                # measured T2.1/T2.2
LLM_H_PER_STUDY = 8.4 / 4407                                     # measured, FINDINGS Stage 2
VAL_STUDIES = 870
TARGETS = [("AUC 0.75", "N_tau0.75"), ("AUC 0.80", "N_tau0.80"), ("AUC 0.85", "N_tau0.85")]
MAX_EXTRAP = 10 * 2600       # beyond 10x the largest training set an estimate is not reported


def manual_cost(n, lvl="central"):
    return n * READ_MIN[lvl] / 60 * READERS[lvl] * WAGE[lvl]


def main() -> int:
    rp = pl.read_parquet(RESULTS / "runs.parquet").filter(
        (pl.col("split") == "test") & (pl.col("finding") == "macro"))
    # measured GPU-hours per (regime, model, N), matched design
    m = rp.filter(pl.col("rotation").is_in([0, 1, 2]) & (pl.col("seed") == 0))
    ft = (m.filter((pl.col("regime") == "ft") & pl.col("selected"))
          .group_by("model", "n_train").agg(pl.col("gpu_h").mean(), pl.col("kwh").mean()))
    fr = (m.filter(pl.col("regime") == "frozen")              # all 4 grid configs, not just the pick
          .group_by("model", "n_train", "rotation").agg(pl.col("gpu_h").sum())
          .group_by("model", "n_train").agg(pl.col("gpu_h").mean()))
    fr = fr.with_columns((pl.col("gpu_h") + (pl.col("n_train") + VAL_STUDIES) * EMBED_H_PER_STUDY)
                         .alias("gpu_h"), pl.lit(None, dtype=pl.Float64).alias("kwh"))
    gh = pl.concat([ft.with_columns(pl.lit("ft").alias("regime")),
                    fr.with_columns(pl.lit("frozen").alias("regime"))], how="diagonal_relaxed")
    gh.sort("regime", "model", "n_train").write_csv(RESULTS / "tables" / "gpu_hours_by_n.csv")

    def gpu_at(regime, model, n):
        g = gh.filter((pl.col("regime") == regime) & (pl.col("model") == model)).sort("n_train")
        x, y = np.log(g["n_train"].to_numpy()), np.log(g["gpu_h"].to_numpy())
        lx = np.log(n)
        if lx <= x[-1]:
            return float(np.exp(np.interp(lx, x, y)))
        k = (y[-1] - y[-3]) / (x[-1] - x[-3])                    # log-log slope of the top end
        return float(np.exp(y[-1] + k * (lx - x[-1])))

    fits = json.loads((RESULTS / "fits" / "fits_macro.json").read_text())
    rows = []
    for key, f in fits.items():
        design, regime, model = key.split("|")
        if design != "matched":
            continue
        for tname, k in TARGETS:
            n, lo, hi = f[k], f[k + "_lo"], f[k + "_hi"]
            reach = bool(np.isfinite(n) and n <= MAX_EXTRAP)
            row = dict(regime=regime, model=model, target=tname, auc_at_2600=f["auc_at_2600"],
                       exams=n, exams_lo=lo, exams_hi=hi, reachable=reach,
                       extrapolated=bool(np.isfinite(n) and n > 2600))
            if reach:
                g = gpu_at(regime, model, n)
                row.update(gpu_h=g, compute_usd=g * GPU_RATE["central"],
                           compute_usd_lo=g * GPU_RATE["low"], compute_usd_hi=g * GPU_RATE["high"],
                           label_manual_usd=manual_cost(n), label_manual_usd_lo=manual_cost(n, "low"),
                           label_manual_usd_hi=manual_cost(n, "high"),
                           label_llm_usd=n * LLM_H_PER_STUDY * GPU_RATE["central"])
            rows.append(row)
    t5 = pl.DataFrame(rows, infer_schema_length=None)
    t5.write_csv(RESULTS / "tables" / "table5_cost.csv")

    # sensitivity grid: manual labelling cost of 1,000 exams
    grid = [dict(minutes=mi, readers=r, wage=w, usd_per_1000_exams=1000 * mi / 60 * r * w)
            for mi in (5.0, 8.0, 11.4, 20.0) for r in (1, 2, 3) for w in (115.0, 170.17, 250.0)]
    pl.DataFrame(grid).write_csv(RESULTS / "tables" / "labeling_cost_sensitivity.csv")

    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=220, float_precision=2):
        print(gh.sort("regime", "model", "n_train"))
        print(t5.select("regime", "model", "target", "auc_at_2600", "exams", "exams_lo", "exams_hi",
                        "extrapolated", "gpu_h", "compute_usd", "label_manual_usd", "label_llm_usd"))
        print("manual labels, central: $%.0f per 1,000 exams; LLM labels: $%.2f per 1,000 exams"
              % (manual_cost(1000), 1000 * LLM_H_PER_STUDY * GPU_RATE["central"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
