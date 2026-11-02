"""T5.4-T5.7: paired comparisons, mixed model, sensitivity checks, finding heterogeneity,
gold-58 agreement and site-shift analysis.

Paired comparison of two configurations A, B at one training size N:
  * unit = one (rotation, seed) cell where both were trained on the SAME subset and scored on
    the SAME test fold; delta_u = AUC_B - AUC_A.
  * bootstrap replicate: resample units with replacement, and replace each unit's delta by
    its study-level paired-bootstrap draw (bootcache: both runs evaluated on the same
    resampled studies, so shared difficulty cancels -- stage4_compare.boot_delta's idea,
    vectorized). The estimate is the mean delta over units; 95% CI = percentiles.
  * two-sided bootstrap p = 2 * min(P(delta* <= 0), P(delta* >= 0)); Holm across the 12
    findings within each (comparison, N).
Comparisons C1-C4 are the pre-registered ones (PROJECT_PLAN 5.10); "X" rows are exploratory.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import polars as pl
import statsmodels.formula.api as smf
from scipy.stats import kendalltau, spearmanr

import bootcache
from collect_results import split_labels
from common import (FINDING_GROUPS, KEYS, LABELS, RESULTS, RUNS, SIZES, auc_counts,
                    load_rotation, load_series_index, load_targets)

warnings.filterwarnings("ignore")
ROWS = bootcache.ROWS
NB = 2000
DESIGN = {"matched": ([0, 1, 2], [0]), "frozen_full": ([0, 1, 2, 3, 4], [0, 1, 2])}
COMPARISONS = [
    ("C1", "matched", ("frozen", "M1"), ("ft", "M1"), "Fine-tuned vs frozen, DINOv3 ViT-B"),
    ("C1", "matched", ("frozen", "M2"), ("ft", "M2"), "Fine-tuned vs frozen, DINOv3 ConvNeXt-B"),
    ("C2", "matched", ("ft", "M1"), ("ft", "M2"), "ConvNeXt-B minus ViT-B, fine-tuned"),
    ("C2", "frozen_full", ("frozen", "M1"), ("frozen", "M2"), "ConvNeXt-B minus ViT-B, frozen"),
    ("C3", "frozen_full", ("frozen", "M2"), ("frozen", "M3"), "ImageNet minus DINOv3 pretraining (ConvNeXt-B, frozen)"),
    ("C4", "frozen_full", ("frozen", "M4"), ("frozen", "M1"), "DINOv3 minus DINOv2 (ViT-B, frozen)"),
    ("X", "matched", ("frozen", "M3"), ("ft", "M2"), "Fine-tuned DINOv3 ConvNeXt-B minus best frozen (ImageNet ConvNeXt-B)"),
    ("X", "matched", ("frozen", "M3"), ("ft", "M1"), "Fine-tuned DINOv3 ViT-B minus best frozen (ImageNet ConvNeXt-B)"),
]


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def boot_p(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean()))) if x.size else np.nan


def main() -> int:
    rp = pl.read_parquet(RESULTS / "runs.parquet")
    sel = rp.filter(pl.col("selected"))
    test = sel.filter(pl.col("split") == "test")
    dr, dg = bootcache.draws("test"), bootcache.draws("gold")
    auc = {(r, f): a for r, f, a in test.select("run", "finding", "auc").iter_rows()}
    gauc = {(r, f): a for r, f, a in sel.filter(pl.col("split") == "gold")
            .select("run", "finding", "auc").iter_rows()}
    runid = {(rg, m, r, s, n): run for run, rg, m, r, s, n in
             test.filter(pl.col("finding") == "macro")
             .select("run", "regime", "model", "rotation", "seed", "n_train").iter_rows()}
    rng = np.random.default_rng(7)
    T = RESULTS / "tables"

    # ------------------------------------------------ T5.4 paired comparisons
    rows = []
    for cid, design, A, B, label in COMPARISONS:
        rots, seeds = DESIGN[design]
        units = [(r, s) for r in rots for s in seeds]
        pu = rng.integers(0, len(units), (NB, len(units)))                 # resampled units
        pd_ = rng.integers(0, bootcache.B, (NB, 5))[np.arange(NB)[:, None],   # one study-level
                                                   np.array([u[0] for u in units])[pu]]  # draw per rotation
        for n in SIZES:
            ra = [runid[(*A, r, s, n)] for r, s in units]
            rb = [runid[(*B, r, s, n)] for r, s in units]
            Dd = (np.stack([dr[b] for b in rb]) - np.stack([dr[a] for a in ra])).transpose(1, 0, 2)
            REP = np.nanmean(Dd[:, pu, pd_], axis=2)                          # (13, NB)
            res = []
            for i, f in enumerate(ROWS):
                du = np.array([auc[(b, f)] - auc[(a, f)] for a, b in zip(ra, rb)])
                rep = REP[i]
                res.append(dict(id=cid, comparison=label, design=design, A="%s %s" % A, B="%s %s" % B,
                                n_train=n, finding=f, k_units=len(units), delta=float(np.nanmean(du)),
                                lo=float(np.nanpercentile(rep, 2.5)), hi=float(np.nanpercentile(rep, 97.5)),
                                p=boot_p(rep), units_B_better=int((du > 0).sum())))
            ph = holm([r["p"] for r in res[:12]])
            for r, h in zip(res, list(ph) + [np.nan]):
                r["p_holm"] = h
            rows += res
        print("[stats] %s %s done" % (cid, label), flush=True)
    pdl = pl.DataFrame(rows)
    pdl.write_csv(T / "paired_deltas.csv")

    # ------------------------------------------------ T5.4 mixed model (M1/M2 x frozen/FT)
    mm = (test.filter((pl.col("finding") == "macro") & pl.col("model").is_in(["M1", "M2"])
                      & pl.col("regime").is_in(["frozen", "ft"]))
          .select("auc", "model", "regime", "rotation", "seed", "n_train")
          .with_columns(pl.col("n_train").log(2).alias("log2n")).to_pandas())
    mm["log2n_c"] = mm["log2n"] - np.log2(520)          # centre at N = 520
    mod = smf.mixedlm("auc ~ C(model, Treatment('M1')) * C(regime, Treatment('frozen')) * log2n_c",
                      mm, groups=mm["rotation"], re_formula="1",
                      vc_formula={"seed": "0 + C(seed)"}).fit(reml=True)
    co = mod.summary().tables[1]
    co.to_csv(T / "mixed_model.csv")
    print(co)

    # ------------------------------------------------ T5.4 sensitivity (a): fallback_R studies
    se = load_series_index()
    fb = set(se.filter(pl.col("side_src") == "fallback_R")["study_dir"].unique().to_list())
    tg = load_targets()
    lat = ["MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA"]
    sens = []
    for (rg, m, r, s, n), run in runid.items():
        if n != 2600 or s != 0 or r > 2 or m == "M5":
            continue
        ids = sorted(load_rotation(r)["test"])
        y, msk = split_labels(tg, ids, "test")
        pr = pl.read_parquet(RUNS / run / "preds_test.parquet")
        pos = {sid: i for i, sid in enumerate(ids)}
        P = np.empty((len(ids), 12))
        P[[pos[x] for x in pr["study_dir"].to_list()]] = pr.select(KEYS).to_numpy()
        keep = np.array([sid not in fb for sid in ids])
        for f in lat:
            j = LABELS.index(f)
            mk = msk[:, j] & keep
            a2 = auc_counts(y[mk, j], P[mk, j], np.ones((1, mk.sum())))[0]
            sens.append(dict(regime=rg, model=m, rotation=r, finding=f, auc_all=auc[(run, f)],
                             auc_excl_fallback=a2, n_excluded=int((msk[:, j] & ~keep).sum())))
    sens = pl.DataFrame(sens).with_columns((pl.col("auc_excl_fallback") - pl.col("auc_all")).alias("change"))
    sens.write_csv(T / "sensitivity_fallbackR.csv")

    # ------------------------------------------------ T5.4 sensitivity (b): findings with >=10 train positives
    pf = test.filter(pl.col("finding") != "macro")
    restricted = (pf.filter(pl.col("n_train_pos") >= 10).group_by("run").agg(
        pl.col("auc").mean().alias("macro_ge10"), pl.len().alias("n_findings")))
    rmac = test.filter(pl.col("finding") == "macro").join(restricted, on="run")
    rsum = (rmac.filter(pl.col("rotation").is_in([0, 1, 2]) & (pl.col("seed") == 0))
            .group_by("regime", "model", "n_train").agg(
                pl.col("auc").mean().alias("macro_all12"), pl.col("macro_ge10").mean(),
                pl.col("n_findings").mean().alias("findings_included"))
            .sort("regime", "model", "n_train"))
    rsum.write_csv(T / "sensitivity_macro_ge10pos.csv")

    # ------------------------------------------------ T5.5 finding heterogeneity
    grp = {lab: g for g, labs in FINDING_GROUPS.items() for lab in labs}
    het = (pf.filter(pl.col("regime").is_in(["frozen", "ft"]))
           .with_columns(pl.col("finding").replace_strict(grp).alias("group"),
                         (pl.col("n_train_pos") + 1).log(2).alias("log2pos"),
                         pl.col("n_train").log(2).alias("log2n"),
                         (pl.col("regime") + "_" + pl.col("model")).alias("cfg"))
           .drop_nulls("auc").to_pandas())
    het_rows = []
    for name, formula in [("A: by training-set size", "auc ~ log2n + C(group, Treatment('fluid_marrow_soft')) + C(cfg)"),
                          ("B: by training positives", "auc ~ log2pos + C(group, Treatment('fluid_marrow_soft')) + C(cfg)")]:
        fit = smf.ols(formula, het).fit(cov_type="cluster", cov_kwds={"groups": het["run"].astype("category").cat.codes})
        for term in fit.params.index:
            if "group" in term or term in ("log2n", "log2pos"):
                het_rows.append(dict(model=name, term=term, coef=fit.params[term],
                                     lo=fit.conf_int().loc[term, 0], hi=fit.conf_int().loc[term, 1],
                                     p=fit.pvalues[term], r2=fit.rsquared))
    pl.DataFrame(het_rows).write_csv(T / "finding_heterogeneity.csv")

    # ------------------------------------------------ T5.6 gold-58 agreement
    g26 = []
    for (rg, m, r, s, n), run in runid.items():
        if s == 0 and r <= 2:
            g26.append(dict(regime=rg, model=m, rotation=r, n_train=n, test=auc[(run, "macro")],
                            gold=gauc[(run, "macro")], run=run))
    g26 = pl.DataFrame(g26)
    cells = g26.group_by("regime", "model", "n_train").agg(pl.col("test").mean(), pl.col("gold").mean())
    rho_all = spearmanr(cells["test"], cells["gold"])
    at = cells.filter(pl.col("n_train") == 2600).sort("test", descending=True)
    rho_26 = spearmanr(at["test"], at["gold"])
    gold_rows = []
    for (rg, m), sub in g26.filter(pl.col("n_train") == 2600).group_by("regime", "model"):
        runs = sub["run"].to_list()
        for i, f in enumerate(ROWS):
            d = np.nanmean(np.stack([dg[x][i] for x in runs]), 0)
            gold_rows.append(dict(regime=rg, model=m, finding=f, k_runs=len(runs),
                                  gold_auc=float(np.mean([gauc[(x, f)] for x in runs])),
                                  lo=float(np.nanpercentile(d, 2.5)), hi=float(np.nanpercentile(d, 97.5)),
                                  test_auc=float(np.mean([auc[(x, f)] for x in runs]))))
    gold_tab = pl.DataFrame(gold_rows).sort("regime", "model")
    gold_tab.write_csv(T / "gold58_n2600.csv")
    # gold paired deltas at N = 2600 for C1-C3 (same 58 studies, paired draws)
    gd_rows = []
    for cid, design, A, B, label in COMPARISONS:
        rots, seeds = DESIGN[design]
        units = [(r, s) for r in rots for s in seeds]
        ra = [runid[(*A, r, s, 2600)] for r, s in units]
        rb = [runid[(*B, r, s, 2600)] for r, s in units]
        i = ROWS.index("macro")
        d = np.mean([dg[b][i] - dg[a][i] for a, b in zip(ra, rb)], 0)
        gd_rows.append(dict(id=cid, comparison=label, gold_delta=float(np.mean([gauc[(b, "macro")] - gauc[(a, "macro")] for a, b in zip(ra, rb)])),
                            lo=float(np.nanpercentile(d, 2.5)), hi=float(np.nanpercentile(d, 97.5)),
                            test_delta=float(pdl.filter((pl.col("comparison") == label) & (pl.col("n_train") == 2600)
                                                        & (pl.col("finding") == "macro"))["delta"][0])))
    pl.DataFrame(gd_rows).write_csv(T / "gold58_deltas_n2600.csv")
    pl.DataFrame([dict(scope="all (regime, model, N) cells, matched design", n_cells=cells.height,
                       spearman=rho_all.statistic, p=rho_all.pvalue),
                  dict(scope="7 configurations at N = 2600", n_cells=at.height,
                       spearman=rho_26.statistic, p=rho_26.pvalue)]).write_csv(T / "gold58_rank_agreement.csv")

    # ------------------------------------------------ T5.7 site shift
    # (a) ranking stability of the four frozen backbones across the 5 test folds (N = 2600, seed-mean)
    fr = (test.filter((pl.col("finding") == "macro") & (pl.col("regime") == "frozen") & (pl.col("n_train") == 2600))
          .group_by("rotation", "model").agg(pl.col("auc").mean()).sort("rotation", "model"))
    mat = fr.pivot(on="model", index="rotation", values="auc").sort("rotation")
    R = np.vstack([np.argsort(np.argsort(-mat.select(["M1", "M2", "M3", "M4"]).to_numpy()[i])) + 1
                   for i in range(mat.height)])
    m_, k_ = R.shape
    Ssum = ((R.sum(0) - R.sum(0).mean()) ** 2).sum()
    W = 12 * Ssum / (m_ ** 2 * (k_ ** 3 - k_))
    mat.write_csv(T / "rotation_frozen_n2600.csv")
    # (b) per-vendor macro AUC inside each test fold, vendors with >= 100 studies there
    vend = []
    for (rg, m, r, s, n), run in runid.items():
        if n != 2600 or s != 0 or r > 2 or m == "M5":
            continue
        ids = sorted(load_rotation(r)["test"])
        y, msk = split_labels(tg, ids, "test")
        v = pl.DataFrame({"study_dir": ids}).join(tg.select("study_dir", "vendor"), on="study_dir", how="left")["vendor"].to_numpy()
        pr = pl.read_parquet(RUNS / run / "preds_test.parquet")
        pos = {sid: i for i, sid in enumerate(ids)}
        P = np.empty((len(ids), 12))
        P[[pos[x] for x in pr["study_dir"].to_list()]] = pr.select(KEYS).to_numpy()
        for vn in np.unique(v):
            sv = v == vn
            if sv.sum() < 100:
                continue
            per = []
            for j in range(12):
                mk = msk[:, j] & sv
                yy = y[mk, j]
                if mk.sum() >= 8 and 0 < yy.sum() < mk.sum():
                    per.append(auc_counts(yy, P[mk, j], np.ones((1, mk.sum())))[0])
            vend.append(dict(regime=rg, model=m, rotation=r, vendor=vn, n_studies=int(sv.sum()),
                             macro=float(np.mean(per)), n_findings=len(per)))
    vend = pl.DataFrame(vend).sort("rotation", "vendor", "regime", "model")
    vend.write_csv(T / "vendor_auc_n2600.csv")

    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=220, float_precision=3):
        print(pdl.filter(pl.col("finding") == "macro").select("id", "comparison", "n_train", "delta", "lo", "hi", "p", "units_B_better"))
        print(sens.group_by("regime", "model").agg(pl.col("change").mean(), pl.col("change").abs().max().alias("max_abs")))
        print(rsum.filter(pl.col("n_train").is_in([65, 260, 2600])))
        print(pl.DataFrame(het_rows))
        print(pl.read_csv(T / "gold58_rank_agreement.csv"))
        print(pl.DataFrame(gd_rows))
        print("Kendall W (frozen backbones ranked across 5 test folds, N=2600): %.3f" % W)
        print(mat)
        print(vend.group_by("rotation", "vendor", "n_studies").agg(pl.col("macro").min().alias("min"), pl.col("macro").max().alias("max")).sort("rotation", "vendor"))
    (T / "site_kendall_w.txt").write_text("Kendall W = %.4f over %d test folds x %d frozen backbones (N=2600)\n" % (W, m_, k_))
    return 0


if __name__ == "__main__":
    sys.exit(main())
