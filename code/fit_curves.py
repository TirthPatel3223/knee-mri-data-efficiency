"""T5.2 + T5.3: learning-curve table and learning-curve fits with sample-size estimates.

Two designs, so that every comparison is like-for-like:
  * "matched"     rotations 0-2, seed 0 -- the exact training subsets and test folds the
                  fine-tuning sweep used. All frozen-vs-FT and M1-vs-M2 statements use it.
  * "frozen_full" rotations 0-4 x seeds 0-2 -- the full frozen design (15 runs per N), for
                  the frozen-only questions (C3 M2 vs M3, C4 M1 vs M4) at higher precision.

Curve model (pre-registered, PROJECT_PLAN 5.11): inverse power law
    AUC(N) = a - b * N^(-c),   a in [0.5, 1], b > 0, c in (0, 2],
fitted by least squares on run-level points. Derived quantities:
    N_eps = (b / eps)^(1/c)          exams to get within eps of the asymptote a
    N_tau = (b / (a - tau))^(1/c)    exams to reach a target AUC tau (only if tau < a)
Values above N = 2,600 are extrapolations and are flagged.

Uncertainty -- a deviation from the plan, logged in PROJECT_PLAN 15: the plan resampled
rotation x seed units only. FT has 3 units, i.e. only 10 distinct resamples, which gives
degenerate intervals. Each replicate therefore (1) resamples units with replacement AND
(2) replaces each unit's AUCs with one study-level bootstrap draw of its test fold
(bootcache; one draw index per rotation, shared by all N and all models). So intervals
carry both across-site variation and test-set sampling noise, and because every curve in
a replicate uses the same units and draws, ratios between curves are paired.

Model-form check (H5): the power law is compared with a log-linear and an exponential-
saturation curve by AIC on the same points.
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

import bootcache
from common import LABELS, RESULTS

warnings.filterwarnings("ignore")
ROWS = bootcache.ROWS                      # 12 findings + "macro"
NB = 1000
EPS = (0.01, 0.02)
TAUS = (0.75, 0.80, 0.85)
NMAX = 2600

DESIGNS = {
    "matched": dict(rot=[0, 1, 2], seeds=[0],
                    curves=[("frozen", m) for m in ("M1", "M2", "M3", "M4")] + [("ft", "M1"), ("ft", "M2")]),
    "frozen_full": dict(rot=[0, 1, 2, 3, 4], seeds=[0, 1, 2],
                        curves=[("frozen", m) for m in ("M1", "M2", "M3", "M4")]),
}
# Paired efficiency ratios N_tau(A) / N_tau(B): > 1 means B needs fewer exams than A.
RATIOS = [
    ("matched", ("frozen", "M1"), ("ft", "M1"), "C1 fine-tuning vs frozen, ViT-B"),
    ("matched", ("frozen", "M2"), ("ft", "M2"), "C1 fine-tuning vs frozen, ConvNeXt-B"),
    ("matched", ("ft", "M1"), ("ft", "M2"), "C2 ConvNeXt-B vs ViT-B, fine-tuned"),
    ("frozen_full", ("frozen", "M1"), ("frozen", "M2"), "C2 ConvNeXt-B vs ViT-B, frozen"),
    ("frozen_full", ("frozen", "M2"), ("frozen", "M3"), "C3 ImageNet vs DINOv3 ConvNeXt-B, frozen"),
    ("frozen_full", ("frozen", "M4"), ("frozen", "M1"), "C4 DINOv3 vs DINOv2 ViT-B, frozen"),
    ("matched", ("frozen", "M3"), ("ft", "M2"), "exploratory: FT DINOv3 ConvNeXt-B vs best frozen"),
]


# ------------------------------------------------------------------ curve forms
def ipl(n, a, b, c):
    return a - b * np.power(n, -c)


def expsat(n, a, b, k):
    return a - b * np.exp(-n / k)


def fit_ipl(n, y, p0=None):
    """Least-squares inverse power law; multi-start because c and b trade off strongly."""
    starts = [p0] if p0 is not None else []
    starts += [(min(0.99, y.max() + 0.03), 1.0, c0) for c0 in (0.15, 0.3, 0.6, 1.0)]
    best = None
    for s in starts:
        try:
            p, _ = curve_fit(ipl, n, y, p0=s, bounds=([0.5, 1e-6, 1e-3], [1.0, 50.0, 2.0]),
                             maxfev=5000)
        except (RuntimeError, ValueError):
            continue
        rss = float(np.sum((ipl(n, *p) - y) ** 2))
        if best is None or rss < best[1]:
            best = (p, rss)
    return best


def derived(p) -> dict:
    a, b, c = p
    d = {"a": a, "b": b, "c": c}
    for e in EPS:
        d["N_eps%.2f" % e] = (b / e) ** (1 / c)
    for t in TAUS:
        d["N_tau%.2f" % t] = (b / (a - t)) ** (1 / c) if a > t else np.inf
    return d


def loglin(n, y):
    """AUC = alpha + beta * log2(N); beta = AUC gained per doubling of the training set."""
    m = np.isfinite(y)
    beta, alpha = np.polyfit(np.log2(n[m]), y[m], 1)
    return alpha, beta


def n_to_target(n, y, tau):
    """Exams needed to reach AUC tau, read off the curve of per-N means.

    Inside the data: log-N interpolation at the first size whose mean reaches tau (returns the
    smallest size if that one already does). Beyond 2,600: extrapolate the log-linear trend of
    the three largest sizes (flagged as extrapolation by the caller); inf if that trend is flat
    or falling, i.e. the target is not in sight."""
    m = np.isfinite(y)
    sizes = np.unique(n[m])
    mu = np.array([y[m][n[m] == s].mean() for s in sizes])
    hit = np.flatnonzero(mu >= tau)
    if hit.size:
        i = hit[0]
        if i == 0:
            return float(sizes[0])
        x0, x1 = np.log(sizes[i - 1]), np.log(sizes[i])
        return float(np.exp(x0 + (tau - mu[i - 1]) * (x1 - x0) / (mu[i] - mu[i - 1])))
    b, a = np.polyfit(np.log(sizes[-3:]), mu[-3:], 1)
    return float(np.exp((tau - a) / b)) if b > 0 else np.inf


def aic(rss: float, n: int, k: int) -> float:
    return n * np.log(rss / n) + 2 * k


# ------------------------------------------------------------------ data
def load_points():
    df = (pl.read_parquet(RESULTS / "runs.parquet")
          .filter(pl.col("selected") & (pl.col("split") == "test")))
    return df


def curve_runs(df, design, regime, model):
    d = DESIGNS[design]
    return (df.filter((pl.col("finding") == "macro") & (pl.col("regime") == regime)
                      & (pl.col("model") == model) & pl.col("rotation").is_in(d["rot"])
                      & pl.col("seed").is_in(d["seeds"]))
            .select("run", "rotation", "seed", "n_train").sort("rotation", "seed", "n_train"))


def main() -> int:
    df = load_points()
    dr = bootcache.draws("test")
    auc = {(r, f): a for r, f, a in df.select("run", "finding", "auc").iter_rows()}
    rng = np.random.default_rng(20261031)

    # ---------------- T5.2 table: mean +- SE over runs, test and gold macro
    gold = (pl.read_parquet(RESULTS / "runs.parquet")
            .filter(pl.col("selected") & (pl.col("split") == "gold") & (pl.col("finding") == "macro"))
            .select("run", pl.col("auc").alias("gold")))
    mac = df.filter(pl.col("finding") == "macro").join(gold, on="run")
    t3 = []
    for design, d in DESIGNS.items():
        sub = mac.filter(pl.col("rotation").is_in(d["rot"]) & pl.col("seed").is_in(d["seeds"]))
        if design == "frozen_full":
            sub = sub.filter(pl.col("regime") == "frozen")
        t3.append(sub.group_by("regime", "model", "n_train").agg(
            pl.len().alias("k"), pl.col("auc").mean().alias("test_mean"),
            (pl.col("auc").std() / pl.len().sqrt()).alias("test_se"),
            pl.col("gold").mean().alias("gold_mean"),
            (pl.col("gold").std() / pl.len().sqrt()).alias("gold_se")).with_columns(
            pl.lit(design).alias("design")))
    t3 = pl.concat(t3).sort("design", "regime", "model", "n_train")
    (RESULTS / "tables").mkdir(exist_ok=True)
    t3.write_csv(RESULTS / "tables" / "table3_curves.csv")

    # ---------------- T5.3 fits
    (RESULTS / "fits").mkdir(exist_ok=True)
    fits, finding_rows, aic_rows, reps = {}, [], [], {}
    for design, d in DESIGNS.items():
        units = [(r, s) for r in d["rot"] for s in d["seeds"]]
        plan_u = rng.integers(0, len(units), (NB, len(units)))
        plan_d = rng.integers(0, bootcache.B, (NB, max(d["rot"]) + 1))
        for regime, model in d["curves"]:
            cr = curve_runs(df, design, regime, model)
            runs = cr["run"].to_list()
            n_all = cr["n_train"].to_numpy().astype(float)
            unit_rows = {u: np.flatnonzero((cr["rotation"].to_numpy() == u[0]) & (cr["seed"].to_numpy() == u[1]))
                         for u in units}
            # replicate b = the runs of the resampled units, each read at its rotation's draw index
            IDX = np.stack([np.concatenate([unit_rows[units[ui]] for ui in plan_u[b]]) for b in range(NB)])
            DRW = np.stack([np.concatenate([np.full(len(unit_rows[units[ui]]), plan_d[b, units[ui][0]])
                                            for ui in plan_u[b]]) for b in range(NB)])
            D = np.stack([dr[r] for r in runs])                      # (runs, 13, B)
            YB = D[IDX, :, DRW].astype(float)                         # (NB, L, 13)
            NNB = n_all[IDX]                                          # (NB, L)
            for row_i, finding in enumerate(ROWS):
                y = np.array([auc[(r, finding)] for r in runs])
                ok = np.isfinite(y)
                if ok.sum() < 6:
                    continue
                n = n_all
                al, be = loglin(n, y)
                be_rep = np.array([loglin(NNB[b], YB[b, :, row_i])[1] for b in range(NB)])
                rec = {"design": design, "regime": regime, "model": model, "finding": finding,
                       "n_points": int(ok.sum()), "gain_per_doubling": be,
                       "gain_per_doubling_lo": float(np.nanpercentile(be_rep, 2.5)),
                       "gain_per_doubling_hi": float(np.nanpercentile(be_rep, 97.5)),
                       "auc_at_2600": float(np.nanmean(y[n == NMAX]))}
                pt = fit_ipl(n[ok], y[ok])
                if pt is not None:
                    rec.update({"ipl_a": pt[0][0], "ipl_b": pt[0][1], "ipl_c": pt[0][2],
                                "ipl_a_at_bound": bool(pt[0][0] > 0.999)})
                if finding != "macro":
                    finding_rows.append(rec)
                    continue
                # macro only: pre-registered power-law bootstrap + exams-to-target
                p, rss = pt
                rep = []
                for b in range(NB):
                    f = fit_ipl(NNB[b], YB[b, :, row_i], p0=tuple(p))
                    rep.append(derived(f[0]) if f else None)
                ok_rep = [r for r in rep if r is not None]
                for k, v in derived(p).items():
                    arr = np.array([r[k] for r in ok_rep])
                    rec["ipl_" + k] = v
                    rec["ipl_" + k + "_lo"] = float(np.percentile(arr, 2.5))
                    rec["ipl_" + k + "_hi"] = float(np.percentile(arr, 97.5))
                ntau_rep = {}
                for t in TAUS:
                    k = "N_tau%.2f" % t
                    v = n_to_target(n, y, t)
                    arr = np.array([n_to_target(NNB[b], YB[b, :, row_i], t) for b in range(NB)])
                    ntau_rep[k] = arr
                    rec[k] = v
                    rec[k + "_lo"] = float(np.percentile(arr, 2.5))
                    rec[k + "_hi"] = float(np.percentile(arr, 97.5))
                    rec[k + "_in_data"] = bool(np.isfinite(v) and v <= NMAX)
                    rec[k + "_p_in_data"] = float((arr <= NMAX).mean())
                fits[f"{design}|{regime}|{model}"] = rec
                reps[(design, regime, model)] = ntau_rep
                # H5: compare curve forms on the same run-level points
                rss_ll = float(np.sum((al + be * np.log2(n) - y) ** 2))
                try:
                    pe, _ = curve_fit(expsat, n, y, p0=(y.max() + 0.02, 0.1, 500),
                                      bounds=([0.5, 1e-6, 10], [1.0, 5.0, 1e5]), maxfev=20000)
                    rss_ex = float(np.sum((expsat(n, *pe) - y) ** 2))
                except RuntimeError:
                    rss_ex = np.nan
                a_ = {"power_law": aic(rss, len(n), 3), "log_linear": aic(rss_ll, len(n), 2),
                      "exp_saturation": aic(rss_ex, len(n), 3)}
                best = min(a_, key=lambda k: a_[k] if np.isfinite(a_[k]) else np.inf)
                aic_rows.append({"design": design, "regime": regime, "model": model,
                                 **{"aic_" + k: v for k, v in a_.items()}, "best": best,
                                 "dAIC_power_minus_best": a_["power_law"] - a_[best],
                                 "power_law_a_at_bound": bool(p[0] > 0.999)})
            print("[fit] %s %-6s %s done" % (design, regime, model), flush=True)

    # paired efficiency ratios (same replicate plan within a design): exams-to-target A / B
    ratio_rows = []
    for design, A, Bm, label in RATIOS:
        ra, rb = reps[(design, *A)], reps[(design, *Bm)]
        fa, fb = fits[f"{design}|{A[0]}|{A[1]}"], fits[f"{design}|{Bm[0]}|{Bm[1]}"]
        for t in TAUS:
            k = "N_tau%.2f" % t
            with np.errstate(invalid="ignore", divide="ignore"):
                arr = ra[k] / rb[k]
            arr = arr[np.isfinite(arr)]
            pt = fa[k] / fb[k] if np.isfinite(fa[k]) and np.isfinite(fb[k]) else np.nan
            ratio_rows.append({"comparison": label, "design": design, "A": "%s %s" % A,
                               "B": "%s %s" % Bm, "tau": t, "N_A": fa[k], "N_B": fb[k],
                               "A_in_data": fa[k + "_in_data"], "B_in_data": fb[k + "_in_data"],
                               "ratio_A_over_B": pt,
                               "ratio_lo": float(np.percentile(arr, 2.5)) if arr.size else np.nan,
                               "ratio_hi": float(np.percentile(arr, 97.5)) if arr.size else np.nan,
                               "p_B_needs_fewer": float((arr > 1).mean()) if arr.size else np.nan})

    (RESULTS / "fits" / "fits_macro.json").write_text(json.dumps(fits, indent=1, default=float))
    t4 = pl.DataFrame(list(fits.values()))
    t4.write_csv(RESULTS / "tables" / "table4_fits.csv")
    pl.DataFrame(finding_rows).write_csv(RESULTS / "tables" / "fits_findings.csv")
    pl.DataFrame(aic_rows).write_csv(RESULTS / "tables" / "aic_curve_forms.csv")
    pl.DataFrame(ratio_rows).write_csv(RESULTS / "tables" / "efficiency_ratios.csv")

    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=200, float_precision=3):
        print(t4.select("design", "regime", "model", "auc_at_2600", "gain_per_doubling", "gain_per_doubling_lo",
                        "gain_per_doubling_hi", "ipl_a", "ipl_a_lo", "ipl_c", "N_tau0.75", "N_tau0.75_lo",
                        "N_tau0.75_hi", "N_tau0.80", "N_tau0.80_lo", "N_tau0.80_hi"))
        print(pl.DataFrame(aic_rows).select("design", "regime", "model", "best", "dAIC_power_minus_best"))
        print(pl.DataFrame(ratio_rows).filter(pl.col("tau") < 0.85))
    return 0


if __name__ == "__main__":
    sys.exit(main())
