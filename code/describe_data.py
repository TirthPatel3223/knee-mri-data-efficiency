"""T1.3: descriptive statistics for report section 2 (Table 1, Fig. 1b).

Table 1 describes the cohort (studies, series, slices, vendors, field strength, sex, which
MRI sequences/planes are present). Fig. 1b contrasts, on the SAME 58 gold studies, how
often each finding is positive according to the radiology report (as read by the LLM) and
according to the expert image labels -- the visual form of the "reports are a biased proxy"
limitation. The gold set is enriched for abnormalities, so its prevalence is not compared
with the whole cohort's; the cohort-wide report prevalence is in prevalence.csv.

Report-language shares are not recomputed here; they come from the earlier pipeline's
classification (Kaggle_Knee_Competition/RESEARCH.md section 1.4) and are cited as such.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from common import FINDING_GROUPS, KEYS, LABELS, RESULTS, SLOTS, load_series_index, load_targets
import viz

SLOT_NAMES = {"Sag_0": "Sagittal, non-fat-suppressed", "Sag_1": "Sagittal, fluid-sensitive FS",
              "Cor_0": "Coronal, non-fat-suppressed", "Cor_1": "Coronal, fluid-sensitive FS",
              "Axi_0": "Axial, non-fat-suppressed", "Axi_1": "Axial, fluid-sensitive FS"}
LANGUAGES = [("English", 1809), ("Spanish", 685), ("Turkish", 514), ("Croatian/Bosnian", 369),
             ("Greek", 321), ("German", 261), ("Bulgarian", 220), ("Dutch", 153), ("French", 64)]


def pct(a: int, b: int) -> str:
    return f"{a:,} ({100 * a / b:.1f}%)"


def main() -> int:
    tg, se = load_targets(), load_series_index()
    n = tg.height
    rows: list[tuple[str, str, str]] = []

    def add(section, item, value):
        rows.append((section, item, value))

    per = se.group_by("study_dir").agg(pl.len().alias("series"), pl.col("n").sum().alias("slices"))
    q = per.select(pl.col("slices").quantile(q).alias(str(q)) for q in (0.25, 0.5, 0.75)).row(0)
    add("Cohort", "Studies (1 study = 1 patient, 1 knee)", f"{n:,}")
    add("Cohort", "  with expert image labels (gold set)", f"{int(tg['is_gold'].sum()):,}")
    add("Cohort", "Series (MRI sequences)", f"{se.height:,}")
    add("Cohort", "Slices (2D images)", f"{int(se['n'].sum()):,}")
    add("Cohort", "Series per study, median (IQR)",
        "%d (%d-%d)" % tuple(per["series"].quantile(q) for q in (0.5, 0.25, 0.75)))
    add("Cohort", "Slices per study, median (IQR)", "%d (%d-%d)" % (q[1], q[0], q[2]))
    for v, c in tg["vendor"].value_counts().sort("count", descending=True).iter_rows():
        add("Scanner vendor", v, pct(c, n))
    fld = tg.with_columns(pl.when(pl.col("field") == 1.5).then(pl.lit("1.5 T"))
                          .when(pl.col("field") == 3.0).then(pl.lit("3.0 T"))
                          .when(pl.col("field").is_null()).then(pl.lit("unknown"))
                          .otherwise(pl.lit("other (1.0-1.16 T)")).alias("f"))["f"]
    for v in ("1.5 T", "3.0 T", "other (1.0-1.16 T)", "unknown"):
        add("Field strength", v, pct(int((fld == v).sum()), n))
    sx = {"F": "female", "M": "male", "O": "other", None: "unknown"}
    for v, c in tg["sex"].value_counts().sort("count", descending=True).iter_rows():
        add("Sex (DICOM tag)", sx[v], pct(c, n))
    has = se.group_by("study_dir").agg(pl.col("slot").unique())
    for s in SLOTS:
        add("Studies with sequence", SLOT_NAMES[s],
            pct(int(has["slot"].list.contains(s).sum()), n))
    for lang, c in LANGUAGES:
        add("Report language*", lang, pct(c, n))
    t1 = pl.DataFrame(rows, schema=["section", "item", "value"], orient="row")
    (RESULTS / "tables").mkdir(parents=True, exist_ok=True)
    t1.write_csv(RESULTS / "tables" / "table1_data.csv")

    # Prevalence: report stance over non-gold studies vs expert labels on the gold 58.
    ng = tg.filter(~pl.col("is_gold"))
    gd = tg.filter(pl.col("is_gold"))
    prev = []
    for k, lab in zip(KEYS, LABELS):
        s = ng["s_" + k]
        pos, neg, na = int((s == "pos").sum()), int((s == "neg").sum()), int((s == "na").sum())
        g = gd["g_" + k]
        grp = next(gname for gname, labs in FINDING_GROUPS.items() if lab in labs)
        prev.append(dict(finding=lab, group=grp, report_pos=pos, report_neg=neg, report_na=na,
                         report_prev=pos / (pos + neg), report_na_share=na / ng.height,
                         gold_pos=int((g == 1).sum()), gold_n=gd.height,
                         gold_prev=float((g == 1).mean()),
                         gold_report_pos=int((gd["s_" + k] == "pos").sum()),
                         gold_report_prev=float((gd["s_" + k] == "pos").mean())))
    pv = pl.DataFrame(prev)
    pv.write_csv(RESULTS / "tables" / "prevalence.csv")

    # Fig. 1b: dot plot, one row per finding, grouped by the pre-registered finding groups.
    viz.setup()
    order = [lab for g in FINDING_GROUPS.values() for lab in g]
    pvd = {r["finding"]: r for r in pv.iter_rows(named=True)}
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    y = np.arange(len(order))[::-1]
    for yi, lab in zip(y, order):
        r = pvd[lab]
        ax.plot([r["gold_report_prev"] * 100, r["gold_prev"] * 100], [yi, yi], color=viz.GRID, lw=2.2,
                zorder=1, solid_capstyle="round")
    ax.scatter([pvd[l]["gold_report_prev"] * 100 for l in order], y, s=34, color=viz.COLORS["M1"],
               edgecolor="white", linewidth=1, zorder=3, label="Radiology report says positive")
    ax.scatter([pvd[l]["gold_prev"] * 100 for l in order], y, s=38, marker="D",
               color=viz.COLORS["M2"], edgecolor="white", linewidth=1, zorder=3,
               label="Expert image label is positive")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Prevalence (% of studies positive)")
    ax.set_xlim(0, 75)
    ax.grid(axis="y", visible=False)
    bounds = np.cumsum([len(v) for v in FINDING_GROUPS.values()])[:-1]
    for b in bounds:
        ax.axhline(len(order) - b - 0.5, color=viz.AXIS, lw=0.6)
    gl = {"focal": "Focal structural", "cartilage_oa": "Cartilage / OA",
          "fluid_marrow_soft": "Fluid / marrow / soft tissue"}
    start = 0
    for gname, labs in FINDING_GROUPS.items():
        mid = len(order) - start - (len(labs) - 1) / 2 - 1
        ax.text(74, mid, gl[gname], ha="right", va="center", fontsize=7.5, color=viz.MUTED,
                style="italic")
        start += len(labs)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, handletextpad=0.3,
              columnspacing=1.2)
    ax.set_title("Same 58 knees, two label sources: reports vs expert image reads", pad=22)
    viz.save(fig, "F1b_prevalence")

    with pl.Config(tbl_rows=60, fmt_str_lengths=50, tbl_width_chars=120):
        print(t1)
        print(pv.select("finding", "report_prev", "report_na_share", "gold_prev")
              .with_columns(pl.col(pl.Float64).round(3)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
