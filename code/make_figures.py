"""T5.9: regenerate every report figure from results/ in one consistent style (viz.py).

  F1   pipeline schematic                      F5  AUC gain per doubling, by finding
  F1b  label prevalence (describe_data.py)     F6  AUC vs training positives (prevalence check)
  F2   main learning curves                    F7  compute cost vs performance
  F3   per-finding curves (appendix)           F8  per-test-fold curves (appendix)
  F4   paired differences with 95% CIs

All curves use the "matched" design (rotations 0-2, seed 0) unless the caption says
otherwise, so frozen and fine-tuned points share training subsets and test folds.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

import viz
from common import FINDING_GROUPS, LABELS, RESULTS, SIZES

T = RESULTS / "tables"
GROUP_COLOR = {"focal": "#e34948", "cartilage_oa": "#6250d6", "fluid_marrow_soft": "#008300"}
GROUP_MARK = {"focal": "o", "cartilage_oa": "s", "fluid_marrow_soft": "^"}
GROUP_NAME = {"focal": "Focal structural", "cartilage_oa": "Cartilage / OA",
              "fluid_marrow_soft": "Fluid / marrow / soft tissue"}
FIND_GROUP = {lab: g for g, labs in FINDING_GROUPS.items() for lab in labs}
CURVES = [("frozen", "M1"), ("frozen", "M2"), ("frozen", "M3"), ("frozen", "M4"), ("ft", "M1"), ("ft", "M2")]


def runs():
    return pl.read_parquet(RESULTS / "runs.parquet").filter(pl.col("selected"))


def matched(df):
    return df.filter(pl.col("rotation").is_in([0, 1, 2]) & (pl.col("seed") == 0))


def style_kw(regime, model):
    return dict(color=viz.COLORS[model], ls=viz.LINESTYLE[regime], marker=viz.MARKERS[model],
                mfc="white" if regime == "frozen" else viz.COLORS[model], mew=1.2, ms=4.5, lw=1.6)


def legend_handles(curves):
    return [Line2D([], [], **style_kw(rg, m), label="%s, %s" % (viz.SHORT[m], viz.REGIME_NAME[rg]))
            for rg, m in curves]


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                fontsize=7.5, color=viz.INK2)


# --------------------------------------------------------------------- F1
def f1_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def box(x, y, w, h, title, body, tone="#f3f2ee"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=tone, ec=viz.AXIS, lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=6.8, weight="semibold", color=viz.INK)
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=5.9, color=viz.INK2, linespacing=1.35)

    def arrow(x0, x1, y):
        ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>", color=viz.MUTED, lw=1))

    box(0.5, 21, 18, 17, "4,407 MRI studies", "~20 sites, 5 vendors\n24,371 series\n819,078 slices\n+ free-text reports")
    box(0.5, 1.5, 18, 16, "Labels", "local LLM reads each\nreport: 12 findings\n(no data leaves the PC)\n58 studies: expert labels")
    box(21.5, 21, 18, 17, "Preprocess", "resample 0.5 mm/px\nlateral-to-medial order\n6 sequence slots\n224 px input")
    box(21.5, 1.5, 18, 16, "Splits", "5 scanner-grouped folds\ntest = unseen sites\nval fold for selection\nnested N = 65 ... 2,600")
    box(42.5, 11, 20.5, 27, "Slice encoder", "ViT-B/16  DINOv3\nConvNeXt-B  DINOv3\nConvNeXt-B  ImageNet\nViT-B/14  DINOv2\nResNet-18  from scratch\n\nfrozen  or  fine-tuned")
    box(66.0, 11, 17.0, 27, "Study head", "all slices, all\nsequences of a study\n12 learned queries\n(attention pooling)\n12 probabilities")
    box(86.0, 11, 13.5, 27, "Evaluate once", "test fold:\nreport labels\n(870 studies)\n\ngold 58:\nexpert labels")
    arrow(18.9, 21.2, 29.5)
    arrow(18.9, 21.2, 9.5)
    arrow(39.9, 42.3, 29.5)
    arrow(39.9, 42.3, 14)
    arrow(63.4, 65.8, 24.5)
    arrow(83.4, 85.8, 24.5)
    ax.text(42.5, 4.5, "Same head, labels, splits and budget rules for every backbone:\n"
            "differences come from the backbone and training regime only.",
            ha="left", va="center", fontsize=6.2, color=viz.MUTED, style="italic")
    return viz.save(fig, "F1_pipeline")


# --------------------------------------------------------------------- F2
def f2_curves(df):
    t3 = pl.read_csv(T / "table3_curves.csv").filter(pl.col("design") == "matched")
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    for rg, m in CURVES:
        s = t3.filter((pl.col("regime") == rg) & (pl.col("model") == m)).sort("n_train")
        x, y, se = s["n_train"].to_numpy(), s["test_mean"].to_numpy(), s["test_se"].to_numpy()
        ax.fill_between(x, y - se, y + se, color=viz.COLORS[m], alpha=0.10, lw=0)
        ax.plot(x, y, **style_kw(rg, m))
    sc = df.filter((pl.col("regime") == "scratch") & (pl.col("split") == "test") & (pl.col("finding") == "macro")).sort("n_train")
    ax.plot(sc["n_train"], sc["auc"], **style_kw("scratch", "M5"))
    viz.size_axis(ax, [65, 130, 260, 520, 1040, 2600])
    ax.set_xlim(55, 4200)
    ax.set_xlabel("Training exams (N, log scale)")
    ax.set_ylabel("Macro AUC on unseen-site test fold\n(12 findings, report labels)")
    ends = {(rg, m): t3.filter((pl.col("regime") == rg) & (pl.col("model") == m) & (pl.col("n_train") == 2600))["test_mean"][0]
            for rg, m in CURVES}
    label_end(ax, 2600, ends[("ft", "M2")], "ConvNeXt-B fine-tuned", viz.COLORS["M2"], dy=5)
    label_end(ax, 2600, ends[("ft", "M1")], "ViT-B fine-tuned", viz.COLORS["M1"], dy=-4)
    label_end(ax, 2600, ends[("frozen", "M3")], "ImageNet ConvNeXt-B frozen", viz.COLORS["M3"])
    label_end(ax, 2600, sc["auc"][-1], "ResNet-18 scratch (fold 0)", viz.COLORS["M5"], dy=-2)
    h = legend_handles(CURVES) + [Line2D([], [], **style_kw("scratch", "M5"), label="ResNet-18, from scratch (fold 0 only)")]
    ax.legend(handles=h, loc="upper left", fontsize=7, ncol=1, handlelength=2.6)
    ax.set_title("Macro test AUC by training-set size")
    return viz.save(fig, "F2_learning_curves")


# --------------------------------------------------------------------- F3
def f3_findings(df):
    d = matched(df.filter(pl.col("split") == "test") .filter(pl.col("finding") != "macro"))
    show = [("frozen", "M3"), ("frozen", "M1"), ("frozen", "M2"), ("ft", "M1"), ("ft", "M2")]
    order = [lab for g in FINDING_GROUPS.values() for lab in g]
    fig, axs = plt.subplots(3, 4, figsize=(7.4, 6.4), sharex=True)
    for ax, lab in zip(axs.flat, order):
        for rg, m in show:
            s = (d.filter((pl.col("finding") == lab) & (pl.col("regime") == rg) & (pl.col("model") == m))
                 .group_by("n_train").agg(pl.col("auc").mean()).sort("n_train"))
            ax.plot(s["n_train"], s["auc"], **{**style_kw(rg, m), "ms": 3, "lw": 1.2})
        viz.size_axis(ax, [65, 260, 1040, 2600])
        ax.tick_params(labelsize=6.5)
        ax.set_title(lab, fontsize=8, color=viz.INK)
        ax.text(0.98, 0.04, GROUP_NAME[FIND_GROUP[lab]].split(" ")[0], transform=ax.transAxes, ha="right",
                fontsize=6, color=viz.MUTED, style="italic")
    for ax in axs[-1]:
        ax.set_xlabel("Training exams", fontsize=7)
    for ax in axs[:, 0]:
        ax.set_ylabel("Test AUC", fontsize=7)
    fig.legend(handles=legend_handles(show), loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Per-finding learning curves (rotations 0-2, seed 0)", x=0.01, ha="left", fontsize=10, weight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    return viz.save(fig, "F3_findings")


# --------------------------------------------------------------------- F4
def f4_deltas():
    pdl = pl.read_csv(T / "paired_deltas.csv").filter(pl.col("finding") == "macro")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
    left = [("Fine-tuned vs frozen, DINOv3 ViT-B", "M1", "ft", "ViT-B"),
            ("Fine-tuned vs frozen, DINOv3 ConvNeXt-B", "M2", "ft", "ConvNeXt-B")]
    right = [("ConvNeXt-B minus ViT-B, fine-tuned", "M2", "ft", "CNN - ViT, fine-tuned"),
             ("ConvNeXt-B minus ViT-B, frozen", "M2", "frozen", "CNN - ViT, frozen"),
             ("ImageNet minus DINOv3 pretraining (ConvNeXt-B, frozen)", "M3", "frozen", "ImageNet - DINOv3 (CNN, frozen)"),
             ("DINOv3 minus DINOv2 (ViT-B, frozen)", "M1", "frozen", "DINOv3 - DINOv2 (ViT, frozen)")]
    for ax, spec, title in [(a1, left, "Gain from fine-tuning (C1)"),
                            (a2, right, "Architecture and pretraining (C2-C4)")]:
        ax.axhline(0, color=viz.INK2, lw=0.8)
        for i, (comp, m, rg, lab) in enumerate(spec):
            s = pdl.filter(pl.col("comparison") == comp).sort("n_train")
            x = s["n_train"].to_numpy() * (1 + 0.035 * (i - (len(spec) - 1) / 2))
            kw = style_kw(rg, m)
            ax.fill_between(x, s["lo"], s["hi"], color=viz.COLORS[m], alpha=0.10, lw=0)
            ax.plot(x, s["delta"], label=lab, **kw)
        viz.size_axis(ax, [65, 260, 1040, 2600])
        ax.set_xlabel("Training exams (log scale)")
        ax.set_title(title, fontsize=9)
        ax.legend(loc="upper left", fontsize=6.8, handlelength=2.6)
    a1.set_ylabel("Difference in macro test AUC\n(paired, 95% bootstrap CI)")
    fig.tight_layout()
    return viz.save(fig, "F4_paired_differences")


# --------------------------------------------------------------------- F5
def f5_gain():
    """Per-finding AUC gained per doubling of the training set (log-linear slope, 65-2,600).
    Replaces the planned 'exams to plateau' plot: no curve shows an identifiable plateau in
    this range (fit_curves.py), so the slope is the quantity the data can support."""
    ff = pl.read_csv(T / "fits_findings.csv").filter(pl.col("design") == "matched")
    show = [("ft", "M2"), ("ft", "M1"), ("frozen", "M3")]
    order = [lab for g in FINDING_GROUPS.values() for lab in g]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    y0 = np.arange(len(order))[::-1]
    offs = {0: 0.22, 1: 0.0, 2: -0.22}
    for k, (rg, m) in enumerate(show):
        s = {r["finding"]: r for r in ff.filter((pl.col("regime") == rg) & (pl.col("model") == m)).iter_rows(named=True)}
        yy = y0 + offs[k]
        for yi, lab in zip(yy, order):
            r = s.get(lab)
            if r is not None:
                ax.plot([r["gain_per_doubling_lo"], r["gain_per_doubling_hi"]], [yi, yi], color=viz.COLORS[m],
                        lw=1.2, alpha=0.6, solid_capstyle="round")
        xs = [s[l]["gain_per_doubling"] if l in s else np.nan for l in order]
        ax.plot(xs, yy, ls="none", **{k2: v for k2, v in style_kw(rg, m).items() if k2 != "ls"},
                label="%s, %s" % (viz.SHORT[m], viz.REGIME_NAME[rg]))
    ax.axvline(0, color=viz.INK2, lw=0.8)
    ax.set_yticks(y0)
    ax.set_yticklabels(order)
    ax.grid(axis="y", visible=False)
    bounds = np.cumsum([len(v) for v in FINDING_GROUPS.values()])[:-1]
    for b in bounds:
        ax.axhline(len(order) - b - 0.5, color=viz.AXIS, lw=0.6)
    start = 0
    xmax = ax.get_xlim()[1]
    for g, labs in FINDING_GROUPS.items():
        ax.text(xmax, len(order) - start - 0.75, GROUP_NAME[g], ha="right", va="top", fontsize=6.8,
                color=viz.MUTED, style="italic")
        start += len(labs)
    ax.set_xlabel("Test AUC gained per doubling of training exams (log-linear slope, 95% bootstrap CI)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=3, fontsize=7, handletextpad=0.3)
    ax.set_title("AUC gained per doubling of training exams, by finding")
    return viz.save(fig, "F5_gain_per_doubling")


# --------------------------------------------------------------------- F6
def f6_positives(df):
    d = matched(df.filter((pl.col("split") == "test") & (pl.col("finding") != "macro")))
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5), sharey=True)
    for ax, (rg, m) in zip(axs, [("ft", "M2"), ("frozen", "M3")]):
        s = d.filter((pl.col("regime") == rg) & (pl.col("model") == m))
        for lab in LABELS:
            g = FIND_GROUP[lab]
            t = (s.filter(pl.col("finding") == lab).group_by("n_train")
                 .agg(pl.col("n_train_pos").mean(), pl.col("auc").mean()).sort("n_train"))
            ax.plot(t["n_train_pos"], t["auc"], color=GROUP_COLOR[g], marker=GROUP_MARK[g], ms=3.2,
                    lw=1.0, alpha=0.85, mfc="white", mew=1)
            if lab in ("MCL", "Fracture", "Effusion", "Medial Meniscus"):
                dy = {"MCL": -5, "Fracture": 5}.get(lab, 0) if rg == "frozen" else 0
                ax.annotate(lab, (t["n_train_pos"][-1], t["auc"][-1]), xytext=(4, dy), textcoords="offset points",
                            fontsize=6.3, color=viz.INK2, va="center")
        ax.set_xscale("log")
        ax.set_xlim(1, 3000)
        ax.set_xlabel("Positive training examples of that finding (log scale)")
        ax.set_title("%s, %s" % (viz.SHORT[m], viz.REGIME_NAME[rg]), fontsize=9)
    axs[0].set_ylabel("Test AUC for that finding")
    h = [Line2D([], [], color=GROUP_COLOR[g], marker=GROUP_MARK[g], mfc="white", mew=1, lw=1, label=GROUP_NAME[g])
         for g in GROUP_COLOR]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return viz.save(fig, "F6_auc_vs_positives")


# --------------------------------------------------------------------- F7
def f7_cost(df):
    d = matched(df.filter((pl.col("split") == "test") & (pl.col("finding") == "macro")))
    gh = pl.read_csv(T / "gpu_hours_by_n.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for rg, m in CURVES:
        a = d.filter((pl.col("regime") == rg) & (pl.col("model") == m)).group_by("n_train").agg(pl.col("auc").mean())
        s = a.join(gh.filter((pl.col("regime") == rg) & (pl.col("model") == m)), on="n_train").sort("n_train")
        ax.plot(s["gpu_h"], s["auc"], **style_kw(rg, m))
        if (rg, m) in (("ft", "M2"), ("frozen", "M3")):
            for n_, x_, y_ in s.select("n_train", "gpu_h", "auc").iter_rows():
                if n_ in (65, 520, 2600):
                    ax.annotate("N=%s" % f"{n_:,}", (x_, y_), xytext=(6, -11) if rg == "frozen" else (6, -3),
                                textcoords="offset points", fontsize=6.5, color=viz.INK2)
    ax.set_xscale("log")
    ax.set_xlabel("GPU-hours to train one model (log scale; laptop RTX 5070 Ti)")
    ax.set_ylabel("Macro test AUC")
    ax.legend(handles=legend_handles(CURVES), loc="upper left", fontsize=7, handlelength=2.6)
    ax.set_title("Compute per trained model vs macro test AUC")
    return viz.save(fig, "F7_cost_vs_auc")


# --------------------------------------------------------------------- F8
def f8_rotations(df):
    d = df.filter((pl.col("split") == "test") & (pl.col("finding") == "macro") & (pl.col("regime") != "scratch"))
    vend = {0: "fold 0: 80% Philips", 1: "fold 1: 63% Siemens", 2: "fold 2: 69% 1.5 T",
            3: "fold 3: 100% 1.5 T", 4: "fold 4: 19% Canon"}
    fig, axs = plt.subplots(1, 5, figsize=(7.6, 2.9), sharey=True)
    for r, ax in enumerate(axs):
        for rg, m in CURVES:
            s = (d.filter((pl.col("rotation") == r) & (pl.col("regime") == rg) & (pl.col("model") == m))
                 .group_by("n_train").agg(pl.col("auc").mean()).sort("n_train"))
            if s.height:
                ax.plot(s["n_train"], s["auc"], **{**style_kw(rg, m), "ms": 2.6, "lw": 1.1})
        viz.size_axis(ax, [65, 520, 2600])
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("Training exams", fontsize=7)
        ax.set_title("Test %s" % vend[r], fontsize=7.2)
    axs[0].set_ylabel("Macro test AUC")
    fig.legend(handles=legend_handles(CURVES), loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return viz.save(fig, "F8_by_test_site")


def main() -> int:
    viz.setup()
    df = runs()
    for f in (f1_pipeline, lambda: f2_curves(df), lambda: f3_findings(df), f4_deltas, f5_gain,
              lambda: f6_positives(df), lambda: f7_cost(df), lambda: f8_rotations(df)):
        print("[fig]", f())
    return 0


if __name__ == "__main__":
    sys.exit(main())
