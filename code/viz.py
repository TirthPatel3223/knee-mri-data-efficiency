"""One figure style for every report figure (T5.9), so a model always looks the same.

Encoding rules (fixed, never per-figure):
  * colour = backbone (model id), in a fixed categorical order validated for colour-vision
    deficiency with the dataviz skill's validator (adjacent CVD dE >= 9.1, normal-vision
    dE >= 19.6). Three slots are below 3:1 contrast on white, so every series also gets a
    marker shape and, where it fits, a direct label -- identity is never colour alone.
  * line style = regime: solid = fine-tuned, dashed = frozen, dotted = from scratch.
  * hairline solid grid, recessive axes, system sans, 300 dpi PNG + vector PDF.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import GP  # noqa: E402

FIG = GP / "figures"

COLORS = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#eda100", "M5": "#e87ba4"}
MARKERS = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "M5": "v"}
NAMES = {"M1": "DINOv3 ViT-B/16", "M2": "DINOv3 ConvNeXt-B", "M3": "ImageNet ConvNeXt-B",
         "M4": "DINOv2 ViT-B/14", "M5": "ResNet-18 (scratch)"}
SHORT = {"M1": "ViT-B (DINOv3)", "M2": "ConvNeXt-B (DINOv3)", "M3": "ConvNeXt-B (ImageNet)",
         "M4": "ViT-B (DINOv2)", "M5": "ResNet-18 scratch"}
LINESTYLE = {"ft": "-", "frozen": (0, (4, 2)), "scratch": (0, (1, 1.5))}
REGIME_NAME = {"ft": "fine-tuned", "frozen": "frozen", "scratch": "from scratch"}

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GRAY = "#9a998f"           # reference lines / de-emphasised series
SIZES_TICKS = [65, 130, 260, 520, 1040, 2080, 2600]


def setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "lines.linewidth": 1.6,
        "lines.markersize": 5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def size_axis(ax, ticks=SIZES_TICKS) -> None:
    """log2 x-axis labelled at the study's training-set sizes."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    ax.minorticks_off()


def save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return FIG / f"{name}.png"
