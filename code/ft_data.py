"""T4.1: study-level dataset for fine-tuning.

Workers only read uint8 slices from the 84 GB memmap (cheap, no torch math); resize,
augmentation and normalisation run on the GPU in gpu_prep(), batched over all slices
of a micro-batch. This keeps each Windows DataLoader worker small -- every worker is a
separate process with its own torch import (~1.7 GB RAM), so we use at most 4.

Slice selection (PROJECT_PLAN 5.7):
  train : per series, K stratified-random slices (K equal bins, one random slice each)
  val   : per series, K evenly spaced slices
  test  : every slice of the study, evenly thinned to 384 -- identical rows to
          build_bags(cap=384), i.e. exactly what the frozen head sees
A study is capped at MAX_SLICES slices in train/val mode by lowering K for studies with
more than 6 series, so the largest training batch has a FIXED size and its memory can
be measured once (bench_ft.py) instead of hoped about.

Augmentation (train only, identical for all slices of one series): affine rotation
+-10 deg, scale 0.9-1.1, translation +-5 %, brightness +-0.1, contrast 0.9-1.1, gamma
0.9-1.1. NO flips or slice reversal: every volume is in the lateral->medial frame and a
mirror would swap the medial/lateral labels.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from common import N_SLICES, PIXELS, PX, SLOTS, build_bags, label_arrays, load_series_index, load_targets

MAX_SLICES = 48
CAP = 384


class StudyDataset(Dataset):
    def __init__(self, sids: list[str], mode: str, k: int = 8, seed: int = 0,
                 tg: pl.DataFrame | None = None, se: pl.DataFrame | None = None):
        assert mode in ("train", "val", "test")
        self.sids, self.mode, self.k, self.seed = list(sids), mode, k, seed
        tg = load_targets() if tg is None else tg
        se = load_series_index() if se is None else se
        lab = label_arrays(tg)
        row = {s: i for i, s in enumerate(tg["study_dir"].to_list())}
        ix = [row[s] for s in self.sids]
        self.T, self.W = lab["T"][ix], lab["W"][ix]
        slot_id = {s: i for i, s in enumerate(SLOTS)}
        want = set(self.sids)
        self.series: dict[str, list[tuple[int, int, int]]] = {}
        for r in se.filter(pl.col("study_dir").is_in(list(want))).iter_rows(named=True):
            self.series.setdefault(r["study_dir"], []).append((r["row0"], r["n"], slot_id[r["slot"]]))
        if mode == "test":
            sub = se.filter(pl.col("study_dir").is_in(list(want)))
            self.bags = build_bags(sub, CAP, np.random.default_rng(0))
        self.mm, self.calls = None, 0

    def __len__(self):
        return len(self.sids)

    def _rows(self, sid: str, rng):
        if self.mode == "test":
            idx, sl, po = self.bags[sid]
            return idx, sl, po, np.zeros(len(idx), np.int64)
        ser = self.series[sid]
        k = min(self.k, max(1, MAX_SLICES // len(ser)))
        idx, sl, po, sr = [], [], [], []
        for j, (r0, n, slot) in enumerate(ser):
            if n <= k:
                take = np.arange(n)
            elif self.mode == "train":
                edges = np.linspace(0, n, k + 1).astype(int)
                take = np.array([rng.integers(a, max(a + 1, b)) for a, b in zip(edges[:-1], edges[1:])])
            else:
                take = np.unique(np.linspace(0, n - 1, k).round().astype(int))
            idx.append(r0 + take)
            sl.append(np.full(len(take), slot))
            po.append(take / (n - 1) if n > 1 else np.full(len(take), 0.5))
            sr.append(np.full(len(take), j))
        return (np.concatenate(idx), np.concatenate(sl).astype(np.int64),
                np.concatenate(po).astype(np.float32), np.concatenate(sr).astype(np.int64))

    def __getitem__(self, i):
        if self.mm is None:                     # opened lazily inside each worker
            self.mm = np.memmap(PIXELS, dtype=np.uint8, mode="r", shape=(N_SLICES, PX, PX))
        self.calls += 1
        rng = np.random.default_rng([self.seed, i, self.calls])
        idx, sl, po, sr = self._rows(self.sids[i], rng)
        order = np.argsort(idx, kind="stable")        # sequential reads; restore order after
        px = np.empty((len(idx), PX, PX), np.uint8)
        px[order] = self.mm[idx[order]]
        return {"px": torch.from_numpy(px), "slot": torch.from_numpy(sl), "pos": torch.from_numpy(po),
                "series": torch.from_numpy(sr), "t": torch.from_numpy(self.T[i]),
                "w": torch.from_numpy(self.W[i]), "i": i}


def collate(items):
    """Concatenate slices of all studies; keep per-study lengths for the scatter."""
    lens = [len(it["slot"]) for it in items]
    B, L = len(items), max(lens)
    slot = torch.zeros(B, L, dtype=torch.long); pos = torch.zeros(B, L)
    mask = torch.zeros(B, L, dtype=torch.bool)
    series = []
    off = 0
    for b, it in enumerate(items):
        n = lens[b]
        slot[b, :n], pos[b, :n], mask[b, :n] = it["slot"], it["pos"], True
        series.append(it["series"] + off)          # globally unique series ids in the batch
        off += int(it["series"].max()) + 1
    return {"px": torch.cat([it["px"] for it in items]), "series": torch.cat(series),
            "slot": slot, "pos": pos, "mask": mask,
            "t": torch.stack([it["t"] for it in items]), "w": torch.stack([it["w"] for it in items]),
            "i": torch.tensor([it["i"] for it in items])}


class ViewSampler(Sampler):
    """Exactly `views` study indices: concatenated seeded permutations; can skip a prefix (resume)."""

    def __init__(self, n: int, views: int, seed: int, skip: int = 0):
        rng = np.random.default_rng(seed)
        reps = math.ceil(views / n)
        self.order = np.concatenate([rng.permutation(n) for _ in range(reps)])[:views][skip:]

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        return iter(self.order.tolist())


def gpu_prep(px: torch.Tensor, series: torch.Tensor, mean, std, size: int = 224,
             aug: bool = False, gen: torch.Generator | None = None) -> torch.Tensor:
    """uint8 (M,320,320) on GPU -> normalised float (M,3,size,size).

    The eval path is the same arithmetic as stage4_embed.SliceSet (x/255, bilinear
    antialiased resize, repeat to 3 channels, timm mean/std), so fine-tuned and frozen
    models see identical inputs.
    """
    x = px.float().div_(255.0)[:, None]
    if size != px.shape[-1]:
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    if aug:
        ns = int(series.max()) + 1
        dev = x.device
        u = lambda lo, hi: torch.rand(ns, device=dev, generator=gen) * (hi - lo) + lo
        ang = u(-10, 10) * math.pi / 180
        sc, tx, ty = u(0.9, 1.1), u(-0.1, 0.1), u(-0.1, 0.1)       # +-0.1 normalised = +-5 % of width
        br, ct, gm = u(-0.1, 0.1), u(0.9, 1.1), u(0.9, 1.1)
        cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc        # inverse map; det > 0, never a mirror
        theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)[series]
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        m = x.mean((2, 3), keepdim=True)
        x = ((x - m) * ct[series].view(-1, 1, 1, 1) + m + br[series].view(-1, 1, 1, 1)).clamp_(0, 1)
        x = x.pow(gm[series].view(-1, 1, 1, 1))
    x = x.expand(-1, 3, -1, -1)
    return (x - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)
