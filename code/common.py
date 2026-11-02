"""Shared paths, constants, loaders, metrics and memory guards for the group project.

Everything that more than one script needs lives here so that splits, label
definitions and metrics cannot drift between the frozen and fine-tuned pipelines.
The model-side pieces (Head, build_bags, macro_auc, bootstrap helpers) are IMPORTED
from the earlier pipeline in T:\\Kaggle_Knee_Competition\\src, never copied.

MEMORY POLICY (12 GB laptop GPU, 31 GB RAM shared with the desktop):
  * gpu_cap() limits this process to a fixed share of VRAM. On Windows the NVIDIA
    driver otherwise silently spills over-allocations into system RAM ("sysmem
    fallback"), which does not crash but runs ~10x slower and starves the desktop.
    With the cap, an over-allocation becomes an ordinary, catchable PyTorch OOM.
  * load_emb_gpu() streams an embedding memmap to the GPU in chunks, so host RAM
    never holds the full 2.5-3.4 GB matrix.
  * preflight() refuses to start a job when free RAM / VRAM is below what it needs,
    instead of failing an hour in.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import polars as pl

KAGGLE = Path(r"T:\Kaggle_Knee_Competition")
WORK = KAGGLE / "work"
CACHE = WORK / "cache320"
GP = Path(r"T:\Group_Project")
MANIFESTS = GP / "manifests"
RUNS = GP / "runs"
RESULTS = GP / "results"
TARGETS = WORK / "targets.parquet"          # v1 labels (decision 2026-10-25)
PIXELS = CACHE / "pixels_train.u8"
N_SLICES, PX = 819078, 320

sys.path.insert(0, str(KAGGLE / "src"))
from stage4_train_head import LABELS, KEYS, SLOTS, NL, Head, build_bags, macro_auc  # noqa: E402,F401
from stage4_compare import boot_ci, boot_delta  # noqa: E402,F401

SIZES = [65, 130, 260, 520, 1040, 2080, 2600]
SEEDS = [0, 1, 2]
ROTATIONS = [0, 1, 2, 3, 4]
FINDING_GROUPS = {
    "focal": ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Fracture"],
    "cartilage_oa": ["Medial OA", "Lateral OA", "PF OA"],
    "fluid_marrow_soft": ["Effusion", "Synovitis", "Baker's", "Contusion"],
}
MODELS = {
    "M1": dict(timm="vit_base_patch16_dinov3.lvd1689m", tag="dinov3_vitb16_224", dim=1536,
               kind="vit", desc="DINOv3 ViT-B/16"),
    "M2": dict(timm="convnext_base.dinov3_lvd1689m", tag="dinov3_convnextb_224", dim=2048,
               kind="cnn", desc="DINOv3 ConvNeXt-B"),
    "M3": dict(timm="convnext_base.fb_in22k_ft_in1k", tag="convnext_base_224", dim=2048,
               kind="cnn", desc="ImageNet-22k ConvNeXt-B"),
    "M4": dict(timm="vit_base_patch14_reg4_dinov2.lvd142m", tag="base_patch14_reg4_dinov2_224",
               dim=1536, kind="vit", desc="DINOv2 ViT-B/14"),
    # T4.9 lower bound: random init, fine-tune only, no frozen embeddings exist for these
    "M5": dict(timm="resnet18", tag=None, dim=1024, kind="cnn", desc="ResNet-18 from scratch", scratch=True),
    "M6": dict(timm="convnext_tiny", tag=None, dim=1536, kind="cnn", desc="ConvNeXt-T from scratch",
               scratch=True),
}


# --------------------------------------------------------------------------- data
def load_targets() -> pl.DataFrame:
    return pl.read_parquet(TARGETS)


def load_series_index() -> pl.DataFrame:
    return pl.read_parquet(CACHE / "index_series_train.parquet")


def load_rotation(r: int) -> dict:
    return json.loads((MANIFESTS / "rotations.json").read_text())[str(r)]


def load_subset(r: int, s: int, n: int) -> list[str]:
    d = json.loads((MANIFESTS / ("subsets_r%d_s%d.json" % (r, s))).read_text())
    return d[str(n)]


def label_arrays(tg: pl.DataFrame) -> dict:
    """Row-aligned with tg: soft targets T, weights W, proxy labels S + mask ADDR, gold G."""
    T = np.stack([tg["t_" + k].to_numpy() for k in KEYS], 1).astype(np.float32)
    W = np.stack([tg["w_" + k].to_numpy() for k in KEYS], 1).astype(np.float32)
    S = np.stack([(tg["s_" + k] == "pos").to_numpy() for k in KEYS], 1)
    ADDR = np.stack([tg["s_" + k].is_in(["pos", "neg"]).to_numpy() for k in KEYS], 1)
    G = np.stack([tg["g_" + k].fill_null(-1).to_numpy() for k in KEYS], 1).astype(np.int64)
    return dict(T=T, W=W, S=S, ADDR=ADDR, G=G)


def y_proxy(tg: pl.DataFrame, k: str):
    return (tg["s_" + k] == "pos").to_numpy(), tg["s_" + k].is_in(["pos", "neg"]).to_numpy()


def y_gold(tg: pl.DataFrame, k: str):
    return tg["g_" + k].to_numpy()


def fmt_lr(lr: float) -> str:
    return ("%.0e" % lr).replace("e-0", "e-")


# ---------------------------------------------------------------------- bootstrap
def boot_counts(n: int, b: int = 2000, seed: int = 0) -> np.ndarray:
    """(b, n) study multiplicities of b bootstrap resamples of n studies.

    One count matrix per split is shared by every finding, every model and every run
    scored on that split, so all of them see the SAME resamples: per-finding CIs, the
    macro CI and paired model-vs-model deltas all come from one consistent bootstrap.
    """
    idx = np.random.default_rng(seed).integers(0, n, (b, n))
    return np.bincount((np.arange(b)[:, None] * n + idx).ravel(), minlength=b * n
                       ).reshape(b, n).astype(np.float64)


def auc_counts(y: np.ndarray, p: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """ROC-AUC of p against y for every resample in cnt (b, n) -> (b,); nan where a
    resample holds one class only.

    Vectorized stand-in for calling roc_auc_score inside stage4_compare.boot_ci's loop
    (that loop costs ~1 s per finding per run; this is ~1000x faster). It is the
    Mann-Whitney statistic with each study weighted by its resample count, ties
    counted as one half, which is exactly what roc_auc_score returns on p[i], y[i].
    """
    order = np.argsort(p, kind="stable")
    ps, ys, cs = p[order], y[order].astype(bool), cnt[:, order]
    starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])       # tie groups
    wpos = np.add.reduceat(cs * ys, starts, axis=1)
    wneg = np.add.reduceat(cs * ~ys, starts, axis=1)
    below = np.cumsum(wneg, 1) - wneg
    num = (wpos * (below + 0.5 * wneg)).sum(1)
    den = wpos.sum(1) * wneg.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# ------------------------------------------------------------------------ memory
def ram_avail_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def gpu_free_gb() -> float:
    import torch
    free, _ = torch.cuda.mem_get_info()
    return free / 2 ** 30


def gpu_cap(frac: float = 0.90) -> float:
    """Cap this process's PyTorch allocations at `frac` of total VRAM. Returns GiB."""
    import torch
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    return torch.cuda.get_device_properties(0).total_memory * frac / 2 ** 30


def preflight(need_gpu_gb: float, need_ram_gb: float, what: str) -> None:
    import torch
    msg = []
    if torch.cuda.is_available():
        g = gpu_free_gb()
        if g < need_gpu_gb:
            msg.append("GPU free %.1f GiB < %.1f needed (is Ollama or another job loaded? "
                       "`ollama ps`, `nvidia-smi`)" % (g, need_gpu_gb))
    r = ram_avail_gb()
    if r < need_ram_gb:
        msg.append("RAM available %.1f GiB < %.1f needed (close browsers/Steam)" % (r, need_ram_gb))
    if msg:
        raise SystemExit("[preflight] %s refused to start:\n  " % what + "\n  ".join(msg))


class MemWatch:
    """Background sampler of process VRAM peak and system RAM minimum, for timing.json."""

    def __init__(self, every: float = 5.0, ram_warn_gb: float = 2.0):
        self.every, self.ram_warn = every, ram_warn_gb
        self.min_ram, self.stop_ev = 1e9, threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        warned = False
        while not self.stop_ev.wait(self.every):
            r = ram_avail_gb()
            self.min_ram = min(self.min_ram, r)
            if r < self.ram_warn and not warned:
                print("[memwatch] WARNING system RAM available %.1f GiB" % r, flush=True)
                warned = True

    def __enter__(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.min_ram = ram_avail_gb()
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop_ev.set()

    def summary(self) -> dict:
        import torch
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0
        res = torch.cuda.max_memory_reserved() / 2 ** 30 if torch.cuda.is_available() else 0
        return {"gpu_peak_alloc_gib": round(peak, 3), "gpu_peak_reserved_gib": round(res, 3),
                "ram_min_avail_gib": round(self.min_ram, 2)}


def load_emb_gpu(tag: str, chunk: int = 32768):
    """Stream emb_train_<tag>.f16 into a GPU float16 tensor without a full RAM copy."""
    import torch
    meta = json.loads((CACHE / ("emb_train_%s.json" % tag)).read_text())
    n, dim = meta["n"], meta["dim"]
    assert meta.get("done", n) == n, "%s embeddings incomplete (%s/%d)" % (tag, meta.get("done"), n)
    mm = np.memmap(CACHE / ("emb_train_%s.f16" % tag), dtype=np.float16, mode="r", shape=(n, dim))
    out = torch.empty((n, dim), dtype=torch.float16, device="cuda")
    for i in range(0, n, chunk):
        out[i:i + chunk] = torch.from_numpy(np.ascontiguousarray(mm[i:i + chunk])).to("cuda")
    del mm
    return out, meta


def pool_musd(model_id: str, r: int, se: pl.DataFrame | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean/sd of the frozen embeddings over rotation r's POOL slices.

    200k sampled slices (seeded by r), read from the memmap in sorted order, cached to
    results/musd/. Shared by the frozen and fine-tuned pipelines so both standardise
    the head input identically.
    """
    if MODELS[model_id].get("scratch"):           # no frozen features: identity standardisation
        d = MODELS[model_id]["dim"]
        return np.zeros(d, np.float32), np.ones(d, np.float32)
    out = RESULTS / "musd" / ("%s_r%d.npz" % (MODELS[model_id]["tag"], r))
    if out.exists():
        z = np.load(out)
        return z["mu"], z["sd"]
    se = load_series_index() if se is None else se
    pool = set(load_rotation(r)["pool"])
    rows = np.concatenate([np.arange(a, a + n) for s, a, n in
                           se.select("study_dir", "row0", "n").iter_rows() if s in pool])
    rows = np.sort(np.random.default_rng(r).choice(rows, min(200_000, len(rows)), replace=False))
    meta = json.loads((CACHE / ("emb_train_%s.json" % MODELS[model_id]["tag"])).read_text())
    mm = np.memmap(CACHE / ("emb_train_%s.f16" % MODELS[model_id]["tag"]), dtype=np.float16, mode="r",
                   shape=(meta["n"], meta["dim"]))
    s1 = np.zeros(meta["dim"]); s2 = np.zeros(meta["dim"])
    for i in range(0, len(rows), 20000):
        x = np.asarray(mm[rows[i:i + 20000]], dtype=np.float64)
        s1 += x.sum(0); s2 += (x * x).sum(0)
    mu = s1 / len(rows)
    sd = np.sqrt(np.maximum(s2 / len(rows) - mu * mu, 0)) + 1e-6
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    return mu.astype(np.float32), sd.astype(np.float32)


def write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    os.replace(tmp, path)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
