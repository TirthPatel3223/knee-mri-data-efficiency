"""T4.2: fine-tunable slice encoder + the shared study Head, and LLRD parameter groups.

    python code/ft_model.py --equiv M1     # embedding-equivalence test vs the cached frozen features

SliceEncoder pools exactly like stage4_embed: ViT -> [CLS ; mean(patch tokens after the
prefix tokens)], CNN -> [GAP ; GMP] of forward_features. Features are standardised with
the SAME pool-slice mu/sd as the frozen run before entering Head, so at step 0 the
fine-tuned model's head sees the frozen model's input distribution.
"""
from __future__ import annotations

import argparse

import numpy as np
import timm
import torch
import torch.nn as nn

from common import CACHE, MODELS, N_SLICES, PIXELS, PX, Head


class SliceEncoder(nn.Module):
    def __init__(self, timm_name: str, drop_path: float = 0.1, ckpt: bool = True, size: int = 224,
                 pretrained: bool = True):
        super().__init__()
        try:
            self.m = timm.create_model(timm_name, pretrained=pretrained, num_classes=0,
                                       drop_path_rate=drop_path, img_size=size)
        except TypeError:
            self.m = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, drop_path_rate=drop_path)
        self.m.set_grad_checkpointing(ckpt)
        self.npre = getattr(self.m, "num_prefix_tokens", 1)
        cfg = self.m.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"], dtype=torch.float32), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"], dtype=torch.float32), persistent=False)

    def forward(self, x):
        f = self.m.forward_features(x)
        if f.ndim == 3:
            return torch.cat([f[:, 0], f[:, self.npre:].mean(1)], 1)
        return torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)


class StudyModel(nn.Module):
    def __init__(self, model_id: str, mu: torch.Tensor, sd: torch.Tensor, d: int = 256,
                 drop: float = 0.2, drop_path: float = 0.1, ckpt: bool = True):
        super().__init__()
        spec = MODELS[model_id]
        self.kind = spec["kind"]
        self.scratch = bool(spec.get("scratch"))
        self.enc = SliceEncoder(spec["timm"], drop_path, ckpt, pretrained=not self.scratch)
        self.head = Head(spec["dim"], d, 0, drop=drop)
        self.register_buffer("mu", mu.float().clone())
        self.register_buffer("sd", sd.float().clone())

    def head_forward(self, feats, slot, pos, mask):
        """feats (M,dim) for the M real slices, row-major over mask -> logits (B,12)."""
        x = feats.new_zeros(mask.shape + (feats.shape[1],))
        x[mask] = feats.to(x.dtype)
        x = ((x.float() - self.mu) / self.sd) * mask[..., None]
        lg, _ = self.head(x, slot, pos, mask)
        return lg

    @torch.no_grad()
    def encode_nograd(self, x, chunk: int = 128):
        return torch.cat([self.enc(x[i:i + chunk]).float() for i in range(0, len(x), chunk)])


def layer_id(name: str, kind: str, top: int) -> int:
    n = name.split("enc.m.", 1)[1]
    if kind == "vit":
        if n.startswith(("cls_token", "reg_token", "pos_embed", "patch_embed", "rope")):
            return 0
        if n.startswith("blocks."):
            return int(n.split(".")[1]) + 1
        return top
    if n.startswith("stem"):
        return 0
    if n.startswith("stages."):
        return int(n.split(".")[1]) + 1                 # one LR per ConvNeXt stage
    return top


def param_groups(model: StudyModel, lr: float, wd: float = 0.05, decay: float = 0.75,
                 head_mult: float = 10.0, partial: bool = False) -> list[dict]:
    """LLRD groups. Backbone top layer gets `lr`, each layer below x`decay`; head 10x.

    partial=True freezes the lower half of the backbone (the T4.4 fallback), the same
    way for both architectures.
    """
    m = model.enc.m
    if model.scratch:                 # random init: one LR everywhere, nothing to protect
        decay, head_mult, partial = 1.0, 1.0, False
        nl = 0
    else:
        nl = len(m.blocks) if model.kind == "vit" else len(m.stages)
    top = nl + 1
    groups: dict[tuple, dict] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_wd = p.ndim <= 1 or name.endswith(("cls_token", "reg_token", "pos_embed"))
        if name.startswith("head."):
            key, scale, part = ("head", no_wd), head_mult, "head"
        elif model.scratch:
            key, scale, part = ("backbone", no_wd), 1.0, "backbone"
        else:
            lid = layer_id(name, model.kind, top)
            if partial and lid <= nl // 2:
                p.requires_grad_(False)
                continue
            key, scale, part = (lid, no_wd), decay ** (top - lid), "backbone"
        g = groups.setdefault(key, {"params": [], "lr": lr * scale, "base_lr": lr * scale,
                                    "weight_decay": 0.0 if no_wd else wd, "part": part})
        g["params"].append(p)
    return list(groups.values())


def equivalence_test(model_id: str, n: int = 16) -> float:
    """Eval-mode features of random slices vs the cached frozen embeddings (cosine)."""
    import json
    from ft_data import gpu_prep
    spec = MODELS[model_id]
    meta = json.loads((CACHE / ("emb_train_%s.json" % spec["tag"])).read_text())
    emb = np.memmap(CACHE / ("emb_train_%s.f16" % spec["tag"]), dtype=np.float16, mode="r",
                    shape=(meta["n"], meta["dim"]))
    pix = np.memmap(PIXELS, dtype=np.uint8, mode="r", shape=(N_SLICES, PX, PX))
    rows = np.sort(np.random.default_rng(0).choice(min(N_SLICES, meta.get("done", meta["n"])), n, replace=False))
    enc = SliceEncoder(spec["timm"], 0.0, False).cuda().eval()
    px = torch.from_numpy(np.ascontiguousarray(pix[rows])).cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        f = enc(gpu_prep(px, torch.zeros(n, dtype=torch.long, device="cuda"), enc.mean, enc.std)).float()
    ref = torch.from_numpy(np.asarray(emb[rows], dtype=np.float32)).cuda()
    cos = torch.nn.functional.cosine_similarity(f, ref, dim=1)
    print("[equiv] %s  cosine min %.5f  mean %.5f over %d slices" % (model_id, cos.min(), cos.mean(), n))
    return float(cos.min())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--equiv", nargs="+", default=["M1", "M2"])
    a = ap.parse_args()
    ok = all(equivalence_test(m) > 0.999 for m in a.equiv)
    print("[equiv] PASS" if ok else "[equiv] FAIL")
    raise SystemExit(0 if ok else 1)
