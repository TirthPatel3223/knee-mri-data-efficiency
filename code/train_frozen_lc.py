"""T3.1: one frozen-backbone learning-curve run = one study-level Head trained on a
training subset of one rotation, selected on that rotation's VALIDATION fold.

    python code/train_frozen_lc.py --model M3 --rotation 0 --seed 0 --n 2600 --lr 3e-4 --drop 0.2

Differences from stage4_train_head.py (which this reuses Head / build_bags / macro_auc from):
  * epoch selection on the val fold, never the test fold; test + gold scored once at the end
  * a nested training subset instead of four folds, with a step floor at small N
    (epochs = max(30, ceil(400*batch/N)) so N=65 still gets ~400 optimizer steps)
  * feature standardisation from the rotation's POOL slices only
  * bags live on the GPU as padded index tensors (FrozenCtx), so a batch is one gather;
    this is what makes the 1,680-run sweep affordable. Bag composition is exactly
    build_bags(cap=384).

`FrozenCtx` + `run_one` are imported by sweep_frozen.py, which keeps one model's
embeddings resident and loops over configs in a single process.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from common import (KEYS, LABELS, MODELS, NL, RUNS, Head, MemWatch, build_bags, fmt_lr,
                    gpu_cap, label_arrays, load_emb_gpu, load_rotation, load_series_index,
                    load_subset, load_targets, macro_auc, now, pool_musd, preflight, write_json)

CAP = 384


class FrozenCtx:
    """One backbone's embeddings + every study's bag, resident on the GPU."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.spec = MODELS[model_id]
        gb = 819078 * self.spec["dim"] * 2 / 2 ** 30
        preflight(gb + 1.5, 3.0, "frozen %s" % model_id)
        self.emb, self.meta = load_emb_gpu(self.spec["tag"])
        self.dim = self.meta["dim"]

        self.tg = load_targets()
        self.se = se = load_series_index()
        bags = build_bags(se, CAP, np.random.default_rng(0))
        self.sids = self.tg["study_dir"].to_list()
        self.row = {s: i for i, s in enumerate(self.sids)}
        S = len(self.sids)
        L = max(len(bags[s][0]) for s in self.sids)
        ix = np.zeros((S, L), np.int64); sl = np.zeros((S, L), np.int64)
        po = np.zeros((S, L), np.float32); mk = np.zeros((S, L), bool)
        self.blen = np.zeros(S, np.int64)
        for i, s in enumerate(self.sids):
            a, b, c = bags[s]
            n = len(a)
            ix[i, :n], sl[i, :n], po[i, :n], mk[i, :n] = a, b, c, True
            self.blen[i] = n
        dev = "cuda"
        self.IX, self.SL = torch.from_numpy(ix).to(dev), torch.from_numpy(sl).to(dev)
        self.PO, self.MK = torch.from_numpy(po).to(dev), torch.from_numpy(mk).to(dev)
        self.lab = label_arrays(self.tg)
        self.Tt = torch.from_numpy(self.lab["T"]).to(dev)
        self.Wt = torch.from_numpy(self.lab["W"]).to(dev)
        self.gold = [s for s, g in zip(self.sids, self.tg["is_gold"]) if g]
        self.cur_r, self._plans = None, {}

    def musd(self, r: int):
        """Standardise the resident matrix IN PLACE with rotation r's pool mu/sd.

        Done once per rotation (the sweep is rotation-outer), so a batch is then a single
        fp16 gather instead of gather + float32 cast + subtract + divide, which was ~2/3 of
        the run time. Switching rotation reloads the raw matrix from disk (~2 s) rather
        than un-standardising, so no fp16 rounding accumulates.
        """
        if self.cur_r != r:
            if self.cur_r is not None:
                del self.emb
                torch.cuda.empty_cache()
                self.emb, _ = load_emb_gpu(self.spec["tag"])
            mu, sd = (torch.from_numpy(a).cuda() for a in pool_musd(self.model_id, r, self.se))
            for i in range(0, len(self.emb), 65536):
                self.emb[i:i + 65536] = ((self.emb[i:i + 65536].float() - mu) / sd).half()
            self.cur_r = r
        return None, None

    def batch(self, rows_np: np.ndarray, mu, sd, rows: torch.Tensor | None = None):
        """rows_np: study rows (numpy). Bag length is looked up on the CPU, so building a
        batch never forces a GPU->CPU sync and kernels stay queued back to back."""
        L = int(self.blen[rows_np].max())
        if rows is None:
            rows = torch.from_numpy(rows_np).cuda(non_blocking=True)
        # padded positions point at row 0; with layers=0 they cannot leak: Head masks them
        # out of the attention softmax, the only place slices interact
        ix, mk = self.IX[rows, :L], self.MK[rows, :L]
        x = self.emb[ix]
        return x, self.SL[rows, :L], self.PO[rows, :L], mk, rows

    def _plan(self, sids, bs):
        """Length-sorted eval batches for a split, built once and reused every epoch."""
        # keyed by CONTENT. The first version used id(sids): CPython reuses the id of a freed
        # list, and val and test both have 870 studies, so a run could pick up another
        # split's plan and score predictions against the wrong studies (AUC ~0.5,
        # 89/1,680 runs, found 2026-10-26)
        key = (tuple(sids), bs)
        if key not in self._plans:
            rows = np.array([self.row[s] for s in sids])
            order = np.argsort(self.blen[rows], kind="stable")
            self._plans[key] = [(rows[order[i:i + bs]], torch.from_numpy(order[i:i + bs]).cuda(),
                                 torch.from_numpy(rows[order[i:i + bs]]).cuda())
                                for i in range(0, len(sids), bs)]
        return self._plans[key]

    @torch.no_grad()
    def predict(self, model, sids, mu, sd, bs: int = 96) -> np.ndarray:
        model.eval()
        out = torch.empty((len(sids), NL), device="cuda")
        for rows_np, pos, rows in self._plan(sids, bs):
            x, sl, po, mk, _ = self.batch(rows_np, mu, sd, rows)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, _ = model(x, sl, po, mk)
            out[pos] = torch.sigmoid(lg.float())
        return out.cpu().numpy()


def run_name(model_id, r, s, n, lr, drop) -> str:
    return "frozen-%s-r%d-s%d-n%04d-lr%s_dr%s" % (model_id, r, s, n, fmt_lr(lr), drop)


def score(ctx: FrozenCtx, sids, p):
    ix = [ctx.row[s] for s in sids]
    return macro_auc(ctx.lab["S"][ix], p, ctx.lab["ADDR"][ix])


def save_preds(path, sids, p):
    d = {"study_dir": sids}
    d.update({k: p[:, j].astype(np.float32) for j, k in enumerate(KEYS)})
    pl.DataFrame(d).write_parquet(path)


def run_one(ctx: FrozenCtx, r: int, s: int, n: int, lr: float, drop: float,
            wd: float = 0.01, d: int = 256, force: bool = False, verbose: bool = False) -> dict | None:
    name = run_name(ctx.model_id, r, s, n, lr, drop)
    out = RUNS / name
    if (out / "metrics.json").exists() and not force:
        return None
    out.mkdir(parents=True, exist_ok=True)
    t0, started = time.perf_counter(), now()

    rot = load_rotation(r)
    train = load_subset(r, s, n)
    val, test = rot["val"], rot["test"]
    mu, sd = ctx.musd(r)
    seed = 1000 * r + 10 * s + 7
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    bs = min(24, max(8, n // 8))
    spe = max(1, n // bs)
    epochs = max(30, math.ceil(400 * bs / n))
    steps = epochs * spe
    model = Head(ctx.dim, d, 0, drop=drop).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.15)
    trows = np.array([ctx.row[x] for x in train])

    best, log = (-1.0, None, -1), []
    with MemWatch(every=10) as mw:
        for ep in range(epochs):
            model.train()
            perm = rng.permutation(n)
            tot = 0.0
            for i in range(spe):
                x, sl, po, mk, b = ctx.batch(trows[perm[i * bs:(i + 1) * bs]], mu, sd)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _ = model(x, sl, po, mk)
                w = ctx.Wt[b]
                loss = (F.binary_cross_entropy_with_logits(lg.float(), ctx.Tt[b], reduction="none")
                        * w).sum() / w.sum().clamp(min=1)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sched.last_epoch < steps - 1:
                    sched.step()
                tot += loss.detach()
            vm, _ = score(ctx, val, ctx.predict(model, val, mu, sd))
            log.append({"epoch": ep, "loss": float(tot) / spe, "val_macro": vm})
            if vm > best[0]:
                best = (vm, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print("  ep %3d loss %.4f val %.4f" % (ep, float(tot) / spe, vm), flush=True)

        model.load_state_dict(best[1])
        pv = ctx.predict(model, val, mu, sd)
        pt = ctx.predict(model, test, mu, sd)
        pg = ctx.predict(model, ctx.gold, mu, sd)
    vm, _ = score(ctx, val, pv)
    tm, tper = score(ctx, test, pt)
    gi = [ctx.row[x] for x in ctx.gold]
    gm, gper = macro_auc(ctx.lab["G"][gi], pg)
    wall = time.perf_counter() - t0

    tix, teix = [ctx.row[x] for x in train], [ctx.row[x] for x in test]
    cfg = {"regime": "frozen", "model": ctx.model_id, "timm": ctx.spec["timm"], "emb_tag": ctx.spec["tag"],
           "rotation": r, "seed": s, "n_train": n, "lr": lr, "drop": drop, "wd": wd, "d": d,
           "layers": 0, "cap": CAP, "batch": bs, "epochs": epochs, "steps": steps, "torch_seed": seed,
           "torch": torch.__version__, "labels": "v1 targets.parquet"}
    write_json(out / "config.json", cfg)
    pl.DataFrame(log).write_csv(out / "train_log.csv")
    save_preds(out / "preds_val.parquet", val, pv)
    save_preds(out / "preds_test.parquet", test, pt)
    save_preds(out / "preds_gold.parquet", ctx.gold, pg)
    timing = {"start": started, "end": now(), "wall_s": round(wall, 2), "gpu_h": wall / 3600, **mw.summary()}
    write_json(out / "timing.json", timing)
    m = {"run": name, "regime": "frozen", "model": ctx.model_id, "rotation": r, "seed": s, "n_train": n,
         "hp": {"lr": lr, "drop": drop}, "best_epoch_or_step": best[2],
         "val_macro": vm, "test_macro": tm, "gold_macro": gm,
         "test_per_finding": tper, "gold_per_finding": gper,
         "test_n_pos": {l: int(ctx.lab["S"][teix, j][ctx.lab["ADDR"][teix, j]].sum()) for j, l in enumerate(LABELS)},
         "train_n_pos": {l: int(ctx.lab["S"][tix, j].sum()) for j, l in enumerate(LABELS)},
         "wall_s": round(wall, 2), "gpu_h": wall / 3600, "kwh": None}
    write_json(out / "metrics.json", m)          # written last = the run is complete
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--rotation", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    print("[frozen] VRAM cap %.1f GiB" % gpu_cap())
    ctx = FrozenCtx(args.model)
    m = run_one(ctx, args.rotation, args.seed, args.n, args.lr, args.drop, force=args.force, verbose=True)
    if m is None:
        print("[frozen] %s already finished; pass --force to overwrite"
              % run_name(args.model, args.rotation, args.seed, args.n, args.lr, args.drop))
        return 1
    print("[frozen] %s  best ep %d  val %.4f  test %.4f  gold %.4f  %.1fs"
          % (m["run"], m["best_epoch_or_step"], m["val_macro"], m["test_macro"], m["gold_macro"], m["wall_s"]))
    t = __import__("json").loads((RUNS / m["run"] / "timing.json").read_text())
    print("[frozen] GPU peak %.2f GiB allocated, RAM min available %.1f GiB"
          % (t["gpu_peak_alloc_gib"], t["ram_min_avail_gib"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
