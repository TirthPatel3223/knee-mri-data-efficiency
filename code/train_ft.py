"""T4.3: one fine-tuning learning-curve run (backbone + study Head end to end).

    python code/train_ft.py --model M1 --rotation 0 --seed 0 --n 520 --lr 3e-5
    python code/train_ft.py --model M2 --rotation 0 --seed 0 --n 130 --lr 3e-5 --views 1000   # smoke

Recipe (PROJECT_PLAN 5.4 / 5.8): AdamW (wd 0.05, none on norms/biases/tokens), layer-wise
LR decay 0.75, head LR 10x the top backbone LR, bf16 autocast, (optional) gradient checkpointing,
drop_path 0.1, grad-clip 1.0, effective batch 16 studies via accumulation. LP-FT: the
backbone is frozen for the first 10 % of optimizer steps, then warms up over 5 % and
follows the same cosine as the head. Budget V(N) = clip(10 N, 6000, 26000) study-views.
Validation (K evenly spaced slices/series) 10x per run; the best val-macro checkpoint is
scored once on test + gold with all-slice bags (cap 384), the frozen head's bags.

MEMORY: every training micro-batch is at most micro*48 slices (ft_data.MAX_SLICES), so
peak VRAM is a fixed number measured by bench_ft.py; eval encodes in fixed chunks under
no_grad. The process is capped at 90 % VRAM, so a surprise becomes a catchable OOM: the
micro-step is dropped and logged, and after 5 the run checkpoints and exits (code 3).
RESUME: last.pt every ~30 min and after each validation; rerun the same command.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time

# Before torch initialises CUDA (2026-10-26 memcheck). Under WDDM, reserved VRAM is charged to host
# commit. M2 allocates <= 7.4 GiB, but fragmentation grew the cache to ~10.5 GiB and commit to
# 38.9/42.4 GiB. Reclaim unused cached blocks above 80 % of the process cap, and split big blocks less.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:256")

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (KEYS, LABELS, MODELS, RUNS, MemWatch, fmt_lr, gpu_cap, label_arrays, load_rotation,
                    load_series_index, load_subset, load_targets, macro_auc, now, pool_musd, preflight,
                    write_json)
from ft_data import StudyDataset, ViewSampler, collate, gpu_prep
from ft_model import StudyModel, param_groups

EFF_BATCH = 16


def budget(n: int, scratch: bool = False) -> int:
    """V(N) study-views; from-scratch models get 3x (PROJECT_PLAN T4.9)."""
    return int(min(max(10 * n, 6000), 26000)) * (3 if scratch else 1)


def run_name(a) -> str:
    tag = "-partial" if a.partial else ""
    tag += "-v%d" % a.views if a.views else ""
    return "ft-%s-r%d-s%d-n%04d-lr%s%s" % (a.model, a.rotation, a.seed, a.n, fmt_lr(a.lr), tag)


def lr_factor(step: int, total: int, part: str, lp_frac: float = 0.10) -> float:
    warm = max(1, round(0.05 * total))
    lp = round(lp_frac * total)
    cos = 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, step - warm) / max(1, total - warm))))
    if part == "head":
        return min(1.0, (step + 1) / warm) * (cos if step >= warm else 1.0)
    if step < lp:
        return 0.0
    return min(1.0, (step - lp + 1) / warm) * cos


def loader(ds, batch, workers, sampler=None):
    # pin_memory OFF (2026-10-26 memcheck): torch's pinned-host cache keeps a page-locked block
    # per batch size it has seen and never frees it; with variable-length studies that grew the
    # main process to ~11 GB of commit. Unpinned H2D of a ~5 MB uint8 batch costs ~1 ms.
    return DataLoader(ds, batch_size=batch, sampler=sampler, shuffle=False, collate_fn=collate,
                      num_workers=workers, pin_memory=False, prefetch_factor=2 if workers else None,
                      persistent_workers=False,
                      # 2026-10-26: an eval worker died under memory pressure and the main process
                      # waited forever. With a timeout it raises instead, and sweep_ft retries the run
                      # (resuming from last.pt)
                      timeout=600 if workers else 0)


@torch.no_grad()
def predict(model, ds, workers, chunk=128) -> np.ndarray:
    model.eval()
    out = np.zeros((len(ds), len(KEYS)), np.float32)
    for b in loader(ds, 2, workers):
        px = b["px"].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = torch.cat([model.enc(gpu_prep(px[i:i + chunk], None, model.enc.mean, model.enc.std))
                               .float() for i in range(0, len(px), chunk)])
            lg = model.head_forward(feats, b["slot"].cuda(), b["pos"].cuda(), b["mask"].cuda())
        out[b["i"].numpy()] = torch.sigmoid(lg.float()).cpu().numpy()
    return out


def save_preds(path, sids, p):
    d = {"study_dir": list(sids)}
    d.update({k: p[:, j] for j, k in enumerate(KEYS)})
    pl.DataFrame(d).write_parquet(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--rotation", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True, help="top backbone LR")
    ap.add_argument("--k", type=int, default=8, help="slices per series in train/val")
    ap.add_argument("--micro", type=int, default=1, help="studies per micro-batch")
    ap.add_argument("--views", type=int, default=0, help="override the study-view budget (smoke tests)")
    ap.add_argument("--n-val", type=int, default=10)
    ap.add_argument("--partial", action="store_true", help="freeze the lower half of the backbone")
    # T4.4 (2026-10-25): micro 1 without checkpointing fits every model with >= 3.3 GiB headroom
    # and is ~20 % faster, so checkpointing is opt-in
    ap.add_argument("--ckpt", dest="ckpt", action="store_true", help="enable gradient checkpointing")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--eval-workers", type=int, default=2)
    ap.add_argument("--val-only", action="store_true", help="LR pilot: skip the final test/gold scoring")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    name = run_name(a)
    out = RUNS / name
    prev = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else None
    if prev and ("test_macro" in prev or a.val_only) and not a.force:
        print("[ft] %s already finished; pass --force to rerun" % name)
        return 0
    # a val-only LR-pilot run that won the pilot: no retraining, just score test + gold once
    finalize = prev is not None and not a.force and (out / "best.pt").exists()
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    cap = gpu_cap(0.90)
    preflight(9.0, 9.0, "ft " + name)
    print("[ft] %s  %s  VRAM cap %.1f GiB" % (now(), name, cap), flush=True)

    tg, se = load_targets(), load_series_index()
    rot = load_rotation(a.rotation)
    train, val, test = load_subset(a.rotation, a.seed, a.n), rot["val"], rot["test"]
    gold = tg.filter(pl.col("is_gold"))["study_dir"].to_list()
    lab = label_arrays(tg)
    row = {s: i for i, s in enumerate(tg["study_dir"].to_list())}

    seed = 1000 * a.rotation + 10 * a.seed + 7
    torch.manual_seed(seed)
    mu, sd = pool_musd(a.model, a.rotation, se)
    model = StudyModel(a.model, torch.from_numpy(mu), torch.from_numpy(sd), ckpt=a.ckpt).cuda()
    groups = param_groups(model, a.lr, partial=a.partial)
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    scratch = bool(MODELS[a.model].get("scratch"))
    lp_frac = 0.0 if scratch else 0.10          # LP-FT protects pretrained features; scratch has none
    views = a.views or budget(a.n, scratch)
    steps = math.ceil(views / EFF_BATCH)
    accum = EFF_BATCH // a.micro
    val_at = sorted({max(1, round(steps * j / a.n_val)) for j in range(1, a.n_val + 1)})
    gen = torch.Generator(device="cuda"); gen.manual_seed(seed)

    step, best_val, best_step, log, ooms, train_s = 0, -1.0, -1, [], 0, 0.0
    last = out / "last.pt"
    if finalize:
        step, best_val, best_step = steps, prev["val_macro"], prev["best_epoch_or_step"]
        log = pl.read_csv(out / "train_log.csv").to_dicts()
        print("[ft] finalizing pilot run (val %.4f @ %d): scoring test + gold" % (best_val, best_step))
    elif last.exists():
        ck = torch.load(last, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, best_val, best_step, log = ck["step"], ck["best_val"], ck["best_step"], ck["log"]
        train_s, ooms = ck.get("train_s", 0.0), ck.get("ooms", 0)
        # RNG states must be CPU ByteTensors; map_location="cuda" moved them
        torch.set_rng_state(ck["rng_cpu"].cpu()); gen.set_state(ck["rng_gen"].cpu())
        print("[ft] resumed at step %d/%d (best val %.4f @ %d)" % (step, steps, best_val, best_step))

    def checkpoint():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "best_val": best_val, "best_step": best_step, "log": log, "train_s": train_s,
                    "ooms": ooms, "rng_cpu": torch.get_rng_state(), "rng_gen": gen.get_state()},
                   str(last) + ".tmp")
        import os
        os.replace(str(last) + ".tmp", last)

    cfg = {"regime": "ft", "model": a.model, "timm": MODELS[a.model]["timm"], "rotation": a.rotation,
           "seed": a.seed, "n_train": a.n, "lr_top": a.lr, "llrd": 1.0 if scratch else 0.75, "head_mult": 1 if scratch else 10, "scratch": scratch, "wd": 0.05,
           "drop_path": 0.1, "head_drop": 0.2, "k": a.k, "max_slices": 48, "micro": a.micro,
           "eff_batch": EFF_BATCH, "views": views, "steps": steps, "lp_frac": lp_frac, "warm_frac": 0.05,
           "partial": a.partial, "grad_ckpt": a.ckpt, "val_at": val_at, "torch_seed": seed,
           "torch": torch.__version__, "labels": "v1 targets.parquet"}
    if not finalize:
        write_json(out / "config.json", cfg)
    smi = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,power.draw,utilization.gpu,memory.used",
                            "--format=csv", "-l", "5"], stdout=open(out / "gpu.csv", "a"),
                           stderr=subprocess.DEVNULL)
    t_start, started = time.perf_counter(), now()
    val_ds = StudyDataset(val, "val", a.k, seed, tg, se)
    S_val, A_val = lab["S"][[row[s] for s in val]], lab["ADDR"][[row[s] for s in val]]

    with MemWatch(every=10) as mw:
        if step < steps:
            ds = StudyDataset(train, "train", a.k, seed + step, tg, se)
            dl = loader(ds, a.micro, a.workers, ViewSampler(len(train), steps * EFF_BATCH, seed,
                                                            skip=step * EFF_BATCH))
            t0, t_ck, micro_i, run_loss, nsl, n_mb = time.perf_counter(), time.perf_counter(), 0, 0.0, 0, 0
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_factor(step, steps, g["part"], lp_frac)
            for b in dl:
                frozen = step < round(lp_frac * steps)
                px = b["px"].cuda(non_blocking=True)
                try:
                    x = gpu_prep(px, b["series"].cuda(), model.enc.mean, model.enc.std, aug=True, gen=gen)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if frozen:
                            model.enc.eval()
                            with torch.no_grad():
                                feats = model.enc(x)
                        else:
                            model.enc.train()
                            feats = model.enc(x)
                        model.head.train()
                        lg = model.head_forward(feats, b["slot"].cuda(), b["pos"].cuda(), b["mask"].cuda())
                    t, w = b["t"].cuda(), b["w"].cuda()
                    loss = (F.binary_cross_entropy_with_logits(lg.float(), t, reduction="none") * w).sum() \
                        / w.sum().clamp(min=1)
                    (loss / accum).backward()
                except torch.cuda.OutOfMemoryError:
                    ooms += 1
                    del px
                    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                    print("[ft] OOM #%d at step %d (%d slices) -- micro-step dropped" % (ooms, step, len(b["slot"].flatten())),
                          flush=True)
                    if ooms > 5:
                        checkpoint(); smi.terminate()
                        print("[ft] too many OOMs; rerun with --micro 1 / --k 6"); return 3
                    micro_i = 0
                    continue
                run_loss += loss.item(); n_mb += 1; nsl += int(b["mask"].sum()); micro_i += 1
                if micro_i < accum:
                    continue
                micro_i = 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * lr_factor(step, steps, g["part"], lp_frac)
                if step % 25 == 0:
                    el = time.perf_counter() - t0
                    print("[ft] step %4d/%d loss %.4f  %.0f sl/s  %s  %.2f GiB peak" %
                          (step, steps, run_loss / max(1, n_mb), nsl / el, "LP" if frozen else "FT",
                           torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
                    log.append({"step": step, "loss": run_loss / max(1, n_mb), "sl_per_s": nsl / el,
                                "val_macro": None})
                    train_s += el; t0, run_loss, nsl, n_mb = time.perf_counter(), 0.0, 0, 0
                if step in val_at:
                    tv = time.perf_counter()
                    # WDDM charges reserved VRAM against host commit; validation needs ~2 GiB, so
                    # hand the training cache back before the eval worker spawns (the commit peak)
                    torch.cuda.empty_cache()
                    vm, _ = macro_auc(S_val, predict(model, val_ds, a.eval_workers), A_val)
                    log.append({"step": step, "loss": None, "sl_per_s": None, "val_macro": vm})
                    if vm > best_val:
                        best_val, best_step = vm, step
                        torch.save(model.state_dict(), out / "best.pt")
                    print("[ft] VAL step %d  macro %.4f  (best %.4f @ %d)  %.0fs" %
                          (step, vm, best_val, best_step, time.perf_counter() - tv), flush=True)
                    checkpoint(); t_ck = time.perf_counter(); t0 = time.perf_counter(); nsl = 0; run_loss = 0.0; n_mb = 0
                elif time.perf_counter() - t_ck > 1800:
                    checkpoint(); t_ck = time.perf_counter()
                if step >= steps:
                    break
            del dl
            torch.cuda.empty_cache()

        tr_wall = time.perf_counter() - t_start
        model.load_state_dict(torch.load(out / "best.pt", map_location="cuda"))
        te = time.perf_counter()
        if a.val_only:
            pt = pg = None
        else:
            pt = predict(model, StudyDataset(test, "test", a.k, 0, tg, se), a.eval_workers)
            pg = predict(model, StudyDataset(gold, "test", a.k, 0, tg, se), a.eval_workers)
        eval_s = time.perf_counter() - te
    smi.terminate()
    wall = time.perf_counter() - t_start
    if finalize:                                  # keep the pilot's training time in the totals
        pt_ = json.loads((out / "timing.json").read_text())
        tr_wall, wall = pt_["train_wall_s"], pt_["wall_s"] + wall

    m = {"run": name, "regime": "ft", "model": a.model, "rotation": a.rotation, "seed": a.seed,
         "n_train": a.n, "hp": {"lr": a.lr}, "best_epoch_or_step": best_step, "val_macro": best_val,
         "train_n_pos": {l: int(lab["S"][[row[s] for s in train], j].sum()) for j, l in enumerate(LABELS)}}
    if pt is not None:
        ti = [row[s] for s in test]
        tm, tper = macro_auc(lab["S"][ti], pt, lab["ADDR"][ti])
        gm, gper = macro_auc(lab["G"][[row[s] for s in gold]], pg)
        save_preds(out / "preds_test.parquet", test, pt)
        save_preds(out / "preds_gold.parquet", gold, pg)
        m.update({"test_macro": tm, "gold_macro": gm, "test_per_finding": tper, "gold_per_finding": gper,
                  "test_n_pos": {l: int(lab["S"][ti, j][lab["ADDR"][ti, j]].sum()) for j, l in enumerate(LABELS)}})
    pl.DataFrame(log, infer_schema_length=None).write_csv(out / "train_log.csv")
    w_ = []                                       # nvidia-smi rows: "ts, 99.79 W, 99 %, 1994 MiB"
    for line in open(out / "gpu.csv", encoding="utf-8", errors="ignore"):
        try:
            w_.append(float(line.split(",")[1].strip().split()[0]))
        except (IndexError, ValueError):
            pass                                  # header lines (one per resume) and blanks
    watts = float(np.mean(w_)) if w_ else None
    kwh = watts * wall / 3600 / 1000 if watts else None
    timing = {"start": started, "end": now(), "wall_s": round(wall, 1), "train_wall_s": round(tr_wall, 1),
              "final_eval_s": round(eval_s, 1), "gpu_h": wall / 3600, "mean_w": watts, "kwh": kwh,
              "ooms": ooms, **mw.summary()}
    write_json(out / "timing.json", timing)
    m.update({"wall_s": round(wall, 1), "gpu_h": wall / 3600, "kwh": kwh})
    write_json(out / "metrics.json", m)
    if last.exists():
        last.unlink()
    print("[ft] DONE %s  best val %.4f @ %d  test %s  gold %s  %.2f h  peak %.2f GiB" %
          (name, best_val, best_step, "%.4f" % m["test_macro"] if pt is not None else "-",
           "%.4f" % m["gold_macro"] if pt is not None else "-", wall / 3600, timing["gpu_peak_alloc_gib"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
