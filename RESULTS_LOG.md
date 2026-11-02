# Results log

Append-only. Date, task ID, command, key numbers, issues, which chat/tool wrote the code.

## 2026-10-25 (evening): setup, embeddings, frozen sanity (training chat, Claude Code; all code in `code/` AI-written)
- **Environment:** system Python 3.11.9, torch 2.11.0.dev+cu128, timm 1.0.28, sklearn 1.9.1. Run scripts with `python -X utf8` (NOT `-I`: `-I` hides the user site-packages, which hold httpx and tqdm). Installed `httpx` and `psutil` (needed by huggingface_hub and the memory guard).
- **Hardware at start:** RTX 5070 Ti Laptop, 12,227 MiB (11.9 GiB usable); RAM 31.4 GB with ~16.7 GB free while normal apps are open; T: 301 GB free.
  **RAM is the binding constraint:** each Windows DataLoader worker costs ~1.7 GB private. The 6-worker embed job left 0.4 GB free, so the M2 pass was run with 4 workers.
- **T0.1, T0.4, T1.1, T1.2 done:** `code/common.py`, `make_splits.py`, `make_subsets.py`. Rotations: pools 2,609–2,610; asserts pass; subsets are bit-identical on rerun. 77 (r, s, N, finding) cells have < 5 training positives.
- **T0.3:** `convnext_base.dinov3_lvd1689m` downloaded (87.6M params, ImageNet mean/std, 224).
- **T2.1 done:** `emb_train_dinov3_vitb16_224` (1536-d), 819,078 slices in 23.9 min (570 sl/s); nonfinite 0, all-zero rows 0.
- **T2.3:** the existing 224 embeddings are full size; their sidecars predate the `done` field.
- **T3.1/T3.2 done:** `train_frozen_lc.py`. Sanity at r0 s0 N=2600, lr 3e-4, drop 0.2: M3 test 0.7664 (pilot 0.791, Δ −0.025), M4 test 0.7518 (pilot 0.766, Δ −0.014). Both are within ±0.03, and M3 > M4 holds, so PASS. Gold: M3 0.795, M4 0.764. GPU peak 3.7 GiB.
- **Memory safety:** 90% VRAM cap per process; `preflight()`; `MemWatch` peaks go into every `timing.json`; `code/memlog.py` logs RAM, commit, VRAM, temp and power every minute to `runs/logs/memlog.csv`; a 30-min check runs `memlog.py --summary 30`. FT tuning without restarts goes through `runs/ft_overrides.json`.
- **T2.2 done:** `emb_train_dinov3_convnextb_224` (2048-d, [GAP ; GMP] of a 1024×7×7 map). 23.6 min at 578 sl/s with 4 workers; clean.
- **Frozen speed (optimized):** one 7-size curve ≈ 50 s, so the full 1,680-run sweep ≈ 3.3 h. Rotation 0, seed 0, lr 3e-4, drop 0.2, test macro by N (65→2600):
  - M1: .704 .716 .740 .742 .757 .775 .767
  - M2: .698 .700 .725 .724 .708 .739 .733
- **T4.4/T4.5 done:** see the §15 decision row for the benchmark.
  - Smoke M5 (40 steps): full path OK.
  - Smoke M1 (50 steps, val-only, then finalize): val 0.612, test 0.672; FT at 198 sl/s; peak 5.35 GiB.
- **2026-10-25 23:47:** main queue `runs/logs/queue_main.sh` started:
  1. frozen r0
  2. FT LR pilot + FT r0
  3. scratch M5
  4. frozen r1–4
  5. FT r1
  6. FT r2
  Progress is in `runs/logs/queue_main.log`.
## 2026-10-26: overnight queue status (training chat)
- **Q1 + Q4 frozen sweep:** all 1,680 runs finished.
  - **BUG found 11:15:** `FrozenCtx._plan` cached evaluation batches by `id(list)`. CPython reuses the ids of freed lists, and val and test are both 870 studies, so some runs scored predictions against the wrong studies.
  - 89 runs show val or test macro < 0.56 (chance level). Fixed by keying on content (`tuple(sids)`).
  - Those 89 runs must be rerun, then `sweep_frozen.py --select-only`. Unflagged runs are correct, because a null-AUC split cannot exceed ~0.53.
  - Provisional curves re-selected from the clean runs only (419/420 cells): see the chat summary. **M3 ImageNet ConvNeXt-B is the best frozen backbone at every N.**
- **Q2 FT pilot:** done; LR 1e-4 for M1 and M2 (top of the grid).
- **Q3 scratch M5:** done. Test macro .644 / .668 / .737 at N = 260 / 1040 / 2600; gold .655 / .683 / .742.
- **Incidents:**
  1. The Claude Code low-memory reaper killed its background shells at ~04:00. `queue_main.sh` survived.
  2. `ft-M1-r0-s0-n0065` hung in final eval because its eval worker vanished. Killed; DataLoader timeout added.
  3. The resume failed on an RNG-state device error; fixed (.cpu()).
  4. sweep_ft now completes earlier rotations first.
- **2026-10-26 11:26–11:38 frozen rerun:**
  - FT was paused (sweep_ft suspended; the M2 n260 run was stopped after its step-188 checkpoint and later resumed from it).
  - 99 flagged runs (val or test < 0.58, listed in `results/frozen_rerun_2026-10-26.csv`) were rerun with the content-keyed plan cache: 0 failed. `frozen_selected.csv` was regenerated (420 rows).
  - Now 0 runs have val < 0.56. The 6 runs with test < 0.56 are all r4 s1 N=65 (M1, M2) and agree across 4 HPs (test .548–.553, gold .59). That is a genuinely poor 65-study draw, not the bug.
- **Frozen test macro, mean over 5 rotations × 3 seeds, N = 65 → 2600:**

  | Model | 65 | 130 | 260 | 520 | 1040 | 2080 | 2600 |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | M1 | .652 | .664 | .688 | .708 | .727 | .746 | .755 |
  | M2 | .637 | .652 | .681 | .695 | .711 | .723 | .729 |
  | M3 | .670 | .696 | .713 | .734 | .756 | .771 | .779 |
  | M4 | .649 | .675 | .693 | .713 | .725 | .737 | .745 |

  Gold-58 gives the same ranking: M3 > M1 > M4 > M2.
## 2026-10-31: FT sweep complete; T5.1 collect_results (analysis chat)
- **T4.7 complete:** queue finished 2026-10-31 03:41, 0 failures. All 42 main FT runs (M1/M2 × r0–2 × 7 N, seed 0, LR 1e-4) have metrics.
- **T5.1 `code/collect_results.py` → `results/runs.parquet`:**
  - 46,579 rows from 1,729 runs: 1,680 frozen, 42 FT main, 6 LR-pilot (val only), 3 scratch.
  - The 465 selected runs carry study-level bootstrap CIs (2,000 resamples, shared per split).
  - Every selected run's test/gold AUCs were recomputed from its saved predictions and match `metrics.json`. Runtime 5 min.
  - Added `common.boot_counts` / `common.auc_counts`: a vectorized bootstrap AUC. It matches `roc_auc_score` to 1e-16 on every resample and reproduces `boot_ci`'s CIs exactly, ~15–100× faster. Written by Claude (analysis chat).
- **Mean test macro, selected runs:**

  | Regime | Model | 65 | 130 | 260 | 520 | 1040 | 2080 | 2600 |
  |---|---|---:|---:|---:|---:|---:|---:|---:|
  | frozen | M1 | .652 | .664 | .688 | .708 | .727 | .746 | .755 |
  | frozen | M2 | .637 | .652 | .681 | .695 | .711 | .723 | .729 |
  | frozen | M3 | .670 | .696 | .713 | .734 | .756 | .771 | .779 |
  | frozen | M4 | .649 | .675 | .693 | .713 | .725 | .737 | .745 |
  | FT | M1 | .659 | .690 | .715 | .737 | .773 | .813 | .825 |
  | FT | M2 | .678 | .711 | .745 | .764 | .791 | .815 | .828 |
  | scratch | M5 | | | .644 | | .668 | | .737 |

  Frozen rows average 5 rotations × 3 seeds; FT rows average rotations 0–2, seed 0. Paired comparisons in T5.4 must use the matched cells.
- **Gold-58 macro at N=2600 (r0–2, s0):**
  - FT: M1 .840, M2 .837.
  - Frozen: M3 .793, M1 .772, M4 .758, M2 .736.
  - Scratch: M5 .742.
  - For reference, the training labels themselves score .866 against gold.
- **Competition context** written to `results/competition_context.md`: final LB, top solutions, comparability caveats, sources.
## 2026-10-31 (cont.): analysis complete (T1.3, T5.2-T5.9) and report workbook (analysis chat, Claude Code; all code AI-written)
- **New code:** `describe_data.py`, `bootcache.py`, `fit_curves.py`, `stats_tests.py`, `cost_model.py`, `viz.py`, `make_figures.py`, `build_workbook.py`. Outputs are in `results/tables/`, `results/fits/` and `figures/` (F1, F1b, F2-F8, PNG + PDF). The figure palette was checked with a colour-vision-deficiency validator.
- **H1** (FT vs frozen):
  - ViT-B: frozen is better at N=65 (Δ -0.019 [-0.032, -0.005]); no difference at 130-520; FT is better from N=1,040 (+0.027) and by +0.068 at 2,600.
  - ConvNeXt-B: FT is better from N=260 (+0.047) and by +0.086 at 2,600.
  - Mixed model: FT adds +0.011 AUC per doubling of N on top of frozen's +0.020.
- **H2** (ConvNeXt-B vs ViT-B, same DINOv3 pretraining):
  - FT: ConvNeXt-B is ahead at N=260-1,040 (+0.02-0.03); tied at 2,080-2,600.
  - Frozen: ViT-B is ahead (-0.025 at 2,600).
  - The model x regime interaction is +0.033 (p<0.001).
- **H3 opposite to prediction:** frozen ImageNet ConvNeXt-B beats DINOv3 ConvNeXt-B at every N (+0.033 to +0.050). It is the best frozen backbone in all 5 test folds (Kendall W 0.94).
- **C4** (DINOv3 vs DINOv2 ViT-B, frozen): within ±0.011.
- **H4:**
  - Focal findings average -0.054 AUC vs fluid/soft-tissue findings, and -0.040 at an equal number of training positives (cartilage/OA -0.025 → -0.021).
  - So only part of the focal gap is a prevalence effect.
- **H5 not supported:** log-linear wins by AIC (8/10); no plateau is identifiable.
- **Gain per doubling and exams to reach a target:**

  | Configuration | Gain per doubling | Exams to AUC 0.75 | Exams to AUC 0.80 |
  |---|---:|---:|---:|
  | Frozen backbones | +0.017 to +0.020 | 870 (ImageNet ConvNeXt-B) | ≥6,355†, extrapolated |
  | FT ViT-B | +0.031 | 671 | 1,652 |
  | FT ConvNeXt-B | +0.027 | 313 | 1,333 |

- **Gold-58:** rank agreement with the proxy test set is Spearman 0.97 (45 cells). Gold macro at N=2,600: FT ViT-B 0.840, FT ConvNeXt-B 0.837, frozen ImageNet ConvNeXt-B 0.793. Gold deltas match the test deltas.
- **Sensitivity:**
  - fallback_R exclusion changes AUC by ≤0.004;
  - restricting to findings with ≥10 positives changes no ranking.
- **Cost:**
  - expert labels ≈ $45k per 1,000 exams (range $9.6k-$250k);
  - local LLM labels ≈ $1 per 1,000;
  - compute ≤ $2 per model.
- **Report workbook:** `report/Final_Report_Workbook.docx`, 24 pages. Rebuild it with `code/build_workbook.py` after any rerun.

## 2026-11-01: report workbook revised for writing (Claude Code; all code AI-written, no report prose)
- **Team decisions:** Claude supports with outlines and draft review only (syllabus AI rule); references in APA 7; pipeline reuse confirmed with the professor/TA; no AI tools besides Claude and the local qwen3:14b labeller; both team members write all sections; 1st-place details verified by the team.
- **Page budget, measured in Word:** keeping all 7 figures + 7 tables in the main text left ~1,400 words. So Table 3 (AUC by N), Table 4 (paired deltas, merged into the Appendix C table), Figure 6 (per-finding gain), Figure 7 (cost vs AUC) and Table 6 (leaderboard) moved to appendices. Measured fit is now **≈2,230 words** (380 / 380 / 520 / 620 / 330); 2,350 spills onto page 11. Frozen AUC-0.85 rows were dropped from the cost table.
- **Corrections:**
  - 4.1 evidence said scratch ResNet-18 was "below every pretrained model" at N=2,600. It is not: on fold 0 it scores 0.737, above frozen DINOv3 ConvNeXt-B (0.727); on gold-58 it scores 0.742 vs 0.731.
  - The cost table's note now states that it uses the matched design (frozen ImageNet 555 vs 870, frozen ViT-B 1,404 vs 2,311 in the fits table) and that the exam counts exclude the ~870-study validation set.
- **Proposal deviation to disclose:** the proposal said every model, including ImageNet ConvNeXt-B, would be fine-tuned; ImageNet ConvNeXt-B was only run frozen (T4.8 not done).
- **New in the workbook:** a paragraph-plan "OUTLINE" box per section (fragments + numbers + word budgets + "decide before you write" questions); appendices re-lettered A-I; an APA 7 reference list (Appendix H; four entries tagged [CHECK]).
- **New script `code/make_submission.py`:** strips the guidance/outline boxes, front matter and section page breaks; lists leftover placeholders; exports `report/Final_Report.docx` + `.pdf` via Word; checks main-text pages against the 10-page limit. Tested on the empty workbook: 6 main-text pages of visuals.
