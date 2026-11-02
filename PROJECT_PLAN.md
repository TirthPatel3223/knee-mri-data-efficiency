# How Much Data Does a Hospital Need?
## Data Efficiency of CNNs vs. Pretrained Transformers for Knee-MRI Diagnosis
### Master project plan and task tracker (MGMTMSA 432 Healthcare Analytics, Fall 2026)

> **Status:** planning complete, implementation not started · **Created:** 2026-10-20 ·
> **Final report due:** Tue 2026-11-03 (≤10 pages, 11 pt, 1.5 spacing, plus appendices)
> **Proposal:** `T:\Group_Project\Group_Project_Proposal.docx` (submitted version)
> **This file is the single source of truth.** Any chat that changes a decision updates
> §15 (Decision log); any chat that finishes a task ticks it in §1 (Status board).

---

## 0. Read this first: ground rules for every chat working on this project

1. **The Kaggle folder is read-only.** Never modify or delete anything under
   `T:\Kaggle_Knee_Competition\`. The only allowed write there is *adding new* embedding
   files to `work\cache320\` via the existing `src\stage4_embed.py`, with a new `--tag`.
   All new code, manifests, runs, results and figures go under `T:\Group_Project\`.
2. **Never train on, tune on, or select with the test fold or the 58 gold studies.** Model,
   epoch and hyperparameter selection use the **validation fold only** (see §5.1). The test
   fold and gold-58 are scored once, after selection.
3. **No left/right-swapping augmentation.** Every volume is canonicalized lateral→medial,
   so a horizontal flip (coronal/axial) or slice-order reversal (sagittal) swaps medial and
   lateral and corrupts 4 of the 12 labels. Also no vertical flips. Allowed augmentations are
   listed in §5.7.
4. **Every run gets a unique, fully descriptive name and its own directory** (§8.3). Never
   overwrite an existing run directory unless `--force` is passed. (The earlier Kaggle work
   lost results once because output files were named after only part of the config.)
5. **Long jobs must be resumable and logged.** Run them in the background. Use one heavy GPU
   job at a time (12 GB VRAM).
6. **Use the system Python** (§3.2). If `import torch` fails inside the Claude Code sandbox
   with `WinError 4551 … Application Control policy has blocked … shm.dll`, rerun with the
   sandbox disabled or ask the user to run it. The code is not broken.
7. **When you finish a task:** tick it in §1, append the outcome and key numbers to
   `T:\Group_Project\RESULTS_LOG.md`, and record any changed decision in §15.
8. **AI-use disclosure is mandatory** for the course (§13). Note in `RESULTS_LOG.md` which
   chat or tool produced which code, so the report's disclosure is accurate.

---

## 1. Status board

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped (say why in §15).
Tier 1 = must have · Tier 2 = should have · Tier 3 = nice to have.

| ID | Task | Tier | Depends on | Owner | Status |
|---|---|---|---|---|---|
| T0.1 | Create project directory skeleton | 1 | — | | [x] |
| T0.2 | Environment check script | 1 | T0.1 | | [ ] |
| T0.3 | Download and verify DINOv3 ConvNeXt-B; record model facts | 1 | T0.2 | | [ ] |
| T0.4 | Shared `common.py` (paths, constants, loaders, metrics) | 1 | T0.1 | | [x] |
| T1.1 | Rotation split manifests | 1 | T0.4 | | [x] |
| T1.2 | Nested training-subset manifests | 1 | T1.1 | | [x] |
| T1.3 | Descriptive statistics for the report (Table 1, Fig. 1b) | 1 | T0.4 | analysis chat | [x] |
| T2.1 | Frozen embeddings: DINOv3 ViT-B/16 @224 | 1 | T0.2 | | [x] |
| T2.2 | Frozen embeddings: DINOv3 ConvNeXt-B @224 | 1 | T0.3 | | [x] |
| T2.3 | Verify existing ImageNet ConvNeXt-B and DINOv2 embeddings | 1 | — | | [x] |
| T2.4 | Frozen embeddings at 320 px (resolution ablation) | 3 | T2.1, T2.2 | | [ ] |
| T3.1 | `train_frozen_lc.py` (one head run per config) | 1 | T1.2, T0.4 | | [x] |
| T3.2 | Sanity check against the pilot numbers | 1 | T3.1, T2.3 | | [x] |
| T3.3 | Frozen sweep driver and full sweep | 1 | T3.2, T2.1, T2.2 | | [~] |
| T4.1 | Fine-tuning dataset/loader `ft_data.py` | 2 | T1.2 | | [x] |
| T4.2 | Fine-tuning model `ft_model.py` + embedding-equivalence test | 2 | T4.1 | | [x] |
| T4.3 | Fine-tuning trainer `train_ft.py` | 2 | T4.2 | | [x] |
| T4.4 | Throughput and memory benchmark; fix the FT config | 2 | T4.3 | | [x] |
| T4.5 | Fine-tuning smoke test | 2 | T4.4 | | [x] |
| T4.6 | Fine-tuning learning-rate pilot (validation only) | 2 | T4.5 | | [~] |
| T4.7 | Main fine-tuning sweep (rotations 0→1→2) | 2 | T4.6 | | [~] |
| T4.8 | Fine-tune the ImageNet ConvNeXt-B (rotation 0) | 3 | T4.7 (rot 0) | | [ ] |
| T4.9 | From-scratch CNN lower bound (rotation 0, 3 sizes) | 3 | T4.3 | | [~] |
| T4.10 | Single-site vs multi-site training (data diversity) | 3 | T3.3 | | [ ] |
| T5.1 | Collect all runs into `results/runs.parquet` | 1 | T3.3 | analysis chat | [x] |
| T5.2 | Main learning-curve table and figure | 1 | T5.1 | analysis chat | [x] |
| T5.3 | Learning-curve fitting and sample-size estimates | 1 | T5.1 | analysis chat | [x] |
| T5.4 | Statistical comparisons (paired bootstrap, mixed model) | 1 | T5.1 | analysis chat | [x] |
| T5.5 | Finding-group analysis and "positives needed" analysis | 1 | T5.3 | analysis chat | [x] |
| T5.6 | Gold-58 sanity check | 1 | T5.1 | analysis chat | [x] |
| T5.7 | Site-shift (rotation) variability analysis | 2 | T5.1 | analysis chat | [x] |
| T5.8 | Cost model and sensitivity analysis | 1 | T5.1 | analysis chat | [x] |
| T5.9 | Final figures in a consistent style | 1 | T5.2–T5.8 | analysis chat | [x] |
| T6.1 | Report outline and page budget | 1 | — | analysis chat | [x] |
| T6.2 | Write sections 1–2 (motivation, data) | 1 | T1.3 | | [ ] |
| T6.3 | Write sections 3–5 (methods, results, implications) | 1 | T5.* | | [ ] |
| T6.4 | Appendices, AI disclosure, references | 1 | T6.3 | | [ ] |
| T6.5 | Internal review and submission | 1 | T6.4 | | [ ] |

**Minimum viable report** = all Tier 1 tasks + T4.1–T4.7 on **rotation 0 only**.
That alone answers every research question with one test site-mix. Rotations 1–2 add
robustness.

---

## 2. Project summary

### 2.1 The healthcare problem
Knee MRI is among the most common musculoskeletal imaging exams. A single study holds 5–6
sequences (series) and ~150–190 images (median 162 slices per study in this dataset) that a
radiologist reads for many possible findings. AI that suggests likely abnormalities could
shorten reads and help prioritize worklists. Hospitals that consider building their own model,
for a new body part, sequence or site, face a cost question before an engineering one: **how
many exams must be collected and labeled, and does the choice of model change that number?**

### 2.2 Research questions
- **RQ1 (data scaling):** How does diagnostic performance (ROC-AUC) grow with the number of
  training exams *N*, from ~65 to ~2,600, for each model and training regime?
- **RQ2 (adaptation regime):** Does fine-tuning a pretrained backbone beat using it frozen
  (fixed feature extractor + trained head), and at what *N* does that change?
- **RQ3 (architecture vs. pretraining):** With pretraining held fixed, is a transformer
  (DINOv3 ViT-B/16) more data-efficient than a CNN (DINOv3 ConvNeXt-B)? And with architecture
  held fixed, does modern self-supervised pretraining (DINOv3, 1.7B images) beat standard
  ImageNet-22k supervised pretraining (ConvNeXt-B)?
- **RQ4 (heterogeneity across findings):** Does the data needed to plateau differ between
  large, diffuse findings (effusion, osteoarthritis) and small, focal ones (ligament and
  meniscal tears, fractures), after accounting for how many positive examples each has?
- **RQ5 (management implication):** What does reaching a target performance cost in exams,
  GPU-hours and labeling effort, and how should a hospital use that in a build-vs-buy decision?

### 2.3 Deliverables
1. Learning curves (macro and per finding) for 3–4 backbones × 2 regimes × 7 data sizes.
2. Fitted sample-size estimates (exams needed to get within 0.01/0.02 AUC of the plateau,
   and to reach target AUCs) with uncertainty.
3. A cost table with sensitivity analysis.
4. The final report (≤10 pages + appendices) with reproducible code under
   `T:\Group_Project\code\`.

---

## 3. Environment and infrastructure

### 3.1 Hardware (measured 2026-10-20)
| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU, **12,227 MiB** VRAM (Blackwell, sm_120) |
| CPU | Intel Core Ultra 9 285H, 16 logical cores |
| RAM | 31.4 GB |
| `T:` | NVMe SSD (WD_BLACK SN7100 1 TB), **~323 GB free**; holds the data and the 84 GB pixel cache |
| `C:` | NVMe SSD 1 TB, ~476 GB free |

Both drives are NVMe, so random slice reads from the memmap are fine; no repacking is needed.
The 84 GB pixel cache does not fit in 31 GB of RAM, so stream it from disk.

### 3.2 Python
- **Use:** `C:\Users\tirth\AppData\Local\Python\pythoncore-3.11-64\python.exe` (Python 3.11.9).
  It has torch `2.11.0.dev20260206+cu128` (the Blackwell build), timm 1.0.28,
  scikit-learn 1.5.2, scipy 1.17, statsmodels 0.14.6, polars 1.43.2, pandas 2.2.3,
  pyarrow, matplotlib 3.10.8, seaborn 0.13.2 and pydicom. All earlier pipeline stages ran on it.
- **Do not use** `T:\Kaggle_Knee_Competition\.venv` for this project. It has torch 2.13/cu130
  and timm but **no scikit-learn and no scipy**.
- `peft` (LoRA) is **not installed**. The plan does not need it (§5.4). If a chat wants LoRA
  anyway, `pip install peft` into the system Python and log it in §15.
- Hugging Face cache (`~/.cache/huggingface/hub`) already holds
  `timm/vit_base_patch16_dinov3.lvd1689m`, `timm/convnext_base.fb_in22k_ft_in1k` and
  `timm/vit_base_patch14_reg4_dinov2.lvd142m`. **`timm/convnext_base.dinov3_lvd1689m` must
  still be downloaded** (T0.3).

### 3.3 Windows-specific coding rules
- Guard all multiprocessing entry points with `if __name__ == "__main__":`.
- Open `np.memmap` lazily **inside each DataLoader worker**, never in `__init__`. Follow the
  `SliceSet` pattern in `T:\Kaggle_Knee_Competition\src\stage4_embed.py`.
- Use `persistent_workers=True`, `pin_memory=True` and 4–8 workers.
- Before overnight runs, keep the laptop on AC power and disable sleep (Windows power
  settings), or runs will die.
- Log GPU power and utilization during every timed run:
  `nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu,memory.used --format=csv -l 5 > runs/<run>/gpu.csv`
  The cost model uses this (T5.8).

### 3.4 Directory layout (created by T0.1)
```
T:\Group_Project\
  PROJECT_PLAN.md          <- this file
  RESULTS_LOG.md           <- append-only log of finished tasks and key numbers
  Group_Project_Proposal.docx
  code\
    common.py              <- paths, constants, loaders, metrics (T0.4)
    env_check.py           (T0.2)
    model_facts.py         (T0.3)
    make_splits.py         (T1.1)
    make_subsets.py        (T1.2)
    describe_data.py       (T1.3)
    train_frozen_lc.py     (T3.1)
    sweep_frozen.py        (T3.3)
    ft_data.py ft_model.py train_ft.py bench_ft.py sweep_ft.py   (T4.*)
    collect_results.py fit_curves.py stats_tests.py cost_model.py make_figures.py (T5.*)
  manifests\               <- rotations.json, subsets_r{r}_s{s}.json
  runs\                    <- one directory per run (§8.3)
  results\                 <- runs.parquet, tables\*.csv, fits\*.json
  figures\                 <- final PNG/PDF figures
  report\                  <- report drafts
```

---

## 4. Data and existing artifacts

### 4.1 Source dataset
RSNA Knee Abnormality Detection (Kaggle, 2026). Only the **data** is used; no public
notebooks or code. Raw data lives at `T:\Kaggle_Knee_Competition\rsna-knee-abnormality-detection\`
(~570 GB of DICOM).
- 4,407 training studies (**one study per patient**, so there is no patient leakage),
  24,371 series and 819,078 slices.
- ~20 sites on 5 continents. Vendors (by study): Siemens 1,951 · Philips 1,301 · GE 905 ·
  Canon/Toshiba 226 · Fuji/Hitachi 24. Field strength: 1.5 T 2,543 · 3.0 T 1,601 · other/unknown 263.
- Free-text radiology reports for every study, in 9+ languages (41% English).
- **12 binary findings:** ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
  PF OA, Effusion, Synovitis, Baker's (cyst), Contusion, Fracture.
- **Expert image-based labels exist for only 58 studies.** Two MSK radiologists labeled each,
  with a third adjudicating. These are the "gold" studies. The Kaggle test set has no public
  labels and is **not used**.

### 4.2 Existing preprocessing (reused as infrastructure; disclose in the report)
Built earlier by a team member, with AI assistance, in `T:\Kaggle_Knee_Competition\src\`.
Its results are documented in `FINDINGS.md`; the literature notes are in `RESEARCH.md`.

| Stage | Script | Output | What it did |
|---|---|---|---|
| 1 | `stage1_scan_dicom.py`, `stage1_laterality.py` | `work\slices_train.parquet`, `series_train.parquet`, `study_train.parquet` | header scan, geometry, laterality |
| 2 | `stage2_extract_labels.py` → `stage2_build_targets.py` | `work\labels_qwen3_14b.jsonl`, `work\targets.parquet` | local LLM (qwen3:14b) read every report; built soft targets, weights and scanner-grouped folds |
| 3 | `stage3_plan.py` → `stage3_build_cache.py` → `stage3_verify.py` | `work\cache320\pixels_train.u8` + index parquets | decode, resample, canonicalize and normalize all slices |
| 4 | `stage4_embed.py`, `stage4_train_head.py`, `stage4_compare.py`, `stage4_ensemble.py` | `work\cache320\emb_*`, `oof_*`, `goldpred_*` | frozen embeddings, attention head, paired bootstrap, ensembling |

### 4.3 Pixel cache (the input to every model)
- **File:** `T:\Kaggle_Knee_Competition\work\cache320\pixels_train.u8`, a raw `uint8` memmap of
  shape **(819078, 320, 320)**, 83.9 GB.
- **Geometry:** 0.5 mm/pixel in-plane, so 320 px is a 160 mm field of view (the dataset median).
  Recentered on the tissue centroid.
- **Orientation:** canonicalized so that **index 0 = LATERAL and the last index = MEDIAL**, for
  both knees. This is on the slice axis for sagittal series and the column axis for
  coronal/axial ones. It was verified by L/R average-volume correlation tests and signed-geometry
  checks (`FINDINGS.md`, Stage 3).
- **Intensity:** per-series percentile window (0.5–99.5) to 0–255. Normalization is per
  series, not per slice, so relative brightness (e.g. effusion) is preserved.
- Read the slices of one series as rows `[row0, row0 + n)`, which are contiguous and ordered
  along the slice normal. To feed a timm model: `float/255`, resize to the model input
  (bilinear, antialias), repeat to 3 channels, normalize with the model's
  `pretrained_cfg["mean"/"std"]`. This is exactly what `stage4_embed.SliceSet` does.

### 4.4 Index tables
`work\cache320\index_series_train.parquet`, 24,371 rows, one per series. Key columns:
`study_dir` (= StudyInstanceUID), `series_dir`, `n` (slices), `row0` (first row in the
memmap; rows cover 0…819,077 exactly once), `slot`, `plane`, `fluid`, `side` (L/R),
`side_src` (geometry/tag/study_consensus/**fallback_R**), plus geometry fields (`ps0, ps1,
rows, cols, thk, iop0..5, cx, scale_row, scale_col, fov_row, fov_col, flip_rows, flip_cols,
rev_slices, slice_axis, axis_align`) and `model` (scanner).

`work\cache320\index_slices_train.parquet`, 819,078 rows: `series_dir, file, row, t`.

**Slots** (Fluid_Sensitive and Fat_Suppression are perfectly collinear here, giving 6 slots):

| Slot | Meaning | Series |
|---|---|---|
| `Sag_0` | sagittal, non-fluid-sensitive | 5,197 |
| `Sag_1` | sagittal, fluid-sensitive fat-suppressed | 4,667 |
| `Cor_0` | coronal, non-FS | 3,985 |
| `Cor_1` | coronal, FS | 4,624 |
| `Axi_0` | axial, non-FS | 1,179 |
| `Axi_1` | axial, FS | 4,719 |

Studies are missing some slots (median 5 series per study), so every model must tolerate
absent slots. 359 series (~65 studies) have `side_src == "fallback_R"`, meaning their
medial/lateral orientation is a coin flip. Keep them, but add a sensitivity check (T5.4).

### 4.5 Target table `work\targets.parquet` (4,407 rows, 59 columns)
For each finding key `k` in `acl, mcl, mm, lm, moa, loa, pfoa, eff, syn, bak, con, frx`:
- `s_k` ∈ {`pos`, `neg`, `na`}: the report's stance as read by the LLM (`na` = the report
  does not address the finding).
- `t_k` ∈ [0.02, 0.98]: soft target, the LLM's probability that the **image** grade is
  positive under the host's rubric.
- `w_k`: loss weight, 1.0 if addressed, 0.5 if `na`.
- `g_k`: gold 0/1 for the 58 gold studies, null otherwise.

Plus `is_gold`, `vendor`, `model`, `coil`, `field`, `sex`, `n_series`, `n_slices`,
`group` (= vendor|model|coil scanner fingerprint, 80 groups) and `fold` (−1 = gold, 0–4 = scanner-grouped folds).

**Label quality:** the LLM-read report labels score **macro AUC 0.866 against the gold 58**
(`FINDINGS.md`, Stage 2). Per finding they range from 0.75 (synovitis) to 0.98 (MCL). Reports
over-call mild effusion and low-grade sprains and under-call synovitis. So the training and
evaluation labels are a **biased proxy**; §5.9 explains how this is handled.

**Report stance counts over the 4,349 non-gold studies:**

| Finding | pos | neg | na | | Finding | pos | neg | na |
|---|---:|---:|---:|---|---|---:|---:|---:|
| ACL | 508 | 3,494 | 347 | | PF OA | 715 | 3,384 | 250 |
| MCL | **123** | 3,855 | 371 | | Effusion | 920 | 3,301 | 128 |
| Medial Meniscus | 1,750 | 2,470 | 129 | | Synovitis | 452 | 1,003 | **2,894** |
| Lateral Meniscus | 672 | 3,402 | 275 | | Baker's | 544 | 2,632 | 1,173 |
| Medial OA | 862 | 3,233 | 254 | | Contusion | 944 | 2,298 | 1,107 |
| Lateral OA | 373 | 3,678 | 298 | | Fracture | **122** | 4,000 | 227 |

**Gold 58** positives: ACL 24, MCL 9, MM 26, LM 23, MOA 15, LOA 11, PFOA 21, Eff 35,
Syn 27, Baker's 12, Contusion 19, Fracture 18. All 58 have at least one positive finding
(abnormality-enriched). Vendors: Siemens 22, Philips 18, GE 16, Canon 2.

### 4.6 Folds: each fold is effectively a different hospital network
`fold` packs whole scanner groups into 5 equal folds (870/870/870/870/869). The vendor mix
per fold is **very different**:

| Fold | n | Siemens | Philips | GE | Canon/Toshiba | Fuji/Hitachi | 1.5 T | 3.0 T |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 870 | 6 | **697** | 114 | 45 | 8 | 352 | 509 |
| 1 | 870 | **549** | 67 | 244 | 10 | 0 | 290 | 551 |
| 2 | 870 | 521 | 199 | 134 | 0 | 16 | 598 | 255 |
| 3 | 870 | 526 | 109 | 235 | 0 | 0 | **869** | 1 |
| 4 | 869 | 327 | 211 | 162 | 169 | 0 | 408 | 259 |

Testing on a fold therefore measures generalization to **unseen scanners/sites**, which is the
realistic deployment question. But **any single test fold gives a site-specific answer.** That
is why the design rotates the test fold (§5.1) instead of fixing one.

### 4.7 Existing frozen embeddings (`work\cache320\emb_train_<tag>.f16` + `.json`)
Each is a float16 memmap of shape `(819078, dim)`, row-aligned with the pixel cache. The JSON
sidecar gives `model, size, dim, layout, mean, std`.

| Tag | Backbone | Input | dim | Layout | Use here |
|---|---|---|---:|---|---|
| `base_patch14_reg4_dinov2_224` | DINOv2 ViT-B/14 reg4 | 224 | 1536 | [CLS ; mean(patch)] | extra frozen baseline (M4) |
| `convnext_base_224` | ConvNeXt-B IN-22k→1k | 224 | 2048 | [GAP ; GMP] | **M3 frozen** |
| `convnext_base_320` | same | 320 | 2048 | [GAP ; GMP] | resolution ablation (T2.4) |
| `base_patch14_reg4_dinov2_322` | DINOv2 | 322 | 1536 | | not needed |
| `effv2s_300` | EfficientNetV2-S | 300 | | | not needed |

(`emb_test_*` files cover only the 3 Kaggle example test studies. Ignore them.)

### 4.8 Pilot results already available (from `FINDINGS.md`)
These came from a different protocol (5-fold OOF on all non-gold studies, epoch selection on
the evaluated fold), so they are **not** part of this study's results. They justify the design
and give sanity targets for T3.2:
- Frozen DINOv2-B @224: proxy macro 0.766, gold macro 0.803.
- Frozen ConvNeXt-B @224: proxy 0.791, gold 0.815. **The CNN beat the transformer on 9 of 12
  findings in the frozen regime.**
- More resolution helped the CNN (224→320: +0.004) but not the ViT.
- The image models beat the report text on large findings (effusion 0.945 gold) and lose on
  small focal ones (meniscus, ACL, fracture). This motivates RQ4.

---

## 5. Experimental design

### 5.1 Splits: 5 rotations of (test, validation, training pool)
For rotation `r ∈ {0,1,2,3,4}`:
- **test** = fold `r` (~870 studies; unseen scanners),
- **val** = fold `(r+1) mod 5` (~870; used only for early stopping and hyperparameter selection),
- **pool** = the other three folds (2,609–2,610 studies). Every training subset is drawn
  from here.
- The **gold 58** (fold −1) are never in any of these. They are scored by every final model.

A fixed-size validation set is used regardless of *N*. This is standard learning-curve
practice: it isolates the effect of training-set size. (A hospital would also need some
validation data. Note this in the limitations and optionally test it in T3.3-ablation, with
the validation set set to 20% of the subset.)

**Which rotations each regime uses:**
- Frozen: all 5 rotations (cheap).
- Fine-tuned: rotations 0, 1, 2 in that order (rotation 0 alone is the minimum viable result).
- All frozen-vs-fine-tuned comparisons are **paired on the same rotation, subset seed and N**.

### 5.2 Training-set sizes and subset sampling
**Sizes:** `N ∈ {65, 130, 260, 520, 1040, 2080, 2600}`, i.e. ≈2.5%, 5%, 10%, 20%, 40%, 80% and
~100% of the pool. The spacing is geometric (doubling), which is what learning-curve fitting
needs, and it contains the 5/10/20% points from the original idea.

**Sampling:** for each (rotation `r`, seed `s ∈ {0,1,2}`), randomly permute the pool with
`numpy.random.default_rng(1000*r + s)` and take prefixes. So the subsets are **nested**
(65 ⊂ 130 ⊂ … ⊂ 2600). That reduces curve noise and mimics a hospital accumulating more
data. Sampling is uniformly random, not stratified, so it represents a random slice of a
hospital archive. Record per-finding pos/neg/na counts of every subset (T1.2).

**Expected training positives per finding** (pool of rotation 0, uniform rates; ⚠ = fewer than 5):

| N | ACL | MCL | MM | LM | MOA | LOA | PFOA | Eff | Syn | Bak | Con | Frx |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65 | 7 | ⚠2 | 24 | 11 | 13 | 5 | 11 | 15 | 9 | 9 | 14 | ⚠2 |
| 130 | 14 | ⚠4 | 49 | 21 | 25 | 11 | 22 | 30 | 18 | 17 | 29 | ⚠4 |
| 260 | 29 | 7 | 98 | 42 | 50 | 22 | 44 | 60 | 37 | 34 | 58 | 8 |
| 520 | 57 | 14 | 196 | 85 | 100 | 44 | 89 | 120 | 74 | 69 | 115 | 16 |
| 1040 | 114 | 28 | 391 | 170 | 201 | 87 | 178 | 240 | 148 | 137 | 231 | 32 |
| 2080 | 229 | 56 | 782 | 339 | 401 | 175 | 356 | 480 | 295 | 275 | 462 | 64 |
| 2600 | 286 | 70 | 978 | 424 | 502 | 218 | 445 | 601 | 369 | 343 | 577 | 81 |

**Consequence:** MCL and Fracture have almost no positives at small N. They are also the
small, focal findings, so **finding size is confounded with prevalence**. T5.5 addresses this
by also plotting AUC against the number of training *positives*.

**Test-fold positives** (rotation 0, fold 0, addressed cells only): ACL 126/688, MCL **23**/788,
MM 406/462, LM 111/739, MOA 174/667, LOA 92/743, PFOA 124/713, Eff 174/690, Syn **32/134**,
Bak 98/453, Con 177/288, Frx **22**/803. Per-finding test AUCs for MCL, Fracture and Synovitis
will have wide CIs (roughly ±0.06). Averaging across rotations helps.

### 5.3 Models
| ID | Backbone (timm name) | Type | Params | Pretraining | Per-slice feature | Regimes |
|---|---|---|---:|---|---|---|
| **M1** | `vit_base_patch16_dinov3.lvd1689m` | Transformer | ~86M | DINOv3 SSL, LVD-1689M (1.7B images) | [CLS ; mean(patch)] = 1536 | frozen + FT |
| **M2** | `convnext_base.dinov3_lvd1689m` | CNN | ~88M | DINOv3 (distilled from the ViT-7B teacher), LVD-1689M | [GAP ; GMP] = 2048 | frozen + FT |
| **M3** | `convnext_base.fb_in22k_ft_in1k` | CNN | ~88M | supervised ImageNet-22k → 1k | [GAP ; GMP] = 2048 | frozen (+ FT in Tier 3) |
| M4 | `vit_base_patch14_reg4_dinov2.lvd142m` | Transformer | ~86M | DINOv2 SSL, LVD-142M | 1536 | frozen only |
| M5 | `convnext_tiny` or `resnet18`, random init | CNN | 28M / 11M | none | — | FT only, Tier 3 lower bound |

T0.3 must confirm the exact parameter counts and feature dims and write them to
`results/tables/models.csv`.

**Why these:**
- **M1 vs M2** keeps pretraining fixed and changes the architecture. This is the clean answer
  to "transformer vs CNN". Caveat: M2 is *distilled* from a ViT teacher, not trained
  independently. State that in the limitations.
- **M2 vs M3** keeps the architecture fixed and changes the pretraining (modern SSL at scale vs
  standard ImageNet). This is the "is a foundation model worth it" question.
- **M1 vs M4** asks whether DINOv3 improves on DINOv2 for MRI (secondary).
- **Input resolution 224** for all models: it matches the pretraining resolution, keeps
  fine-tuning affordable, and is identical across models, which a controlled comparison
  needs. 320 px is a Tier 3 frozen-only ablation (T2.4).

### 5.4 Training regimes
**Frozen.** The backbone is run once (`stage4_embed.py`) and every slice's feature vector is
cached. Only the study-level head (§5.5) is trained.

**Fine-tuned (FT).** The backbone and the head are trained end-to-end:
- **Full fine-tuning with layer-wise LR decay (LLRD):** decay 0.75 per ViT block; for ConvNeXt,
  0.75 per stage and the same across the blocks within a stage. Head LR = 10× the top backbone LR.
- **LP-FT warm start** (Kumar et al., 2022): the backbone is frozen for the first 10% of steps,
  so the randomly initialized head does not wreck the pretrained features.
- bf16 autocast, gradient checkpointing (`model.set_grad_checkpointing(True)`),
  `drop_path_rate=0.1`, AdamW (β=0.9/0.999, wd 0.05; no wd on norms, biases or embeddings),
  cosine schedule with 5% warmup, grad-clip 1.0.
- Fallback if memory or throughput fails (T4.4): fine-tune only the last 50% of blocks/stages.
  Apply it **identically** to M1 and M2 and log it in §15.

### 5.5 Study-level head (shared by all models and both regimes)
Reuse `Head` from `T:\Kaggle_Knee_Competition\src\stage4_train_head.py` unchanged (import it;
do not copy-edit it):
- per-slice features → LayerNorm → Linear(dim → d=256),
- \+ learned **slot embedding** (6 slots) + **position-in-series** encoding (sin/cos, 6
  frequencies of the normalized slice position 0–1, where 0 = lateral on sagittal),
- dropout, optional transformer layers (`layers=0` by default),
- **12 learned queries**, one per finding, attending over all slices of the study (masked
  softmax) → 12 pooled vectors → 12 logits.
- One head design for everything means differences come from the **backbone and regime only**.
- Bags (sets of slices per study) come from `build_bags` in the same file: all slices of all
  series, evenly thinned to `cap=384` if longer.

### 5.6 Objective
Weighted soft-label binary cross-entropy, exactly as in `stage4_train_head.py`:
`loss = Σ_k w_k · BCEWithLogits(logit_k, t_k) / Σ_k w_k`, averaged over the batch.
Targets are `t_k`, weights `w_k` (0.5 for `na`).

### 5.7 Inputs and augmentation
- **Frozen:** no augmentation (features are precomputed).
- **FT training:** for each series, sample **8 slices** by stratified random sampling (split the
  series into 8 equal bins, take one random slice per bin; if n < 8 take all). That gives
  ≤48 slices per study. Each slice keeps its true normalized position.
- **FT augmentation**, applied identically to all sampled slices of one series:
  - random affine: rotation ±10°, scale 0.9–1.1, translation ±5%,
  - intensity: brightness ±10%, contrast ±10%, gamma 0.9–1.1.
  - **Forbidden:** horizontal flip, vertical flip, slice-order reversal, any mirroring.
- **FT validation** (for selection): 8 evenly spaced slices per series, no augmentation.
- **Final test and gold evaluation, all models and regimes:** all slices, evenly thinned to a
  384-slice cap (the same bags the frozen head sees). That keeps frozen-vs-FT comparisons
  fair at test time.

### 5.8 Hyperparameter and checkpoint selection (validation fold only)
**Frozen head.** For every (model, rotation, seed, N), run a small grid and keep the best by
**val macro proxy AUC**:
- `lr ∈ {3e-4, 1e-3}` × `drop ∈ {0.2, 0.5}`, with `wd = 0.01`, `d = 256`, `layers = 0`.
- Batch size: `min(24, max(8, N // 8))`.
- Epochs: `max(30, ceil(400 · batch / N))`, so at least ~400 optimizer steps at small N.
- OneCycle schedule. Evaluate val every epoch; keep the best epoch.
- Feature standardization (mu/sd) is computed on the **pool slices of that rotation only**,
  not on all 819k slices as the old script did (a minor leakage fix).

**Fine-tuning.**
- One LR pilot (T4.6) on rotation 0, seed 0, N = 520, val only:
  top backbone LR `∈ {1e-5, 3e-5, 1e-4}` for each of M1 and M2. Freeze the winning recipe
  per model and record it in §15. The **same tuning budget for both architectures** keeps it fair.
- **Training budget per run:** `V(N) = clip(10·N, 6000, 26000)` study-views (one view = one
  study with its sampled slices). Effective batch is 16 studies (micro-batch 1–2 studies ×
  gradient accumulation).
- Validate 10 times per run, evenly spaced. Keep the best val-macro checkpoint and run the
  final test/gold evaluation from it.

### 5.9 Evaluation: two rulers
1. **Primary: proxy AUC on the test fold.** For each finding, the label is `s_k == "pos"`
   over studies with `s_k ∈ {pos, neg}` (exclude `na`). Report ROC-AUC per finding and the
   **macro average of the 12**. This is the same definition `stage4_train_head.macro_auc` and
   `stage4_compare.py` use. It has plenty of data (~870 studies), but the labels are
   report-derived. Bias largely cancels when **comparing models on the same labels**, which
   is what every research question asks.
2. **Secondary: gold-58 AUC.** These are expert image labels, the right target, but very
   coarse (a single finding's 95% CI is ~±0.12; macro ~±0.03). Use them only as a sanity
   check that rankings agree (T5.6). **Never select on gold.**
- Context line for the report (different ruler, do not plot on the same axis as proxy): the
  LLM reading the report text scores gold macro 0.866. That is the quality ceiling of the
  training labels.
- **Uncertainty within a run:** study-level bootstrap (2,000 resamples) of test AUC, reusing
  `boot_ci` / `boot_delta` from `stage4_compare.py`.
- **Uncertainty across runs:** mean ± SE over (rotation × seed) runs.

### 5.10 Pre-registered hypotheses and predictions
Write these into §15 with a timestamp **before** looking at main results (T3.3/T4.7 outputs).

| ID | Hypothesis | Prediction (direction / size) | Test |
|---|---|---|---|
| H1 | FT − frozen gap grows with N; frozen ≥ FT at the smallest N | crossover N* between 130 and 1,040 | paired Δ(N) with CIs; mixed model (T5.4) |
| H2 | At matched DINOv3 pretraining, ViT-B and ConvNeXt-B differ in data efficiency | two-sided. Pilot (frozen) favored CNN; literature suggests parity after FT | paired Δ(N), M1 vs M2, per regime |
| H3 | DINOv3 pretraining beats ImageNet pretraining most at small N | M2 − M3 > 0, larger at N ≤ 260 | paired Δ(N), frozen |
| H4 | Diffuse/OA findings plateau at smaller N than focal findings | N₀.₀₂(focal) > N₀.₀₂(diffuse), also after controlling for training positives | fitted N_ε by finding group; positives-based curves (T5.5) |
| H5 | Learning curves follow an inverse power law in this range | power law fits better than log-linear by AIC for most models | T5.3 |

**Pre-registered finding groups** (do not change after seeing results):
- **Focal structural:** ACL, MCL, Medial Meniscus, Lateral Meniscus, Fracture.
- **Cartilage / OA:** Medial OA, Lateral OA, PF OA.
- **Fluid / marrow / soft tissue:** Effusion, Synovitis, Baker's, Contusion.

**Primary endpoint:** macro proxy test AUC, averaged over rotations and seeds, as a function of N.
**Primary comparisons:** C1 = FT vs frozen (M1, M2); C2 = M1 vs M2 within each regime;
C3 = M2 vs M3 (frozen). Secondary: C4 = M1 vs M4 (frozen); per-finding versions of C1–C3.

### 5.11 Learning-curve model
- Fit the inverse power law **AUC(N) = a − b·N^(−c)** with `scipy.optimize.curve_fit`,
  bounds `a ∈ [0.5, 1.0]`, `b > 0`, `c ∈ (0, 2]`, on run-level points (every rotation × seed),
  for each (model, regime) and for each (model, regime, finding).
- **Derived quantities:**
  - asymptote `a`,
  - **N_ε = (b/ε)^(1/c)**, the exams needed to get within ε of the asymptote, for ε = 0.01 and 0.02,
  - **N_τ = (b/(a−τ))^(1/c)**, the exams needed to reach a target AUC τ (τ ∈ {0.75, 0.80, 0.85}; only if τ < a),
  - **data-efficiency ratio** N_τ(model A) / N_τ(model B).
- **Uncertainty:** cluster bootstrap over runs (resample rotation×seed units with replacement,
  refit; 1,000 resamples). Report medians and 95% intervals.
- **Model comparison:** also fit log-linear (`AUC = α + β·log N`) and exponential saturation;
  compare by AIC (H5).
- **Extrapolation flag:** any N_ε or N_τ above 2,600 is an extrapolation. Mark it with "†" in
  tables and do not overclaim (Viering & Loog, 2022).

### 5.12 Cost model (RQ5)
- **Compute:** measured GPU wall-hours per run (from `timing.json`), and energy from logged
  `power.draw` (kWh). Report a $ figure at an assumed cloud price for a comparable GPU, and
  label it as an assumption with a citation.
- **Labeling:**
  - LLM report reading was **measured**: 8.4 h for 4,407 reports on this GPU (~6.9 s/report;
    `FINDINGS.md` Stage 2).
  - Manual radiologist labeling is an **assumption** with a citation: minutes per exam for 12
    findings × radiologist hourly cost (e.g. BLS/MGMA). Run a sensitivity analysis over
    5–20 min/exam and a range of hourly rates.
- **Output:** for each model and regime, the cost to reach τ = 0.80 / 0.85 and to reach the
  plateau − 0.02, in exams, GPU-h, $ compute, and $ labeling (manual vs LLM). This is the
  management deliverable.

---

## 6. Run matrix and compute budget

| Block | Configs | Est. time per unit | Est. total GPU-h |
|---|---|---|---:|
| T2.1–T2.2 frozen embeddings (2 new backbones @224) | 2 passes × 819k slices | ~25 min (measured 574 sl/s for ViT-B) | ~1 |
| T3.3 frozen sweep | 4 models × 5 rot × 3 seeds × 7 N × 4 HP = **1,680 head runs** | ~5–20 s | ~4–6 |
| T4.4–T4.5 bench + smoke | — | — | ~1 |
| T4.6 FT LR pilot | 2 models × 3 LR at N = 520 | ~0.7 h | ~4 |
| **T4.7 main FT sweep** | 2 models × 3 rotations × 1 seed × 7 N = **42 runs** | per (model, rotation): ~78.8k study-views × ~44 slices ≈ 3.5M slices ÷ ~190 sl/s ≈ 5 h + ~1 h eval | **~36** |
| T4.8 FT of M3 (rotation 0) | 7 runs | | ~6 |
| T4.9 from-scratch (rotation 0, N ∈ {260, 1040, 2600}) | 3 runs | | ~4 |
| T2.4 320 px embeddings | 3 passes | ~45 min | ~2.5 |
| **Total** | | | **~55–60 GPU-h** of ~300 wall-hours available |

⚠ The **~190 slices/s fine-tuning throughput is an estimate** (≈ ⅓ of the measured
inference speed). T4.4 measures it. **Decision rule:** below 120 sl/s → use K = 6
slices/series and run FT on rotations {0, 1} only. Below 80 sl/s → also drop N = 2080 and
use the partial-FT fallback.

**Execution order for the GPU queue:**
T2.1 → T2.2 → T3.3 (frozen sweep; short head runs) → T4.4 → T4.5 → T4.6 (overnight) →
T4.7 rotation 0 (both models, all N, small N first) → rotation 1 → rotation 2 → Tier 3.
CPU-only analysis (T5.*) proceeds in parallel as results land.

---

## 7. Detailed task specifications

Each task lists **Goal / Inputs / Outputs / Steps / Acceptance criteria**. File paths are
relative to `T:\Group_Project\` unless absolute.

### Phase 0: Setup

#### T0.1 Directory skeleton
- **Outputs:** the folders of §3.4; an empty `RESULTS_LOG.md` with a header.
- **Accept:** folders exist; nothing written under `T:\Kaggle_Knee_Competition`.

#### T0.2 `code/env_check.py`
- **Steps:** print the Python path and version; torch, timm, sklearn, scipy, polars,
  statsmodels versions; `torch.cuda.is_available()`, the device name and total VRAM; run a bf16
  matmul on the GPU. Open `pixels_train.u8` as a memmap with shape `(819078,320,320)` and read
  rows 0 and 819077. Load `targets.parquet` and assert 4,407 rows and a fold distribution of
  {−1:58, 0:870, 1:870, 2:870, 3:870, 4:869}. Load `index_series_train.parquet` and assert
  `sum(n) == 819078`.
- **Accept:** exits 0 and prints everything. Paste the output into `RESULTS_LOG.md`.

#### T0.3 `code/model_facts.py`: download DINOv3 ConvNeXt-B and record model facts
- **Steps:** `timm.create_model(name, pretrained=True, num_classes=0)` for M1–M4. This
  downloads `convnext_base.dinov3_lvd1689m`. If the Hugging Face download is gated or fails,
  the user may need to accept the DINOv3 license on HF and `hf auth login`. For each model,
  record the parameter count, `pretrained_cfg` mean/std/input size, `num_prefix_tokens`
  (DINOv3 ViT has CLS + 4 register tokens), the `forward_features` output shape at 224, and
  the pooled feature dim using the §5.3 layout.
- **Outputs:** `results/tables/models.csv`.
- **Accept:** 4 rows; M2 is downloaded; dims are 1536/2048/2048/1536.

#### T0.4 `code/common.py`
- **Contents:**
  - Paths: `KAGGLE = T:\Kaggle_Knee_Competition`, `WORK = KAGGLE\work`, `CACHE = WORK\cache320`,
    `GP = T:\Group_Project`.
  - `LABELS`, `KEYS`, `SLOTS` (copied exactly from `stage4_train_head.py`).
  - `FINDING_GROUPS` (§5.10), `SIZES = [65,130,260,520,1040,2080,2600]`, `SEEDS = [0,1,2]`,
    `ROTATIONS = [0..4]`.
  - `MODELS` dict mapping model id → timm name, embedding tag, dim and layout.
  - Loaders: `load_targets()`, `load_series_index()`, `load_rotation(r)`,
    `load_subset(r, s, N)`.
  - `y_proxy(tg, k)` → (labels, mask) per §5.9; `y_gold`.
  - `macro_auc` (re-export from stage4).
  - Bootstrap helpers (re-export `boot_ci`, `boot_delta` from `stage4_compare.py`).
- **Reuse rule:** import from `T:\Kaggle_Knee_Competition\src` via `sys.path.insert`.
  Do not duplicate `Head`, `build_bags`, `collate`, `predict`, `macro_auc`, `boot_ci`,
  `boot_delta`.
- **Accept:** `python -c "import common"` works; a unit check reproduces fold sizes and
  label counts from §4.5.

### Phase 1: Data manifests

#### T1.1 `code/make_splits.py` → `manifests/rotations.json`
- For r = 0..4: `{"test": [study_dir...], "val": [...], "pool": [...]}` per §5.1.
- **Asserts:** the three sets are disjoint; their union is all 4,349 non-gold studies; no gold
  study appears; pool size is 2,609–2,610.
- Also write `results/tables/rotation_summary.csv`: per rotation and split, n, vendor counts,
  field-strength counts, per-finding pos/neg/na.
- **Accept:** the asserts pass; the CSV matches §4.6.

#### T1.2 `code/make_subsets.py` → `manifests/subsets_r{r}_s{s}.json`
- Per §5.2: `rng = np.random.default_rng(1000*r + s)`, permute the pool, `{N: perm[:N]}` for
  every N (N = 2600 is capped at the pool size).
- Also write `results/tables/subset_label_counts.csv` (r, s, N, finding, pos, neg, na).
- **Accept:** subsets are nested; reruns are bit-identical; every subset ⊂ pool.

#### T1.3 `code/describe_data.py`: descriptive statistics for report §2
- **Table 1:** studies, series, slices, slices per study (median/IQR), vendors, field strength,
  slot availability, report language mix (see `RESEARCH.md` §1.4), report-label prevalence vs
  gold prevalence per finding.
- **Fig 1b:** label prevalence bar chart (report-derived vs gold).
- **Accept:** CSV + PNG in `results/tables/` and `figures/`.

### Phase 2: Frozen features

#### T2.1 DINOv3 ViT-B/16 embeddings
```
<python> T:\Kaggle_Knee_Competition\src\stage4_embed.py --split train --px 320 ^
  --model vit_base_patch16_dinov3.lvd1689m --size 224 --tag dinov3_vitb16_224 --batch 128 --workers 6
```
- **Accept:**
  - `emb_train_dinov3_vitb16_224.f16` + `.json` exist, with dim 1536 and done = 819078;
  - the final sanity line shows `nonfinite 0` and `all-zero rows 0`;
  - throughput is logged to `RESULTS_LOG.md` (needed for the cost table).
- The script is resumable. If it dies, rerun the same command.

#### T2.2 DINOv3 ConvNeXt-B embeddings
Same command with `--model convnext_base.dinov3_lvd1689m --tag dinov3_convnextb_224`.
**Accept:** dim 2048 with layout [GAP ; GMP]; the same sanity checks.

#### T2.3 Verify existing embeddings
Check that `convnext_base_224` and `base_patch14_reg4_dinov2_224` exist with done = 819078.
Nothing to compute.

#### T2.4 (Tier 3) 320 px embeddings
M1/M2 at `--size 320`; M3 already exists as `convnext_base_320`. Used for one frozen
resolution ablation at all N on rotation 0.

### Phase 3: Frozen learning curves

#### T3.1 `code/train_frozen_lc.py`
- **CLI:** `--model {M1,M2,M3,M4} --rotation r --seed s --n N --lr --drop [--force]`.
- **Steps:**
  1. Load the embedding matrix to the GPU (as `stage4_train_head.py` does).
  2. Compute mu/sd from pool slices only.
  3. Build bags with `build_bags(cap=384)` for train subset ∪ val ∪ test ∪ gold.
  4. Train `Head` (d=256, layers=0) with the §5.6 loss and the §5.8 schedule.
  5. Evaluate val macro proxy every epoch and keep the best state.
  6. Score test and gold **once**, at the end.
- **Outputs** (run dir, §8.3): `config.json`, `metrics.json`, `preds_test.parquet`,
  `preds_val.parquet`, `preds_gold.parquet` (study_dir + 12 probabilities), `train_log.csv`
  (epoch, loss, val_macro), `timing.json`.
- **Key differences from `stage4_train_head.py`:**
  - selection on the **val fold, not the test fold**;
  - a training subset instead of 4 folds;
  - a step floor at small N;
  - pool-only standardization;
  - full-config run names.
- **Accept:** a run at N = 2600 finishes in under 1 minute and writes all files; rerunning the
  same config refuses to overwrite.

#### T3.2 Sanity check vs the pilot
M3 (convnext_base_224), rotation 0, seed 0, N = 2600, plus the same for M4.
- **Accept:** test macro proxy is within ±0.03 of the pilot (0.791 / 0.766), and M3 > M4 holds
  (the pilot direction). If not, debug before sweeping.

#### T3.3 `code/sweep_frozen.py` + full sweep
- Iterate all 1,680 configs (§6), skip finished run dirs (resumable), run them sequentially
  in one process with the embeddings kept resident on the GPU, and log progress.
- After the sweep: for each (model, r, s, N), write the **selected** HP (best val macro) to
  `results/frozen_selected.csv`. Only selected runs enter the analysis.
- **Optional ablation:** rotation 0, validation = 20% of the subset instead of the fixed fold.
- **Accept:** 1,680 metrics files; no failed runs; the selected table has 4 × 5 × 3 × 7 = 420 rows.

### Phase 4: Fine-tuning

#### T4.1 `code/ft_data.py`
- `StudyDataset(study_ids, mode={"train","val","test"}, model_cfg, seed)`.
- Per study, look up its series in `index_series_train.parquet` and sample slices per §5.7.
  Read rows from the memmap (opened lazily per worker), apply §5.7 augmentations (train only;
  same affine per series), resize 320→224 (bilinear, antialias), repeat to 3 channels and
  normalize with the model's mean/std.
- **Returns:** `pixels (L,3,224,224)`, `slot (L,)`, `pos (L,)`, `t (12,)`, `w (12,)`, `study_id`.
- Collate pads to the max L in the batch, with a mask.
- **Accept:** a unit test checks that
  - no flips happen (an augmented slice with the affine disabled equals the input),
  - positions are in [0, 1],
  - test mode reproduces the frozen bag composition (same rows as `build_bags(cap=384)`).

#### T4.2 `code/ft_model.py`
- `SliceEncoder`: the timm backbone (`num_classes=0`, `drop_path_rate=0.1`, grad checkpointing)
  + the §5.3 pooling (ViT: `[CLS ; mean(patch tokens after num_prefix_tokens)]`;
  CNN: `[GAP ; GMP]` of `forward_features`).
- `StudyModel = SliceEncoder → Head`.
- Parameter-group builder for LLRD (§5.4).
- **Equivalence test (important):** in eval mode, with no augmentation, the per-slice features
  for 16 random rows must match the cached frozen embeddings for M1 and M2 within float16
  tolerance (cosine similarity > 0.999). This proves the FT and frozen pipelines see identical
  inputs.
- **Accept:** the equivalence test passes for both models.

#### T4.3 `code/train_ft.py`
- **CLI:** `--model --rotation --seed --n --lr [--partial] [--force]`.
- **Implements:** §5.4 optimizer/schedule/LP-FT; the §5.8 budget `V(N)`; effective batch 16
  via accumulation; 10 val evaluations; best-checkpoint save; final test + gold evaluation with
  all-slice bags.
- **Logs:** `train_log.csv` (step, loss, lr, val_macro), `gpu.csv` via nvidia-smi, `timing.json`
  (wall-s, GPU-h, mean W, kWh).
- **Resumable:** checkpoint every ~30 min (model, optimizer, scheduler, RNG states, step).
- **Outputs:** same files as frozen runs, plus `best.pt` (keep; ~350 MB each).
- **Accept:** killing and restarting mid-run resumes at the same step with the same loss curve.

#### T4.4 `code/bench_ft.py`: throughput and memory
- For M1 and M2: micro-batch ∈ {1, 2} studies, K ∈ {6, 8} slices/series, checkpointing on/off.
  Run 50 steps each; record train slices/s and `torch.cuda.max_memory_allocated`.
- **Outputs:** `results/tables/throughput.csv`. Choose the fastest config that fits, using the
  **same K for both models**, and record it in §15. Recompute the §6 budget with measured
  numbers and apply the decision rule.
- **Accept:** a chosen config with no OOM, for both models.

#### T4.5 Smoke test
M1 and M2, rotation 0, seed 0, N = 130, budget overridden to 1,000 views.
- **Accept:** loss decreases; val macro > 0.60; all output files written; resume works.

#### T4.6 LR pilot
6 runs (§5.8), rotation 0 / seed 0 / N = 520, scored on **val only**. Pick the best LR per
model and record it in §15. Do not look at test numbers when choosing.

#### T4.7 Main FT sweep: `code/sweep_ft.py`
- Order: rotation 0 → 1 → 2. Within a rotation, alternate M1/M2 at each N, from smallest N up,
  so a partial sweep still yields matched pairs. 42 runs in total. Skip finished runs.
- **Accept:** every run has metrics; failures are logged and retried once.

#### T4.8–T4.10 (Tier 3)
- **T4.8:** fine-tune M3 (rotation 0, all N) to complete the 2×2 for C3 in the FT regime.
- **T4.9:** a from-scratch ConvNeXt-T (or ResNet-18) at N ∈ {260, 1040, 2600}, rotation 0,
  with 3× the budget. This is the lower bound that quantifies the value of pretraining.
- **T4.10:** data diversity (frozen, cheap). At fixed N ∈ {260, 520}, train on studies from
  **one scanner group** (single hospital) vs a random multi-site subset, and test on fold r.
  Question: are 500 exams from one site worth 500 from twenty?

### Phase 5: Analysis (CPU; can start as soon as frozen results exist)

#### T5.1 `code/collect_results.py` → `results/runs.parquet`
Long format, one row per (run × split × finding), plus macro rows.

Columns: `run, regime, model, rotation, seed, n_train, hp_id, selected (bool), split
(val/test/gold), finding (or "macro"), auc, auc_lo, auc_hi, n_pos, n_neg, n_train_pos,
gpu_h, kwh`.

#### T5.2 Main curve: **Fig. 2** + **Table 3**
- **Fig. 2:** macro test AUC vs N (log₂ axis) for every (model, regime). Points show the mean
  across runs; the band is ±1 SE. Use the selected frozen runs.
- **Table 3:** mean ± SE at each N.

#### T5.3 `code/fit_curves.py` (§5.11)
- **Outputs:**
  - `results/fits/*.json`;
  - **Table 4:** `a`, `c`, N₀.₀₁, N₀.₀₂, N_τ, with 95% intervals and † for extrapolations;
  - **Fig. 5:** N₀.₀₂ per finding, grouped by §5.10 finding group, dot plot with intervals;
  - AIC comparison (H5).

#### T5.4 `code/stats_tests.py`
- **Paired comparisons** C1–C4 at each N. Per run pair, take the study-level paired bootstrap
  Δ on the same test fold (`boot_delta`). Then combine across runs with the mean Δ and a
  cluster bootstrap CI.
- **Fig. 4:** FT − frozen Δ vs N with CI and the crossover N* marked (H1).
- **Mixed model** (statsmodels `MixedLM`):
  `auc ~ C(model) * C(regime) * log2(n_train)` with a random intercept for `rotation`
  (seed nested). Report the interaction terms.
- Holm correction across the 12 per-finding tests.
- **Sensitivity checks:** (a) exclude the ~65 `fallback_R` studies from test scoring and
  confirm medial/lateral results are unchanged; (b) macro over only the findings with ≥10
  training positives at that N.

#### T5.5 Finding heterogeneity (H4)
- Compare N₀.₀₂ across the three pre-registered groups.
- **Fig. 6:** per-finding AUC vs **number of training positives** (log axis). If the focal
  findings line up with the others once plotted by positives, then the "focal needs more
  data" effect is really a prevalence effect. Report which explanation the data supports.

#### T5.6 Gold-58 check
For the selected models at N = 2600, report gold macro + per-finding AUC with CIs, and the
Spearman correlation between proxy and gold model rankings. The appendix table states plainly
that gold resolves only ~0.03 macro.

#### T5.7 Site shift
Per-rotation curves (small multiples; **Fig. 8**). Within each test fold, per-vendor AUC where
n ≥ 100. Ask whether the model ranking is stable across site-mixes.

#### T5.8 `code/cost_model.py` (§5.12)
- **Fig. 7:** GPU-h vs test AUC, a Pareto plot.
- **Table 5:** cost to reach the targets, with sensitivity ranges.
- Every assumption is listed with a citation (a sub-task for the writer: find sources for
  radiologist read and label time, radiologist wage, and cloud GPU price).

#### T5.9 `code/make_figures.py`
One script regenerates all figures from `results/` with one style: colorblind-safe palette,
consistent colors per model, solid = FT, dashed = frozen, 300 dpi PNG + PDF.

**Figure list:**
- F1 pipeline schematic (draw by hand or in a diagram tool);
- F1b label prevalence;
- F2 main curves;
- F3 12-panel per-finding curves;
- F4 FT − frozen gap;
- F5 N₀.₀₂ by finding;
- F6 AUC vs training positives;
- F7 cost Pareto;
- F8 per-rotation.

### Phase 6: Report

#### T6.1 Outline and page budget (10 pages, 11 pt, 1.5 spacing), mapped to the syllabus §3
| Report section | Syllabus requirement | Pages | Figures/tables |
|---|---|---:|---|
| 1. Motivation and question | "Motivate the question" | 1.25 | — |
| 2. Data | "Describe the data", incl. limitations | 1.5 | T1, F1b |
| 3. Methods | "Describe the analysis" | 2.5 | F1, T2 |
| 4. Results | "Present and interpret your findings" | 3.25 | F2, F4, F5, F6, T3, T4 |
| 5. Implications for hospitals and limitations | "implications for the healthcare problem" | 1.5 | F7, T5 |
| Appendices (unlimited) | — | — | F3, F8, HP tables, gold-58, per-finding tables, AI disclosure, code reuse statement |

#### T6.2–T6.5 Writing, appendices, review, submission
- **Writing checklist:** every number in the text traces to a file in `results/`; every figure
  has a one-sentence takeaway caption; limitations are stated:
  - proxy labels,
  - fixed validation set,
  - 224 px input,
  - M2 distilled from a ViT teacher,
  - single dataset,
  - laptop-GPU compute.
- **Submission:** PDF to Bruin Learn by Tue 2026-11-03.

---

## 8. Code conventions (all chats)

### 8.1 Style
Python 3.11, type hints where cheap, argparse CLIs, polars for tables, numpy/torch for arrays.
Module docstrings explain *why*, matching the style of the existing `src/` files.

### 8.2 Determinism
Seed numpy, torch and Python `random` from the run's (rotation, seed). Set
`torch.backends.cudnn.benchmark=True` for speed; exact bitwise reproducibility of FT is not
required, but subset membership is.

### 8.3 Run naming and directory layout
`runs/{regime}-{model}-r{rot}-s{seed}-n{N:04d}-{hpid}/`

Examples: `runs/frozen-M2-r0-s1-n0260-lr3e-4_dr0.5/`, `runs/ft-M1-r1-s0-n2600-lr3e-5/`.

`config.json` holds **every** argument plus the git-free code hash (sha1 of the script files)
and timm/torch versions.

### 8.4 `metrics.json` schema
```json
{"run": "...", "regime": "frozen|ft", "model": "M1", "rotation": 0, "seed": 0, "n_train": 260,
 "hp": {"lr": 3e-4, "drop": 0.2}, "best_epoch_or_step": 41,
 "val_macro": 0.781, "test_macro": 0.774, "gold_macro": 0.80,
 "test_per_finding": {"ACL": 0.85, "...": 0.0}, "gold_per_finding": {"ACL": 0.0},
 "test_n_pos": {"ACL": 126}, "train_n_pos": {"ACL": 29},
 "wall_s": 18.2, "gpu_h": 0.005, "kwh": null}
```

### 8.5 Logging
Every task appends to `RESULTS_LOG.md`: date, task ID, what was run (command), key numbers,
issues, and which chat/tool wrote the code.

---

## 9. Timeline (today = Tue 2026-10-20; due Tue 2026-11-03)

| Date | GPU queue (background) | People (foreground) |
|---|---|---|
| Wed 10-21 | T2.1, T2.2 embeddings | T0.1–T0.4, T1.1–T1.2; start T4.1–T4.2 |
| Thu 10-22 | T3.2 sanity → T3.3 frozen sweep | T3.1 done; T4.3; T1.3 descriptives; T6.1 outline |
| Fri 10-23 | T4.4 bench, T4.5 smoke, **T4.6 LR pilot (overnight)** | T5.1–T5.3 on frozen results; write §15 pre-registration |
| Sat 10-24 – Sun 10-25 | **T4.7 rotation 0** (~12 h) | T6.2 draft (motivation, data); literature and cost citations |
| Mon 10-26 – Tue 10-27 | T4.7 rotation 1 | T5.4–T5.6 analysis code on rot-0 results (class on 10-27) |
| Wed 10-28 – Thu 10-29 | T4.7 rotation 2; Tier 3 if ahead | T5.7–T5.8; draft results section |
| Fri 10-30 | Tier 3 / reruns | T5.9 final figures (individual presentations in class today) |
| Sat 10-31 – Sun 11-01 | — | T6.3 results, implications, limitations |
| Mon 11-02 | — | T6.4 appendices, AI disclosure; T6.5 review |
| Tue 11-03 | — | **Submit** |

**Cut lines if behind:** after 10-27, drop rotation 2. After 10-29, stop all new GPU work and
write with what exists.

---

## 10. Suggested team split (2–3 people)
| Role | Owns | Tasks |
|---|---|---|
| A: ML engineering | fine-tuning pipeline and GPU queue | T0.2–T0.3, T4.1–T4.9 |
| B: experiments and statistics | splits, frozen sweep, curve fitting, tests | T0.4, T1.1–T1.2, T3.*, T5.1–T5.7 |
| C: healthcare framing and report | data description, cost model, literature, figures, writing | T1.3, T5.8–T5.9, T6.* |

With 2 people, B also takes T5.8–T5.9. Everyone reviews the final report.

---

## 11. Risks and mitigations
| # | Risk | Mitigation |
|---|---|---|
| R1 | FT throughput lower than estimated | T4.4 decision rule (§6): smaller K, fewer rotations, partial FT, drop N = 2080 |
| R2 | GPU OOM | micro-batch 1 study, gradient checkpointing, K = 6; partial FT as last resort (applied to both models) |
| R3 | FT overfits or diverges at small N | LP-FT warm start, early stopping on val, lower LR. Report it; it is a finding about small-data FT |
| R4 | DINOv3 ConvNeXt download gated or fails | accept the license on HF and `hf auth login`. Else drop M2 and state that C2 is confounded (M1 vs M3 only) |
| R5 | Rare findings have few or no positives at small N | report curves by positives (T5.5), flag cells with <5 training positives, add a sensitivity macro (T5.4b) |
| R6 | Proxy labels are biased | compare models on the same labels (bias cancels in Δ), gold-58 check, state the limitation |
| R7 | Results depend on the test site-mix | 5 / 3 rotations; per-rotation figure (T5.7) |
| R8 | Laptop sleeps or overheats overnight | AC power, sleep disabled, checkpoint + resume (T4.3) |
| R9 | Timeline slip | tiering + cut lines (§9); the minimum viable report is defined in §1 |
| R10 | Claude sandbox blocks torch DLLs | run with the sandbox disabled, or the user runs the commands |
| R11 | Leakage via feature standardization or selection | pool-only mu/sd; selection on val only; gold and test scored once |
| R12 | Test-fold peeking during development | analysis code may read test metrics only after the sweep for that block finishes; the LR pilot uses val only |

---

## 12. Related work (for the report's background; verify each citation before use)
- **CNN vs ViT for medical images:** Matsoukas et al., *Is it Time to Replace CNNs with
  Transformers for Medical Images?*, ICCV-W 2021 (arXiv 2108.09038). CNNs win from scratch,
  ImageNet-pretrained ViTs ≈ CNNs, self-supervised ViTs slightly better. 2D only. Extended
  version: arXiv 2303.07034.
- **Sample efficiency of ViT vs CNN:** Murphy et al., Radiology: AI 2022 (PMC9745440). DeiT-B ≈
  DenseNet121 in sample efficiency on chest X-ray/MURA; stable down to ~10% of the data, then a
  drop. Behrendt et al. 2022 (arXiv 2208.08166): data-efficient ViTs on chest X-ray.
- **Self-supervision and data efficiency:** Azizi et al., *REMEDIS*, Nature Biomedical
  Engineering 2023. Matched supervised baselines with 1–33% of the retraining data.
- **DINOv3:** Siméoni et al. 2025 (arXiv 2508.10104). Medical evaluation: *Does DINOv3 Set a
  New Medical Vision Standard?* (arXiv 2509.06467), frozen features only; MedDINOv3
  (arXiv 2509.02379).
- **Foundation models for 3D classification:** AnyMC3D (arXiv 2512.12887): frozen 2D
  backbone + attention pooling beats 3D models; DINOv3 ≈ medical models once adapted.
- **Knee MRI:** MRNet (Bien et al., PLOS Medicine 2018); CoPAS (Nature Communications 2024,
  12 knee abnormalities, AUC 0.812).
- **Learning curves and sample size:** Cho et al. 2015, *How much data is needed to train a
  medical image deep learning system…?* (arXiv 1511.06348); Figueroa et al. 2012, *Predicting
  sample size required for classification performance*, BMC Med Inform Decis Mak (inverse
  power law fitting); Hestness et al. 2017, *Deep Learning Scaling is Predictable,
  Empirically*; Viering & Loog 2022, *The Shape of Learning Curves: A Review*, IEEE TPAMI.
- **Methods:** Ilse et al. 2018 (attention-based MIL); Kumar et al. 2022, *Fine-Tuning can
  Distort Pretrained Features…* (LP-FT); Liu et al. 2022 (ConvNeXt); Dosovitskiy et al. 2021 (ViT).
- **The contribution in one line:** the first data-size study on multi-sequence, multi-site 3D
  knee MRI with 12 findings that compares frozen vs fine-tuned 2025 foundation models, with an
  architecture-controlled (same-pretraining) ViT vs CNN pair, translated into hospital cost terms.

---

## 13. Integrity, data use and AI disclosure
- **Course rule:** "you may not reproduce or adapt existing code, notebooks, or analyses
  associated with the dataset." This project uses **no public Kaggle notebooks**. The
  preprocessing pipeline (§4.2) is a team member's own prior work and is reused as
  infrastructure. The report states this explicitly. **Email the professor/TA to confirm**
  (status: [ ] sent, [ ] confirmed).
- **Dataset terms:** check the competition's Rules/Data tab for permitted uses
  (non-commercial research/education). Status: [ ] checked.
- **AI disclosure** (required): the report appendix lists how AI tools were used:
  - idea refinement and proposal writing (Claude);
  - planning (this document);
  - code generation (list which scripts);
  - the earlier preprocessing.
  - Analysis interpretation and conclusions are the team's own.
  - Keep `RESULTS_LOG.md` accurate so this is easy to write.
- **No protected health information** leaves the machine. Do not upload images or reports to
  external services.

---

## 14. Appendix: constants and command cheat-sheet

```
PYTHON = C:\Users\tirth\AppData\Local\Python\pythoncore-3.11-64\python.exe
LABELS = ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA,
         Effusion, Synovitis, Baker's, Contusion, Fracture
KEYS   = acl, mcl, mm, lm, moa, loa, pfoa, eff, syn, bak, con, frx
SLOTS  = Sag_0, Sag_1, Cor_0, Cor_1, Axi_0, Axi_1      (_1 = fluid-sensitive fat-suppressed)
SIZES  = 65, 130, 260, 520, 1040, 2080, 2600
SEEDS  = 0, 1, 2      ROTATIONS: frozen 0-4, FT 0-2
PIXELS = T:\Kaggle_Knee_Competition\work\cache320\pixels_train.u8   uint8 (819078,320,320)
INDEX  = T:\Kaggle_Knee_Competition\work\cache320\index_series_train.parquet
TARGET = T:\Kaggle_Knee_Competition\work\targets.parquet
EMB    = T:\Kaggle_Knee_Competition\work\cache320\emb_train_<tag>.f16 (+ .json)
REUSE  = stage4_train_head.py: Head, build_bags, collate, predict, macro_auc
         stage4_compare.py:    boot_ci, boot_delta
         stage4_embed.py:      SliceSet pattern, embedding CLI
```

Example commands (filled in by implementers as scripts land):
```
%PYTHON% code\env_check.py
%PYTHON% code\make_splits.py && %PYTHON% code\make_subsets.py
%PYTHON% code\train_frozen_lc.py --model M2 --rotation 0 --seed 0 --n 260 --lr 3e-4 --drop 0.2
%PYTHON% code\sweep_frozen.py
%PYTHON% code\train_ft.py --model M1 --rotation 0 --seed 0 --n 520 --lr 3e-5
%PYTHON% code\sweep_ft.py --rotations 0,1,2
%PYTHON% code\collect_results.py && %PYTHON% code\fit_curves.py && %PYTHON% code\stats_tests.py
%PYTHON% code\make_figures.py
```

**Glossary:**
- **study / exam:** one patient's MRI session (here, 1 study = 1 patient).
- **series:** one acquisition (plane + sequence).
- **slice:** one 2D image.
- **slot:** plane × fluid-sensitivity category.
- **proxy label:** a report-derived label.
- **gold:** an expert image label.
- **frozen:** pretrained weights fixed.
- **FT:** fine-tuned.
- **LLRD:** layer-wise learning-rate decay.
- **LP-FT:** linear-probe-then-fine-tune.
- **macro AUC:** the mean of the 12 per-finding ROC-AUCs.

---

## 15. Decision log (append-only; newest last)
| Date | Decision | Reason | By |
|---|---|---|---|
| 2026-10-20 | Rotating test folds (5 frozen / 3 FT) instead of one fixed test fold | folds differ sharply by vendor (fold 0 is 80% Philips; fold 3 is 100% 1.5 T) | planning |
| 2026-10-20 | Sizes {65…2600}, geometric and nested; fixed validation fold | learning-curve fitting needs log spacing; nesting reduces noise | planning |
| 2026-10-20 | M1 vs M2 (same DINOv3 pretraining) as the main architecture comparison | isolates architecture from pretraining data | planning |
| 2026-10-20 | 224 px for all models; 320 only as a Tier 3 frozen ablation | fair comparison + affordable FT | planning |
| 2026-10-20 | Selection on the val fold only; gold-58 never used for selection | prevents optimistic bias; gold too small | planning |
| 2026-10-25 | **No new LLM labeling run is needed.** `work\labels_qwen3_14b.jsonl` covers 4,407/4,407 studies, every row `ok`, 0 missing cells (37,174 neg / 8,167 pos / 7,543 `na`) | audited 2026-10-25. `na` = the report does not address the finding; re-running an LLM cannot recover it. Already handled via weight 0.5 in `targets.parquet` (§4.5). Only 11 studies are `na` on all 12 findings (truncated or one-line reports, median 143 chars) | audit |
| 2026-10-25 | **`na` labels audited by hand; kept as is.** Of 84 random `na` cells (7 per finding): 69 correct, 11 clear errors, 4 arguable. **All 15 should have been `neg`, none `pos`**, and their soft targets were already low (p = 0.05–0.15). The errors cluster in Bulgarian (5/6) and Greek (3/6) reports, where the model missed group statements such as "cruciate ligaments intact", "collateral ligaments unremarkable", "articular cartilage normal" and "bones normal". Gold-58 cross-check: 76 of the 89 `na` cells are truly negative; Synovitis is the exception (9/32 positive) | effect on training ≈ weight 0.5 instead of 1.0 on ~2% of cells, with the right target direction. It is identical across all models, so the comparisons are unaffected. State it as a limitation. Files: `T:\Group_Project\label_audit\` | audit |
| 2026-10-25 | **`pos` labels audited by hand; kept as is for the main study.** 84 random `pos` cells (7 per finding): 64 correct, 12 arguable, 6 over-calls, 2 outright misreads. Wrong cells carry high targets (mean p = 0.79). ACL, MM, MOA, PFOA and Synovitis were 7/7 correct. Weak spots: **MCL 2/7, Lateral OA 3/7, Contusion 3/7, Effusion 4/7.** The LLM does not apply the rubric thresholds: grade I–II sprains, intrasubstance meniscal signal, <50% cartilage loss and a 1 cm Baker's cyst were all called positive. Gold-58 PPV of `pos` is 147/182 = 0.81, lowest for Lateral OA (7/13) and Contusion (16/27), which agrees with this audit | label noise is identical across all compared models, so the comparisons stay fair. Report it as a limitation, and interpret per-finding results for MCL / Lateral OA / Contusion with care. Optional targeted re-label: §7 / user decision. Files: `T:\Group_Project\label_audit\pos_audit.csv` | audit |
| 2026-10-25 | **Relabel tested and rejected; train on the v1 labels (`work\targets.parquet`).** v2.1 prompt on 226 dev studies, gold-58 macro AUC: v1 0.8653 · qwen3:14b v2.1 0.8337 (Δ −0.031 [−0.061, −0.004]) · same without evidence quotes 0.8210 · qwen3.5:9b v2.1 0.8680 (Δ +0.001) · mean(v1, 9b) 0.8826 (Δ +0.017 [+0.003, +0.033]). The ensemble gain was found on the same 58 studies, so it is mildly optimistic, and it would cost ~10 GPU-h. Code kept in `T:\Group_Project\labeling\` | user chose to start training now on v1 labels | user + labeling chat |
| 2026-10-25 22:50 | **Pre-registration (before any main result exists):** H1–H5 and the three finding groups exactly as written in §5.10, unchanged | required by §5.10 | training chat |
| 2026-10-25 | **Implementation details that deviate from or refine the plan.** (a) The frozen head keeps every study's bag as padded GPU index tensors built from `build_bags(cap=384)`. Same bags, one gather per batch, needed to make 1,680 runs fast. (b) Feature mu/sd come from 200k sampled slices of **all** pool-study slices (`common.pool_musd`, cached in `results/musd/`), shared by the frozen and FT pipelines. (c) FT train/val bags are capped at **48 slices per study** (K lowered for studies with more than 6 series; max is 14) so the largest batch has a fixed size. (d) Every process caps itself at 90% of VRAM (`torch.cuda.set_per_process_memory_fraction`), so Windows cannot silently spill to shared memory. Over-allocation is a catchable OOM, and FT drops that micro-step and logs it. (e) FT DataLoader uses 3 train + 2 eval workers: each Windows worker costs ~1.7 GB RAM. (f) The FT resize/augment runs on the GPU (`ft_data.gpu_prep`) | memory safety on a 12 GB GPU / 31 GB RAM laptop | training chat |
| 2026-10-25 23:45 | **T4.4 FT config: micro-batch 1 study, ≤48 slices, K=8, gradient checkpointing OFF, 3 train + 2 eval workers, identical for all models.** Worst-case benchmark (`results/tables/throughput.csv`): M1 189 sl/s, 5.5 GiB reserved; M2 189 sl/s, 7.4 GiB; M5 ResNet-18 scratch 1,234 sl/s, 1.0 GiB; M6 ConvNeXt-T scratch 468 sl/s, 3.4 GiB. The VRAM cap is 10.7 GiB. Checkpointing saves memory but is ~20% slower (148–155 sl/s). M2 with micro 2 and no checkpointing hit the cap and was caught as an OOM, so the cap works. Loader: 3 workers give 2,710 sl/s. Equivalence test: cosine ≥ 0.99994 for M1 and M2, PASS. Measured cost per FT run: validation 70 s × 10, final test+gold 5.2 min. Throughput is above the §6 decision threshold (120 sl/s), so the full plan stands | T4.4 decision rule | training chat |
| 2026-10-25 | **T4.9 from-scratch lower bound = M5 ResNet-18** (random init, rotation 0, N ∈ {260, 1040, 2600}, 3× budget, AdamW lr 1e-3, no LLRD, no LP phase). M6 ConvNeXt-T is supported as an option (~3.6 GPU-h) | fast enough to fit the queue (~1.6 GPU-h) | training chat |
| 2026-10-26 03:32 | **T4.6 LRs: 1e-4 (top backbone LR) for both M1 and M2**, chosen on val only (r0 s0 N=520). Val macro AUC at LR 1e-5 / 3e-5 / 1e-4: M1 .700 / .733 / .749; M2 .721 / .754 / .786. ⚠ The best LR is the **top of the grid for both models**, so a higher LR (e.g. 3e-4) might do better still. This applies equally to both models, so the comparison is fair, but FT may be slightly under-tuned. Optional: a 3e-4 pilot (2 runs, ~1 h) after rotation 0 | §5.8 | training chat |
| 2026-10-31 | **Competition results are cited as external context, with a label-quality caveat** (`results/competition_context.md`). Final private LB: 1st 0.960 (Team Kangaroo), gold cutoff 0.954. The winner built supervision with frontier multimodal LLM agents (reported GPT-6 Astra/Luna, GPT-6.1 Sol) and trained a single ConvNeXt-T (~28M params). Our labels come from a local 14B model reading report text (gold-58 0.866); our best FT models reach gold-58 0.84 at N=2600. The gap is reported as a bundle (labels, test set, data size, resolution, ensembles), not attributed to labels alone. We could not replicate frontier labelling for cost reasons, and the no-data-leaves-the-machine rule rules out hosted APIs anyway | user request; answers "how big a model is needed" | user + analysis chat |
| 2026-10-31 | **Comparisons use a "matched" design.** Every frozen-vs-fine-tuned and fine-tuned M1-vs-M2 statement uses rotations 0-2, seed 0 only: the same training subsets and test folds as FT. Frozen-only contrasts (C2-frozen, C3, C4) use the full 5 folds x 3 seeds | frozen means shift by 0.01-0.02 depending on which folds/seeds are averaged, so unmatched averages would bias the contrasts | analysis chat |
| 2026-10-31 | **Bootstrap deviation:** curve-fit and paired-difference intervals resample rotation x seed units AND, within each unit, the test studies (shared study-level resamples, `common.boot_counts`), instead of resampling units only | FT has 3 units, i.e. only 10 distinct unit resamples, which gives degenerate intervals | analysis chat |
| 2026-10-31 | **H5 not supported; no plateau is identifiable in 65-2,600 exams.** Log-linear beats the power law by AIC for 8/10 curves, and the power-law plateau sits at its 1.0 bound for 4/10. So N_0.01/N_0.02 ("exams to plateau") are **not reported**. They are replaced by (a) AUC gained per doubling of N (log-linear slope, bootstrap CI) and (b) exams needed to reach AUC 0.75/0.80/0.85, read off the observed curve inside the data and log-linearly extrapolated beyond 2,600 (flagged †, not reported beyond 26,000). Fig. 5 shows per-finding gain per doubling instead of N_0.02 | pre-registered quantity not estimable; reporting it would mean quoting 10^6-10^9 exams | analysis chat |
| 2026-10-31 | **Cost-model assumptions:** cloud GPU $0.59/h (range 0.34-0.89; RunPod, updated 2026-10-08); expert labelling 8 min/exam (range 5-20; Kwee et al., Tomography 2024), 2 readers (range 1-3), $170.17/h (range 115-250; BLS OEWS May 2023); LLM labels measured at 8.4 GPU-h per 4,407 reports | T5.8; all assumptions cited in the workbook (Appendix F) | analysis chat |
| 2026-10-31 | **Final deliverable is a report workbook, not report prose.** `report/Final_Report_Workbook.docx` holds the figures, tables, evidence with source files, syllabus checklist, page budget and per-section questions; the team writes the text | syllabus "Use of AI": AI only as support, no AI-generated report | user + analysis chat |
