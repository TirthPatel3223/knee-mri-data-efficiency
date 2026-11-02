# Competition context and the label-quality caveat (T5 analysis notes)

Compiled 2026-10-31 for the report's results/limitations discussion. These are **notes with
sources**, not report text: the team writes the report (course AI policy, §13). Every external
number below carries its source and how well it was verified.

## 1. Final results of the RSNA 2026 Knee Abnormality Detection competition

**Metric:** macro ROC-AUC over the 12 findings. It is scored against the host's **expert,
image-based, multi-reader labels** (two MSK radiologists + adjudicator; RESEARCH.md §1.5) on a
hidden test set of ~1,300 studies (private leaderboard = 70%).

**Private leaderboard** (Kaggle API, `kaggle competitions leaderboard
rsna-knee-abnormality-detection -s`, pulled 2026-10-31; **verified**):

| Rank | Team | Private AUC |
|---:|---|---:|
| 1 | Team Kangaroo | 0.960 |
| 2 | MURAD | 0.960 |
| 3 | DataTech Club | 0.959 |
| 10 | MEJAL | 0.957 |
| 20 | YC & NB | 0.955 |

Medal cutoffs, as reported in one bronze team's repo (TranBaDat2607, **secondary source**):

| Medal | Private AUC | Rank | Field |
|---|---:|---:|---|
| Gold | 0.954 | 22 | 6,425 teams |
| Silver | 0.943 | 321 | |
| Bronze | 0.942 | 642 | |

The field is extremely compressed: 20 places span 0.960 → 0.955.

## 2. What the top teams did (labels and model size)

### 1st place: Team Kangaroo

Kaggle write-up: kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution.
⚠ The Kaggle page did not render for automated fetching. The points below come from search
summaries of the write-up and a third-party repo, so **open the write-up and confirm them before
quoting**.

- **Labels:** they treated the contest mainly as a **labelling problem**. Frontier LLM agents
  produced three kinds of supervision for the training set:
  - per-study probabilities for the 12 findings;
  - a 3D ROI box per series and finding (anatomy-level; an agent checks each box);
  - per-slice lesion boxes.

  These agents read **reports and images**, using **GPT-6 Astra** (confirmed by the team on 2026-10-31). The write-up also names GPT-6 Luna and GPT-6.1 Sol.
  **GPT-6 Luna, GPT-6.1 Sol and GPT-6 Astra**.
- **Image model:**
  - a single **ConvNeXt-T (~28M parameters) at 256 px**, with ROI-box pooling and auxiliary
    lesion/report heads;
  - the write-up's own table puts the plain 12-label ConvNeXt-T baseline at ~0.933;
  - a third-party repo (Bassel1000) reports **0.964 public / 0.954 private for a single model**
    and 0.967 public for the ensemble.

  These figures are **unverified**.

### Other top solutions

| Placing | Labels | Image models | Score |
|---|---|---|---|
| 5th | report-derived "yes" labels near p≈0.75 were right only **38%** of the time | EfficientNetV2-M (2 families) + ConvNeXt-B, weighted ensemble | — |
| 8th | two rounds of pseudo-labels | CoAtNet-2 and CAFormer-S36 at 384×384 crops, **12–16-model ensembles** | private 0.957–0.958 |
| Bronze (TranBaDat2607) | `gpt-5.4-mini`, plus a Claude Opus 4.5 mix | CoAtNet ensemble | private 0.943 |
| Mid-table (aakibinesar) | LLM soft labels | ResNet-50 / ConvNeXt-T / ConvNeXt-S, 2.5D, 5-fold | private 0.874 |
| Public notebook, Aug (RESEARCH.md §1.8) | LLM report labels | frozen DINOv2-small + closed-form head | public 0.891 in 6 min |

Sources and verification:
- **5th place:** Kaggle write-up; search summary, **unverified**.
- **8th place:** TomokiHirose repo, **verified**.
- **Bronze (TranBaDat2607):** **verified**.
- **Mid-table (aakibinesar):** search summary.

**Takeaways (stated in the write-ups):**
- Label quality was the main lever, not backbone size.
- The winner's core image model is a **ConvNeXt-T, roughly a third the size of our 86–88M
  backbones**.
- Note that many other gold-zone teams still used ensembles of medium-to-large backbones at
  higher resolution. "Small models win" is true of the winner's single model, not of the
  whole top 10.

## 3. How this relates to our results

**Our labels:**
- a **local** qwen3:14b reads the **report text only**;
- they score **gold-58 macro 0.866** (FINDINGS.md, Stage 2);
- the stricter v2.1 prompt was worse (0.834; §15, 2026-10-25).

**Our best models (gold-58 macro, N = 2,600, rotations 0–2, seed 0; `results/runs.parquet`):**

| Regime | Model | Gold-58 macro (mean, range) |
|---|---|---|
| FT | M1 DINOv3 ViT-B | 0.840 (0.837–0.842) |
| FT | M2 DINOv3 ConvNeXt-B | 0.837 (0.825–0.853) |
| Frozen | M3 ImageNet ConvNeXt-B (best frozen) | 0.793 |
| Scratch | M5 ResNet-18 (r0 only) | 0.742 |

**Observation that supports the caveat with our own data:**
- The fine-tuned models at the largest N (0.84) are already within ~0.03 of the gold-58 score
  of the labels they were trained on (0.866).
- Their learning curves are flattening (T5.3 will quantify the asymptote).
- This is consistent with **label quality, not model capacity or architecture, being the
  binding constraint** in our setup.
- It is not proof: a model can in principle exceed its noisy labels, and gold-58 resolves only
  ~±0.03.

**Why the ~0.12 gap to the leaderboard (0.84 vs 0.96) cannot be pinned on labels alone.**
These differences come bundled:

1. **Labels:** frontier multimodal LLM agents reading images and reports, with box supervision,
   vs a local 14B model reading report text.
2. **Test set:** ~1,300 hidden studies with expert labels vs our gold-58 (CI ±0.03). Our
   primary metric (proxy AUC on held-out scanner folds) is a different ruler again; never put
   it on the same axis as leaderboard scores.
3. **Training data:** all ~4,400 studies vs at most 2,600 here, by design (learning curves).
4. **Resolution and cropping:** 256–384 px ROI crops vs a 224 px whole field of view.
5. **Ensembles:** 12–47 checkpoints vs single models.
6. **Task-specific architecture:** ROI pooling and auxiliary heads vs one shared attention head.

**Why we could not copy the winning labelling.** Two reasons, both worth stating:
- **Cost:** frontier-API labelling of ~4,400 studies with images plus agent loops is beyond the
  team's budget.
- **Data governance (stronger reason):** this project's rule is that **no images or reports
  leave the machine** (PROJECT_PLAN §0/§13). That rules out hosted LLM APIs regardless of cost.
  The Kaggle host allowed hosted LLMs for the competition (RESEARCH.md §1.7). A hospital,
  however, faces the same governance constraint we chose, which makes it a realistic
  limitation, not just a budget one.

**Answer to "how big a model is needed?":**
- Competition evidence: a ~28M-parameter ConvNeXt-T reached ≥0.95 when trained on high-quality
  supervision.
- Our evidence:
  - at N = 2,600 the fine-tuned ViT-B and ConvNeXt-B are tied (H2);
  - in the frozen regime, an older ImageNet ConvNeXt-B beat both DINOv3 backbones.
- Both point the same way: **beyond a modest pretrained backbone, model size and architecture
  matter far less than label quality and the amount of labelled data.** For the hospital
  question (RQ5) this means investing in labels, not in a bigger model.

## 4. Course-integrity note

The course rule forbids reproducing or adapting existing code, notebooks or analyses for this
dataset.
- **Citing** published competition results as external context, with links, is citation, not
  adaptation.
- We used none of their code, labels or methods.
- Cite the write-ups/leaderboard as sources, and keep them out of our methods section.

## Sources

- Leaderboard: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard (via Kaggle API)
- 1st place: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/1st-place-solution
- 5th place: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/writeups/5th-place-solution
- 8th place: https://github.com/TomokiHirose/rsna_2026_8th_place_solution
- Kangaroo summary + single-model scores: https://github.com/Bassel1000/rsna-knee-abnormality-detection
- Bronze, medal cutoffs: https://github.com/TranBaDat2607/RSNA-Knee-Abnormality-Detection
- 0.874 entry: https://github.com/aakibinesar/RSNA-Knee-Abnormality-Detection-Kaggle-Competition
- 18th public (Aug): https://huggingface.co/blog/bishnoiyash/rsna-competetion
