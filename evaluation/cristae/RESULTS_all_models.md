# Cristae Model Evaluation — All Models on Pinned Test Split

_Last updated: 2026-06-29. bs24-AMP eval completed 2026-06-29. Test split: `SYNAPSENETV1_TEST_SPLIT` — 15 files (11 wichmann + 4 cooper),
held out of every training run. Primary metric: **macro AP** (threshold-free average precision within
`state==1` annotated-mito voxels). Secondary: Dice/Precision/Recall over the whole volume at fixed
threshold.  `—` = not measured for that model._

---

## Leaderboard (sorted by AP)

| Short name | Date | AP ↓ | AUC | Dice | P | R | Notes |
|---|---|---|---|---|---|---|---|
| **repro0601-exactsplit** ★ | 2026-06-18 | **0.837** | 0.952 | 0.744 | 0.750 | 0.753 | per-sample loss · aug · exact 120-file split |
| **persample-aug-legacy** | 2026-06-18 | **0.830** | 0.953 | 0.744 | 0.720 | 0.780 | per-sample loss · aug · extended · bad files excluded |
| **persample-aug-allfiles** | 2026-06-25 | **0.828** | 0.953 | 0.748 | 0.728 | 0.779 | per-sample loss · aug · allfiles (incl. needs_corrections) |
| **bs24-AMP-allfiles** | 2026-06-26 | **0.832** | 0.953 | 0.748 | 0.742 | 0.769 | per-sample · aug · allfiles · bs=24 · AMP bf16 |
| newloss-aug-legacy | 2026-06-16 | 0.822 | 0.949 | 0.733 | 0.703 | 0.783 | pooled loss · aug · extended · bad excluded |
| flashoptim-allfiles | 2026-06-26 | 0.807 | 0.946 | — | — | — | fp32 (no AMP) · bs=12 · early-stopped ep 14 |
| repro0601-legacy | 2026-06-15 | 0.767 | 0.936 | 0.699 | 0.682 | 0.740 | pooled loss · **no aug** · legacy |
| 2026-06-01 (original) | 2026-06-01 | 0.837† | — | 0.763 | 0.769 | 0.768 | original gold; AP inferred from repro-exactsplit |
| synapsenet cristae3 | — | — | — | 0.760 | 0.803 | 0.726 | external SynapseNet model (1.44 nm) |
| synapsenet cristae4 | — | — | — | 0.759 | 0.766 | 0.758 | external SynapseNet model (1.74 nm) |
| repro0601-oldloss | 2026-06-16 | — | — | 0.713 | 0.664 | 0.792 | old pooled loss · bad files incl. |
| 2026-06-04 | 2026-06-04 | — | — | 0.722 | 0.689 | 0.772 | +5 high-res cooper B4 tomos |
| norm-variant | — | — | — | 0.707 | 0.686 | 0.743 | normalize during training; inference bug patched |
| groupedsplit-corrected | 2026-06-11 | — | — | 0.681 | 0.758 | 0.636 | grouped split, corrected annotations |
| flashoptim-groupedsplit | 2026-06-12 | — | — | 0.671 | 0.744 | 0.635 | flashoptim/fp32 · grouped split |

★ Gold standard — same recipe as original 06-01, reproduced on current codebase.  
† AP for the original 06-01 is inferred from its reproduction (repro0601-exactsplit); the original
  checkpoint was not re-evaluated with the AP script.

---

## Per-file AP breakdown (AP-evaluated models only)

Rows sorted by difficulty (hardest → easiest). `repro-exact` = repro0601-exactsplit.

| Test file | repro-exact | bs24-AMP | persample-legacy | persample-allfiles | newloss-legacy | flashoptim | repro-legacy |
|---|---|---|---|---|---|---|---|
| Otof_D3 (D3_4model)      | 0.517 | 0.534 | 0.534 | 0.508 | 0.522 | 0.521 | 0.532 |
| M8_eb6                   | 0.684 | 0.688 | 0.625 | 0.653 | 0.619 | 0.693 | 0.530 |
| Otof_455L_KO (B3_2)      | 0.779 | 0.709 | 0.780 | 0.778 | 0.746 | 0.475 | 0.369 |
| M2_eb5                   | 0.797 | 0.811 | 0.827 | 0.824 | 0.806 | 0.794 | 0.800 |
| KO8_eb4                  | 0.817 | 0.817 | 0.809 | 0.790 | 0.798 | 0.832 | 0.806 |
| WT22_eb5                 | 0.849 | 0.849 | 0.844 | 0.836 | 0.828 | 0.821 | 0.798 |
| WT40_eb10                | 0.841 | 0.834 | 0.836 | 0.830 | 0.822 | 0.843 | 0.802 |
| WT21_syn4                | 0.855 | 0.860 | 0.855 | 0.854 | 0.848 | 0.848 | 0.833 |
| WT21_eb5                 | 0.864 | 0.862 | 0.859 | 0.831 | 0.848 | 0.876 | 0.810 |
| M5_eb1                   | 0.896 | 0.893 | 0.886 | 0.887 | 0.891 | 0.837 | 0.820 |
| M1_eb6                   | 0.872 | 0.855 | 0.836 | 0.872 | 0.855 | 0.837 | 0.835 |
| WT20_eb5                 | 0.892 | 0.890 | 0.892 | 0.889 | 0.877 | 0.865 | 0.863 |
| 36859_J1_PS_03 (cooper)  | 0.939 | 0.939 | 0.931 | 0.920 | 0.928 | 0.924 | 0.847 |
| 36194_B4_SC_22 (cooper)  | 0.976 | 0.972 | 0.976 | 0.976 | 0.978 | 0.971 | 0.957 |
| 36194_B4_SC_01 (cooper)  | 0.976 | 0.967 | 0.964 | 0.972 | 0.972 | 0.963 | 0.900 |
| **Macro AP**             | **0.837** | **0.832** | **0.830** | **0.828** | **0.822** | **0.807** | **0.767** |

**Key observations:**
- `Otof_D3` and `M8_eb6` are consistently the hardest (~0.51 and ~0.63–0.69); annotation-quality ceiling.
- `Otof_455L_KO` is anomalous: AP=0.78 for the best models but collapses to 0.37–0.47 for repro0601-legacy and flashoptim — those recipes over- or under-predict severely on this file.
- Cooper tomograms (`36194_B4`, `36859_J1`) are the easiest (0.92–0.98); high-SNR, clean annotations.
- The gap between best (0.837) and worst (0.767) is driven by `Otof_455L_KO`, `M8_eb6`, and `36859_J1` — the hard wichmann + middle-difficulty cooper file.

---

## Metric definitions

**AP (average precision):** threshold-free; computed only within annotated-mito voxels (`state==1`).
Foreground probabilities (channel 1 of model output) are swept from 0→1; area under the P-R curve.
Computed by `evaluation/cristae/evaluate_cristae_ap.py`.

**AUC:** area under the ROC curve (same foreground channel, same `state==1` mask).

**Dice / Precision / Recall:** voxel-level binary metrics over the full volume. Threshold = 0.5 on
foreground channel. Computed by `evaluation/cristae/evaluate_cristae.py`.
Dice = 2·TP / (2·TP + FP + FN); reported as mean over the 15 test files.

**Why AP > Dice as primary metric:** Dice depends on a fixed threshold and the background/foreground
imbalance (cristae ≈ 16.7% of annotated-mito voxels). AP is threshold-free and measures the model's
ability to rank cristae voxels above background within the mito mask — more informative for how the
model will perform in downstream analysis.

---

## Full model paths (checkpoints)

| Short name | Checkpoint path |
|---|---|
| repro0601-exactsplit | `/mnt/lustre-grete/usr/u12103/cristae/checkpoints/cristae-net32-lr1e-4-bs8-ps32x256x256-repro0601-exactsplit-2026-06-18` |
| persample-aug-legacy | `/mnt/lustre-grete/usr/u12103/cristae/checkpoints/cristae-net32-lr1e-4-bs8-ps32x256x256-persample-aug-legacy-2026-06-18` |
| persample-aug-allfiles | `/mnt/lustre-grete/usr/u12103/cristae/checkpoints/cristae-net32-lr1e-4-bs8-ps32x256x256-persample-aug-allfiles-2026-06-25` |
| bs24-AMP-allfiles | `/mnt/lustre-grete/usr/u12103/cristae/checkpoints/cristae-net32-lr1.7e-4-bs24-ps32x256x256-persample-aug-allfiles-2026-06-26` |
| newloss-aug-legacy | `/mnt/lustre-grete/usr/u12103/cristae/checkpoints/cristae-net32-lr1e-4-bs8-ps32x256x256-newloss-aug-legacy-2026-06-16` |
| flashoptim-allfiles | `/mnt/lustre-grete/usr/u12103/cristae/checkpoints/cristae-net32-lr1.2e-4-bs12-ps32x256x256-flashoptim-persample-aug-allfiles-2026-06-26` |
| 2026-06-01 (original) | `/mnt/lustre-grete/usr/u12103/cristae/checkpoints/cristae-net32-lr1e-4-bs8-ps32x256x256-2026-06-01` |
| synapsenet cristae3 | via `synapsenet` CLI |
| synapsenet cristae4 | via `synapsenet` CLI |

Test segmentations (predictions + result CSVs) under:
`/mnt/lustre-grete/usr/u12103/cristae/test_segmentations/<model-name>/`
