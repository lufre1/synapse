# Why cristae `2026-06-01` outperforms ALL later models

Extends `DIAGNOSIS_cristae_06-01_vs_06-04.md` to the post-refactor from-scratch models.
Threshold-free diagnostic complete (15 test files, within annotated mito, state==1).

## Result

| model | iters/epochs | AP | AUC | fg-prob **true** | fg-prob **non** | signature |
|---|---|---|---|---|---|---|
| **m0601** | 74676 / 41 | **0.837** | 0.954 | 0.762 | 0.078 | balanced (best) |
| m0604 (warm-start of 0601) | 65275 / 34 | 0.775 | 0.939 | 0.771 | 0.100 | **over**-predicts |
| grouped_best | 13488 / 7 | 0.777 | 0.935 | 0.642 | 0.057 | **under**-predicts |
| grouped_latest | 40464 / 23 | 0.788 | 0.927 | 0.585 | 0.036 | **under**-predicts (more) |

Source CSVs: `/mnt/lustre-grete/usr/u12103/cristae/diagnostics/why_0601_wins/diag_{summary,sweep}.csv`.

## What this proves

1. **06-01's advantage is a genuine, threshold-free discrimination gain** (AP 0.837, per-file it beats
   both grouped variants almost everywhere). Not a calibration/threshold artifact.
2. **The two later models fail in OPPOSITE directions** — m0604 over-predicts (prob on non-cristae
   0.078→0.100), grouped under-predicts (prob on true cristae 0.762→0.64). 06-01 is the only model
   simultaneously confident on true cristae **and** quiet on non-cristae.
3. **Post-refactor training actively degrades a good model:** m0604 was warm-started from 06-01
   (≈0.84) and continued training pulled it down to 0.775.
4. **Val/test divergence:** grouped's val soft-Dice (best_metric 0.87) is *better* than 06-01's (1.04),
   yet its test AP is worse — the masked soft-Dice val metric (dominated by easy background, which both
   recipes include) does not track in-mito discrimination.

## RULED OUT (definitively)

- **Architecture** — all checkpoints `norm=None`, identical `model_kwargs`.
- **Inference threshold / operating point** — AP/AUC are threshold-free; 06-01 dominates.
- **Early-stopping / training length** — grouped_latest (ep23) ≈ grouped_best (ep7) in AP; training
  *longer* lowers `fg|true` (0.642→0.585), i.e. it converges to an under-confident regime regardless.
- **Loss reduction (new vs legacy)** — numerically inert (verified 2026-06-09).
- **Loss/masking mechanism** — **PROVEN equivalent**: 06-01's old `DiceLoss(ignore_state_value=2,
  state_channel=1)` builds `valid=(state!=2)` and applies it in num+den; `MaskedDiceLoss`+
  `MitoStateMaskTransform` build `mask=(state!=2)` and apply `pred·mask, tgt·mask`. With mask∈{0,1}
  (mask²=mask) the dice is algebraically identical. Both keep background+annotated-mito, exclude state 2.
- **Eval harness / test labels** — identical `eval_voxels`/`gt_fg` across all CSVs.

## Remaining candidates (need ONE controlled retrain to isolate)

The gap is a real discrimination loss common to every post-refactor model. With code-level causes
eliminated, the cause lies in the **training data / recipe**:

1. **Data regeneration / drift** — training `_combined.h5` modified after 06-01 finished (5 cooper
   files 06-04; 26 wichmann 06-01 — the latter *before* 06-01 finished on 06-03, so seen by 06-01 too).
   Cooper drift alone (5 files) is likely too small to explain a broad drop.
2. **Split / data composition** — legacy 80/10/10 (train ~120) vs `grouped_stratified` (train 115);
   different train file *set* and val distribution.
3. **Training dynamics** — 06-01 ran 74k iters; the from-scratch grouped optimum is under-confident
   and worsens with more iters → the recipe converges to a different, worse operating point.

## Decisive next experiment (one run)

**Replicate the 06-01 recipe on current code:** from-scratch, `split_strategy=legacy` (80/10/10),
same data dirs, ~41 epochs / ~75k iters, `MaskedDiceLoss` (proven == old loss), pinned test split.
Then segment+eval+diagnose.
- Reaches ≈0.84 AP → cause was the **newer configs' recipe** (split/data/duration), not the refactor.
- Stuck ≈0.78 AP → cause is in the **refactored data pipeline / regenerated data** (deeper dig:
  per-file label/raw diff of regenerated H5s; sampler/patch-composition; augmentation pipeline).
