# Diagnosis: cristae `2026-06-01` vs `2026-06-04` test-set divergence

**Question.** Why does `cristae-net32-...-2026-06-04` score *worse* on the held-out test split
than `...-2026-06-01` (macro Dice 0.763 → 0.722, precision 0.769 → 0.689, FP +40%), despite
*better* train/val during training?

## Headline answer

The 06-04 model **genuinely lost discriminative power** at separating cristae from non-cristae
**inside annotated mitochondria** (mito state == 1 — the only region the eval scores), and the loss
is **concentrated on hard/rare files**. It is **not** a tunable-threshold/calibration artifact: the
degradation shows up in **threshold-free** metrics (average precision, ROC-AUC) and as full
**precision–recall-curve dominance**, so no foreground threshold recovers 06-01's quality.

The train/val signal looked better because the validation metric is a **masked soft-Dice** over a
different region/distribution (and a different val split) — an easier, different objective that does
not capture hard-file discrimination on the test set.

## How the pipeline constrains the question (key facts)

- **Eval region.** `synapse/cristae/evaluate.py` sets `ignore_mask = mito_states != 1`, so all
  metrics are computed **only within annotated mito (state==1)**; `eval_voxels` == the state==1 voxel
  count (matches both published CSVs). ⇒ the measured +40% FP is **inside annotated mito**, not in
  the ignored unannotated (state==2) regions.
- **Inference.** `synapse_net` cristae segmentation = `foreground_prob > 0.5` (hard-coded) within the
  `mito > 0` mask → connected components → `min_size=500`. The boundary channel is computed but
  **unused**. ⇒ within state==1, precision is determined entirely by the **foreground probability map**.

## Evidence (diagnostic over all 15 test files; `evaluation/cristae/diagnose_cristae_probs.py`)

Within state==1, foreground-probability vs cristae GT:

| model | macro AP | macro AUC | mean fg-prob on **true cristae** | mean fg-prob on **non-cristae** |
|---|---|---|---|---|
| **06-01 best** | **0.840** | 0.954 | 0.762 | **0.078** |
| 06-04 best | 0.782 | 0.939 | 0.771 | **0.100** |
| 06-04 latest | 0.795 | 0.941 | 0.741 | 0.075 |

1. **Threshold-free degradation.** AP 0.840→0.782, AUC 0.954→0.939. Probability on *true* cristae is
   ~unchanged (0.76→0.77) while probability on *non-cristae inside mito* **rises** (0.078→0.100) — the
   classes become **less separable**. AP/AUC are ranking-based, so this is real, not a threshold offset.
2. **PR-curve dominance.** At matched recall, 06-04 precision < 06-01 on **every** file (e.g.
   Otof_AVCN07_455L at recall≈0.73: 0.64 vs 0.39; even 06-04's best-achievable precision never reaches
   06-01's curve). ⇒ threshold tuning cannot recover it.
3. **File-dependence.** Degradation tracks difficulty: Otof_AVCN07_455L **−0.340 AP**, M8_eb6 −0.136,
   WT21_eb5 −0.091, … down to the clean cooper tomograms 36194_B4_* at −0.002/−0.006 (unchanged).
4. **Worse than its own starting point.** 06-04 was **warm-started from 06-01**; continued training
   pushed hard-file AP *below* the 06-01 init (Otof_455L 0.764 → 0.424). ⇒ the continued-training run
   actively degraded discrimination on hard files.
5. **Not monotonic in epochs.** `latest` (epoch ~50) is slightly *better* than the val-selected `best`
   (epoch 34): AP 0.795 vs 0.782, non-cristae prob back to 0.075. ⇒ the masked-val checkpoint
   selection picked a point that over-predicts more than later training does.

## Ruled out

- **`ed2c08d` MaskedDiceLoss reduction rewrite** — numerically inert (proven equal to `torch_em.DiceLoss`).
- **Eval harness / data leakage** — identical eval region; both runs held out the **same 15** files
  (confirmed in both training logs). Identical `gt_fg`/`eval_voxels` per file across the two CSVs.
- **Normalization, label transform, sampler, model architecture** — identical across the two eras
  (git diff + checkpoint `init`; both checkpoints share the same `model_kwargs` and key layout).
- **State==2 masking-mechanism as the *primary* driver** — the worst-degrading file (Otof_455L) has
  **zero** state==2 voxels, yet degrades most.

## What actually differs between the two models (authoritative, from checkpoint `init`)

| | 06-01 | 06-04 |
|---|---|---|
| loss / metric | old `torch_em DiceLoss(channelwise, sum, eps, **ignore_label=None, ignore_state_value=2, state_channel=1**)` | `synapse.util.MaskedDiceLoss()` + `MitoStateMaskTransform` |
| init | (its own training) | **warm-started from 06-01** |
| training | ~41 epochs | ~50 epochs continued from 06-01 |
| data pool / split | 150 files, 80/10/10 (train 120) | 155 files, 90/10 of non-test (train 126) |
| architecture | AnisotropicUNet (norm, Sigmoid, 2→2) | **identical** |

So the masking *mechanism* changed (old DiceLoss built-in `ignore_state_value` vs
`MaskedDiceLoss`+`MitoStateMaskTransform`; both ignore state==2), alongside **continued training** and
a **modest data-pool/split** change.

## Attribution (ranked)

1. **Continued training over-fitting the bulk distribution (leading).** 06-04 = 06-01 + ~50 epochs and
   ends up *worse than its own init* specifically on rare/hard morphologies (Otof-KO, M8), while common
   clean tomograms are untouched and the masked soft-Dice val (dominated by common annotated mito)
   improved. This is the classic signature of over-fitting the majority distribution at the expense of
   the tail.
2. **Masking-mechanism change** (old `DiceLoss(ignore_state_value)` → `MaskedDiceLoss`+transform) —
   plausible secondary contributor, but *not* the main driver (worst file has no state==2).
3. **Data-pool/split change** — minor.

**Tier-2 probe (warm-start 06-01 under the current recipe): attempted, not viable at "quick node" scale.**
A reduced-setting probe (`configs/training/cristae/cristae_probe_warmstart.yaml`: warm-start from 06-01,
`MaskedDiceLoss`, bs4, patch 24×192×192) warm-started correctly and trained, but the pipeline is
**data-loading-bound at ~2 s/iter** (`MinInstanceSampler(p_reject=0.95)` + large-H5 I/O — the same rate
as the 65k-iter production run). The over-prediction effect in 06-04 developed over **~65k iters**, so a
genuinely short probe (~1–2k iters ≈ tens of minutes) would show negligible drift and be inconclusive;
a faithful probe needs **hours** and is effectively a full retrain. It was therefore stopped.
Importantly, the attribution does **not** depend on this probe: the production checkpoints already show
the effect directly — **06-04 was warm-started from 06-01 and ended up worse than that very init on the
hard files** (Otof_455L AP 0.764 → 0.424), which *is* the continued-training degradation, observed in the
real run. A definitive single-variable isolation would require full-scale controlled retrains (the
deferred SLURM option), not a quick interactive probe.

## Recommendations (deferred — this round is diagnose-only)

- **Select / early-stop on a hard, region-matched test metric**, not the masked soft-Dice — the val
  metric does not track test-set discrimination on hard files.
- **Counter the majority-distribution over-fit**: per-file/morphology-balanced sampling or loss
  weighting so rare hard cases (Otof-KO) aren't sacrificed; consider shorter / earlier-stopped training.
- **Threshold tuning** helps the easy files marginally but **cannot** fix the hard-file degradation
  (PR-dominance) — do not expect a threshold sweep alone to close the gap.

## Reproduce

```
# per-file AP/AUC + threshold sweep within state==1 (3 checkpoints, 15 files)
python evaluation/cristae/diagnose_cristae_probs.py \
  --model "m0601:<ckpts>/cristae-net32-...-2026-06-01:best" \
  --model "m0604:<ckpts>/cristae-net32-...-2026-06-04:best" \
  --model "m0604_latest:<ckpts>/cristae-net32-...-2026-06-04:latest" \
  --tile 32 256 256 --out_dir <out> --cache_dir <out>/cache
```
Outputs: `diag_summary.csv` (AP/AUC/prob stats), `diag_sweep.csv` (per-threshold precision/recall/Dice).
