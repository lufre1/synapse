# Cristae loss: OLD (2026-06-01) vs NEW — structural & statistical differences

**TL;DR.** The old and new cristae losses are **not** the same function. The earlier "numerically
identical" check only compared `MaskedDiceLoss` (new) vs `MaskedDiceLossLegacy` — **both batch-pooled**.
The genuinely *old* loss (what the 2026-06-01 model used, restored on the `cristae-oldloss-repro`
torch-em branch) is **per-sample** (mean-of-ratios), reads its mask from the model **input** via a
trainer hook, and — as a side effect — leaves the loader's augmentation slot free so training runs
**with** torch_em's default Kornia augmentation. Three differences, summarized:

| | OLD (06-01) | NEW (current) |
|---|---|---|
| where the mask comes from | model **input** channel, passed to the loss by the trainer (`loss(pred, y, state=x)`) | appended to the **target** by `MitoStateMaskTransform` |
| batch reduction of Dice | **per-sample**: Dice per batch element, then averaged (mean-of-ratios) | **batch-pooled**: one Dice per channel over all `B·voxels` (ratio-of-sums) |
| augmentation | default `KorniaAugmentationPipeline` (transform slot free) | **none** (transform slot occupied by the mask transform) |

Only the **batch reduction** changes the loss *function*; the augmentation difference is a *training-data*
difference that the mask-mechanism change caused as a side effect (and which also shifts the loss curve).

---

## 1. Structural differences (code)

**OLD** (`torch_em/loss/dice.py` on branch `cristae-oldloss-repro`, + `torch_em/trainer/default_trainer.py`):
- `DiceLoss(channelwise=True, reduce_channel="sum", ignore_state_value=2, state_channel=1)`,
  `forward(input_, target, state)`. `valid = (state[:, 1:2] != 2)` is built from the **model input**.
- The trainer calls `loss = self.loss(pred, y, state=x)` and `metric = self.metric(pred, y, x)` — it
  hands the network input `x` (channels `[raw, mito_state]`) to the loss.
- `dice_score(..., valid=valid)` reduces **spatial dims only** → `[B, C]`, then **`dice.mean(dim=0)`**
  (average over the batch), then `1 − ·`, then `sum` over channels.
- Mask is *not* in the target; the target is just `BoundaryTransform` output `[boundary, fg]`.

**NEW** (`synapse/util.py`):
- `MitoStateMaskTransform` appends the mask to the target → `[boundary, fg, mask, mask]`.
- `MaskedDiceLoss.forward(pred, target)` splits target into `tgt` + `mask`, multiplies
  `p = pred·mask`, `t = tgt·mask`, then flattens `[B, C, …] → [C, B·DHW]` and computes **one** Dice per
  channel over the whole batch (`num = Σ_{B·DHW} p·t`), `1 − ·`, `sum` over channels.

Both mask exactly the same voxels (`state == 2` excluded; background + annotated mito kept) and both
sum Dice over the 2 channels. The difference is purely **how the batch is reduced**.

---

## 2. The mathematics

For channel `c`, batch element `b`, valid voxels `v` (mask `m`), prediction `p`, target `t`, define

```
num_{b,c} = Σ_v m·p·t           den_{b,c} = Σ_v m·p²  +  Σ_v m·t²           D_{b,c} = 2·num_{b,c} / den_{b,c}
```

- **OLD (per-sample, mean-of-ratios):**
  `L_old = Σ_c [ 1 − (1/B) Σ_b D_{b,c} ]`   — each patch contributes its **own** Dice, equally weighted.

- **NEW (batch-pooled, ratio-of-sums):**
  `L_new = Σ_c [ 1 − ( 2·Σ_b num_{b,c} ) / ( Σ_b den_{b,c} ) ]`

Rewrite the pooled Dice as a **weighted** average of the per-sample Dice:

```
D_c^pool = Σ_b w_{b,c} · D_{b,c},     with weights   w_{b,c} = den_{b,c} / Σ_b den_{b,c}
```

i.e. the pooled Dice weights each patch by `den_{b,c}` — roughly the **number of positive voxels**
(its "size"). The per-sample Dice uses **equal** weights `1/B`. They are equal **iff** every patch has
the same `den_{b,c}` (uniform batch); otherwise they differ. This is the classic *mean-of-ratios ≠
ratio-of-means* (a Jensen-/Simpson-type effect).

---

## 3. Why the OLD loss reads HIGHER (statistically)

Cristae are **sparse and unevenly distributed** across patches, so `den_{b,c}` varies a lot within a
batch (some patches have many cristae voxels, some almost none; the boundary channel is sparser still).

- In the **pooled** Dice, low-density patches have **small weight** `w_{b,c}` → their (typically low)
  Dice barely affects the score → the batch score is dominated by the easy, high-density patches → Dice
  high → loss **low**.
- In the **per-sample** Dice, every patch gets weight `1/B` → the hard, low-density patches (low Dice,
  or Dice ≈ 0 for a near-empty channel) **count fully** → average Dice lower → loss **higher**.

So the per-sample loss sits systematically **above** the pooled loss, and the gap **grows with batch
heterogeneity**. Numerical check (this repo, `DiceLoss` per-sample vs `MaskedDiceLoss` pooled, identical
inputs/mask):

| batch | OLD per-sample | NEW pooled | ratio |
|---|---|---|---|
| uniform density | 1.160 | 1.107 | 1.05× |
| half-empty patches | 1.617 | 1.387 | 1.17× |
| mixed density (0.5 … 0.001) | 1.580 | 1.285 | 1.23× |

(And `MaskedDiceLossLegacy == MaskedDiceLoss` only on `main`, where `torch_em.DiceLoss` is also pooled;
on the repro branch the legacy wrapper becomes per-sample because it delegates to the patched
`torch_em.DiceLoss`.)

---

## 4. Why it matters for training (not just the curve)

The two reductions imply **different gradients**:

- **Per-sample (old):** every patch contributes equally to the gradient, regardless of how many cristae
  voxels it contains. Rare/small structures and sparse patches get the **same say** as dense ones — a
  built-in counter to class/size imbalance. Plausibly why 06-01 generalizes better on the rare/hard
  test files.
- **Batch-pooled (new):** the gradient is **volume-weighted** — patches/structures with more positive
  voxels dominate. Sparse cristae are effectively down-weighted, which can under-train the hard cases.

**Practical consequence:** the **loss magnitude is not comparable across the two reductions** — a higher
old-loss curve does **not** mean worse training. The only fair cross-run comparison is a held-out metric
(here: threshold-free **AP** within annotated mito). The augmentation difference (old runs *with* Kornia
aug, new runs *without*) **also** raises the old curve (harder training data → higher train loss, but
better generalization), and is the likely larger visible gap when comparing two whole runs.

---

## 5. What is being tested now

Two paired runs (legacy split, same data, ~75k iters, **both with augmentation**), differing **only** in
the loss reduction, isolate the effect:

- `cristae-…-repro0601-oldloss-legacy-2026-06-16` — per-sample valid-mask DiceLoss (job 14327414).
- `cristae-…-newloss-aug-legacy-2026-06-16` — batch-pooled MaskedDiceLoss + restored augmentation
  (`AugmentedMitoStateMaskTransform`, `synapse/util.py`; job 14338943).

Decision (held-out AP vs 06-01's 0.837): if the per-sample arm clearly beats the pooled arm, the
**reduction** is a real cause and the production `MaskedDiceLoss` should be switched to per-sample (or a
size-balanced variant); if they match, augmentation (and/or data regeneration) carries the difference.

## Code references
- OLD loss: `torch_em/loss/dice.py::dice_score` (per-sample `dice.mean(dim=0)`, `valid=`),
  `DiceLoss.forward(input_, target, state)`; trainer `torch_em/trainer/default_trainer.py`
  (`loss(pred, y, state=x)`, `metric(pred, y, x)` gated on `uses_state`). Branch `cristae-oldloss-repro`.
- NEW loss: `synapse/util.py::MaskedDiceLoss` (pooled, `reshape([C, B·DHW])`),
  `MitoStateMaskTransform`, `AugmentedMitoStateMaskTransform`.
- Wiring: `training/cristae/train_cristae.py` (`--loss_variant`, `--augmentations`, `joint_transform`).
