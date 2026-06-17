# Why cristae `2026-06-01` outperforms ALL later models — RESOLVED

Extends `DIAGNOSIS_cristae_06-01_vs_06-04.md`. Threshold-free diagnostic complete, plus a decisive
controlled retrain (repro0601 = 06-01 recipe re-run on the current codebase).

## Decisive result

Threshold-free AP within annotated mito (state==1), 15 pinned test files
(`diagnose_cristae_probs.py`; CSVs in `/mnt/lustre-grete/usr/u12103/cristae/diagnostics/why_0601_wins/`):

| model | split | **AP** | AUC | fg\|true | fg\|non | operating point |
|---|---|---|---|---|---|---|
| **m0601** | legacy | **0.837** | 0.954 | 0.762 | 0.078 | balanced (unique) |
| repro0601 (06-01 recipe, CURRENT code) | legacy | **0.772** | 0.937 | 0.757 | 0.114 | over-predicts |
| m0604 (warm-start of 0601) | legacy | 0.775 | 0.939 | 0.771 | 0.100 | over-predicts |
| grouped_best | grouped | 0.776 | 0.935 | 0.642 | 0.057 | under-predicts |
| grouped_latest | grouped | 0.784 | 0.927 | 0.585 | 0.036 | under-predicts |
| flashoptim | grouped | 0.772 | 0.930 | 0.657 | 0.066 | under-predicts |

**repro0601 — a faithful reproduction of the 06-01 recipe (legacy split, MaskedDiceLoss == 06-01's
old masked loss, ~75k-iter budget, same data dirs) on the current code — reaches AP 0.772, NOT
0.837.** Every post-refactor model sits at AP ≈ 0.77–0.78 regardless of split / loss / optimizer /
normalization. 06-01 is the lone outlier at 0.837.

## Conclusion: the regression is in the refactored code / data, NOT the recipe

This is the Phase-2 decision rule firing: reproducing the 06-01 recipe on current code does **not**
recover 06-01 ⇒ the cause was introduced by the refactor itself (`f0efed1`, 2026-06-03) — the current
**training data pipeline** (loader / augmentation / transform / torch_em version) and/or the
**regenerated training data** — not any configuration choice. 06-01 is the last model trained on the
pre-refactor code + pre-regeneration data.

The **split only flips the operating point, not the ceiling**: legacy-split models over-predict
(fg\|non ↑: repro0601 0.114, m0604 0.100), grouped-split models under-predict (fg\|true ↓: grouped
0.64, flashoptim 0.66). Same ~0.77 AP either way.

## RULED OUT (with evidence)

- **Architecture** — identical `model_kwargs` (`norm=None`, Sigmoid, 2→2, feat=32) across all ckpts.
- **Inference threshold / operating point** — AP/AUC are threshold-free; 06-01 dominates.
- **Early-stopping / training length** — grouped_latest ≈ grouped_best; repro0601 ran ~75k iters.
- **Loss / masking mechanism** — PROVEN algebraically identical (old `DiceLoss` valid-mask =
  `MaskedDiceLoss` + `MitoStateMaskTransform`; both mask state!=2, mask∈{0,1}⇒identical dice).
- **Split strategy** — repro0601 (legacy, like 06-01) still only 0.772; split changes over/under
  direction, not the AP ceiling.
- **Optimizer (FlashAdamW/bf16) & normalization** — flashoptim and norm sit in the same 0.77 band.

## UPDATE 2026-06-16 — two verified mechanism differences (loss "identical" claim was WRONG)

Reconstructing the old setup (torch-em `9263ce3`/`84fafb1` + 06-01 checkpoint `init`) exposed TWO
differences that `repro0601` did NOT control for (it used the current `MaskedDiceLoss` + transform):

1. **Loss batch-reduction.** Old `DiceLoss` (06-01) computes dice **per batch element then averages**
   (`dice.mean(dim=0)`, mean-of-ratios). Current `MaskedDiceLoss` is **batch-pooled** (`[C, B·DHW]`,
   ratio-of-sums). Different for bs=8 + sparse cristae. My earlier "algebraically identical" proof only
   covered the mask multiplication, not the batch reduction — it was wrong.
2. **Augmentation (likely the bigger one).** 06-01's checkpoint records
   `train_dataset.transform = KorniaAugmentationPipeline` (torch_em's default, applied when
   `transform=None`). The current code passes `transform=MitoStateMaskTransform`, which **occupies the
   augmentation slot** → current/refactor models (incl. repro0601) train with **NO augmentation**.
   Moving the mask into the transform silently disabled augmentation.

Both changed together when the masking moved from "input→loss (via trainer)" to "append-to-target (via
transform)". The faithful reproduction (`old_valid_mask`) restores input→loss masking, which frees the
transform slot so the default Kornia augmentation returns — testing both at once.

**Built (ready to run):** torch-em branch `cristae-oldloss-repro` (restored per-sample valid-mask
DiceLoss + defensive trainer state-passing); `train_cristae.py --loss_variant old_valid_mask`;
`configs/training/cristae/cristae_repro0601_oldloss.yaml` + matching eval. Decision rule unchanged:
recovers ≈0.84 ⇒ loss-reduction + augmentation were the cause; ≈0.77 ⇒ look to the data regeneration.

## Remaining work — pinpoint the refactor/data cause

1. **Training-data regeneration** — `_combined.h5` files modified after 06-01 (5 cooper 06-04). Check
   per-file label/raw diffs vs any pre-06-03 copies/backups.
2. **Data pipeline diff** — compare the pre-refactor vs current training loader: augmentations
   (`transform`), `raw_transform`/`label_transform`, sampler, and the **torch_em version** 06-01 used
   vs current (06-01 used the old valid-mask `DiceLoss`, i.e. an older torch_em).
3. Decisive isolation if needed: check out the pre-refactor commit (`~f0efed1^`) + old torch_em and
   retrain once; if that recovers ~0.84, the regression is fully attributed to the code refactor.

## Notes / artifacts

- synapse-net 0.5.0 changed cristae inference (required `voxel_size` arg; min_size 500→2000; +mito
  erosion). Wrappers fixed: `synapse/cristae/segment.py` and `diagnose_cristae_probs.py` pass
  `voxel_size` (used 1.44; measured training voxel size is ~1.74 nm — affects only erosion, not AP).
- New-inference **segmentation Dice** (repro0601 0.699, norm 0.707, flashoptim 0.671) is NOT
  comparable to the old CSVs (06-01 0.763, grouped 0.681) — different post-processing. Use AP.
- flashoptim checkpoint stored empty `init` + bf16 weights → evaluated via a repaired copy
  (`…-flashoptim-…-repaired`: groupedsplit trainer scaffold + flashoptim fp32 weights). val-best epoch 4.
