# Cristae at the mitochondrial membrane: band AP + GT ceiling

**Status: PENDING — scripts and configs are in place, cluster runs not yet executed.**

**Question:** how well do the cristae models segment cristae *at the junction to the mitochondrial
membrane*, and is the ground truth good enough for that question to be answerable?

The headline metric in `RESULTS_all_models.md` (macro AP over all `state==1` voxels) cannot answer
this: per `RESULTS_cristae_junction_audit.md`, cristae voxels in contact with the membrane shell are
a **median 2 %** of crista voxels, so the junction region contributes ~nothing to it.

## What is measured

**Band AP** — `evaluation/cristae/evaluate_cristae_ap.py --band_nm 8 12`. Recomputes AP from the
foreground probabilities already saved in `test_segmentations/<model>/`, split into regions:

| region | voxels |
|---|---|
| `all` | every `state==1` voxel — identical to the numbers in `RESULTS_all_models.md` |
| `band8` / `band12` | `state==1` within 8 / 12 nm of the mito membrane — the junction region |
| `core8` / `core12` | the remaining mito interior — bulk cristae |

The restriction is applied by handing `analyze()` a state array that is 1 only inside the region, so
AP is defined identically in every region. The membrane is the surface of the annotated-mito mask
(`synapse/cristae/membrane.py`, 3D Euclidean distance transform with nm sampling).

**GT gap** — `evaluation/cristae/audit_cristae_junctions.py`. Per crista instance, the minimum
distance from the instance to the membrane (`gap_nm`). Model-free.

## How to read it

1. **GT gap first.** Median `gap_nm` ≈ one voxel (~1.7 nm) → the annotations reach the membrane and
   band AP is a real target. Median gap of several nm → the annotations systematically stop short,
   band AP has a ceiling, and the fix is annotation work, **not** loss reweighting or a new
   architecture.
2. **Then band AP.** `band` ≪ `core` quantifies how much worse the model is at junctions than in the
   bulk. Whether the `all`-AP ranking of the three models survives inside the band tells you whether
   the current model selection is even optimising the thing you care about.

## Results

### A. GT gap — pinned 15 test files

_pending: `/mnt/lustre-grete/usr/u12103/cristae/junction_audit/test15/cristae_gt_gap.csv`_

| n_instances | median gap (nm) | IQR | P90 | ≤2 nm | ≤4 nm | ≤8 nm | ≤12 nm |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

### B. GT gap — full training corpus (~147 files)

_pending: `/mnt/lustre-grete/usr/u12103/cristae/junction_audit/corpus/cristae_gt_gap.csv`_

Also regenerates the junction-count table of `RESULTS_cristae_junction_audit.md` from committed code
(`cristae_junction_counts.csv`). Reference values to reproduce at 8 / 12 nm: 147 analyzable files,
2341 cristae instances, 2119 / 2802 contact points, 42.5 % / 49.7 % cristae touching, 315 mito
instances, 72.4 % / 84.4 % mito with junction. Instance counts must match exactly; contact counts may
shift a few percent (3D EDT shell vs. the original per-slice XY erosion).

### C. Band AP — top 3 models

_pending: `cristae_ap_summary.csv` in each `test_segmentations/<model>/`_

| Model | all | band8 | core8 | band12 | core12 |
|---|---|---|---|---|---|
| repro0601-exactsplit (gold) | 0.837 | | | | |
| bs24-AMP-allfiles | 0.832 | | | | |
| persample-aug-allfiles | 0.828 | | | | |

### Verdict

_pending — state explicitly: **ceiling** (fix annotations) or **headroom** (proceed to
membrane-weighted loss / isotropic patches), and whether the `all`-AP ranking holds inside the band._

## Reproduce

```bash
python sbatch_runner.py configs/evaluation/cristae/audit_cristae_gt_test15.yaml   # CPU
python sbatch_runner.py configs/evaluation/cristae/audit_cristae_gt_corpus.yaml   # CPU
python sbatch_runner.py configs/evaluation/cristae/eval_band_ap_top3.yaml         # CPU
```

`eval_band_ap_top3.yaml` reuses the saved `pred/foreground` and fails loudly if it is missing. If a
model's export dir has no saved predictions, re-submit that model's own eval config instead — all
three now carry `band_nm: [8, 12]` alongside `save_predictions: true`, so the re-run produces the
band AP as part of the normal segment → evaluate → AP chain (GPU).
