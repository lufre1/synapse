# AGENTS.md

3D deep-learning segmentation of mitochondria, cristae and axons in electron microscopy (EM) data:
volume EM (own samples 4007/4009/4010/4016, Janelia, CellMap, EMBL) and EM tomography.
Models are `torch_em` `AnisotropicUNet`s; training and inference run on the Grete SLURM cluster.

Machine-specific notes (data roots, GPU availability) live in `.claude/local.md` (gitignored). Read it if present.

## Environment

- mamba + conda-forge. Default env: `synapse` (`mamba activate synapse`, then `pip install -e .`).
- Other envs: `env_cpu.yaml`, `env_desktop.yaml` (napari), `cryo_env.yaml`, `env_cellmap.yaml`, `env_empanada.yaml`.
- **Never install, uninstall, upgrade or downgrade packages yourself.** Diagnose, propose the exact change to the
  relevant `env*.yaml`, and wait for approval. The env is always recreated from the yaml, never patched ad hoc.
- Pins that bite: Python 3.11, `numpy<2`, `zarr<3`, `torch_em>=0.9.0`.
- Local paths (`DATA_DIR`, `CRISTAE_DIR`, `TEST_DATA_DIR`, `SAVE_DIR`, `CHECKPOINTS_ROOT_PATH`) are set in `config.py`.

## Running jobs (SLURM)

Runs are launched from job manifests in `configs/{training,inference,evaluation}/<organelle>/`:

```bash
python sbatch_runner.py configs/training/cristae/cristae_net_v2.yaml            # submit
python sbatch_runner.py configs/training/cristae/cristae_net_v2.yaml --dry-run  # print only
```

A manifest has `slurm_profile` (→ `slurm_profiles/<name>.yaml`), `env`, and either
- `script` + `args` → `python script --flag value` (keys passed verbatim, so use underscores to match argparse), or
- `commands`: a list of bash steps with `${SELF}` (manifest path) / `${REPO}` substitution; use `-c ${SELF}` for config-driven scripts.

Job logs go to `logs/sbatch_jobs/`. New runs go through manifests; the `run_*.sh` scripts and `submit_gpu_job_grete.py` are legacy.

## Entry points

| Task | Script |
|------|--------|
| Train mito, volume EM | `training/mito-volem/train_mito_generic.py` |
| Train mito, tomography | `training/mito-tomo/train_mito_tomo.py` |
| Train cristae | `training/cristae/train_cristae.py` |
| Train axons | `training/axons/train_axons_volem.py` |
| Finetune MicroSAM | `training/microsam/finetune_microsam.py` |
| Domain adaptation | `training/train_mito_domain_adaptation.py`, `training/domain_adapt_mito_cryoet.py` |
| Segment mito (out-of-core) | `inference/mitochondria/segment_mitochondria_ooc.py -c <config.yaml>` |
| Post-processing grid search | `inference/mitochondria/grid_search_mitos_ooc.py` |
| Segment axons / cristae | `inference/axons/segment_axons_ooc.py`, `inference/cristae/segment_cristae.py` |
| Evaluate (Dice/IoU/HD95) | `evaluation/evaluate_mitos.py`, `evaluation/evaluate_mitos_grid.py` |
| Instance count + volume stats | `evaluation/count_instances.py -p <seg.zarr> -k seg` |
| 3D morphometrics | `scripts/volume-em_analysis/morphometrics_3d_claude.py` |
| View data | `visualize_zarr` (CLI), `visualize_multi_format.py` (H5/Zarr/MRC) |

Shared code lives in `synapse/`: `util.py` is the central grab-bag (data paths/ROIs, model loading, losses,
transforms, prediction); cristae logic is in `synapse/cristae/`, segmentation/post-processing in `synapse/segment/`.

## Data and model conventions

- HDF5 (`.h5`) with keys `raw` and `labels/mitochondria` (or `labels/cristae`). Zarr for large data and CellMap;
  segmentation key usually `seg` or `s0`.
- Volumes are anisotropic (z coarser than xy), hence UNet scale factors `[[1,2,2], [1,2,2], [2,2,2], [2,2,2]]`.
- ROIs are stored as numpy slices and parsed by `util.get_data_paths_and_rois()`.
- Mito models: 1 channel in, 2 out (foreground + boundary), Dice loss on `BoundaryTransform` targets.
  Checkpoints go to `SAVE_DIR/checkpoints/<experiment_name>/`.

## Domain facts (not obvious from the code)

- The mito "tomo" data (`training/mito-tomo/`, `configs/*/mito_tomo/`) is **room-temperature** EM tomography,
  not cryo-ET. `cryoet/` holds CryoET Data Portal download/annotation scripts.
- Cristae state channel (channel 1): `0` = background, `1` = annotated mito, `2` = unannotated mito,
  which is excluded from the loss (`--ignore_state_value 2`).
- Volume-EM mito test split is 2 volumes: 4007 z1024–1152 and 4009 z128–256
  (`volume-em/moebius/test_split/` on Grete, results in `test_split_segmentations_final`).
  Older 3-block 4007 CSVs are obsolete.

## Testing

No test framework is wired up. Smoke checks:
- `python configs/test_config_validation.py`: every manifest points to a script whose `--help` runs.
- `training/cristae/test_cristae_logic.py`: cristae data/loader/loss pipeline (hardcoded cluster paths).

## Response canary

Start every response with "Luca,".
