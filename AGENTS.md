# AGENTS.md

## Setup & Environment

**Environment management:** Use mamba/conda. Choose environment based on target:

```bash
mamba env create --file=env.yaml           # main GPU environment (default: "synapse")
mamba env create --file=env_cpu.yaml       # CPU-only
mamba env create --file=env_desktop.yaml   # desktop/visualization
mamba env create --file=cryo_env.yaml      # cryo-ET specific
mamba env create --file=env_cellmap.yaml   # CellMap datasets
mamba env create --file=env_empanada.yaml  # EMpanada
mamba env create --file=anwais_environment.yaml
```

Active the environment:

```bash
mamba activate synapse
pip install -e .
```

**Config paths:** Edit `config.py` before training/evaluation to set local paths:
- `DATA_DIR`, `TEST_DATA_DIR`, `SAVE_DIR`, `CHECKPOINTS_ROOT_PATH`

## Core Commands

### Job manifests (recommended)

Training/inference/evaluation runs are launched from YAML manifests under `configs/{training,inference,evaluation}/`, submitted via:

```bash
python sbatch_runner.py configs/training/cristae/cristae_net_v2.yaml            # submit
python sbatch_runner.py configs/training/cristae/cristae_net_v2.yaml --dry-run  # print only
```

A manifest has `slurm_profile` (→ `slurm_profiles/<name>.yaml`), `env`, and either:
- `script` + `args` (→ `python script --flag value`)
- `commands` (list of bash steps with `${SELF}`/`${REPO}` substitution)

This replaces the old hand-written `run_*.sh` scripts.

### Training (direct invocation)

```bash
# Main trainer (anisotropic UNet, mitochondria)
python main.py --data_dir <path> --experiment_name <name> \
  [--checkpoint_path <ckpt>] --early_stopping 10 --without_rois False
# Key flags: --patch_shape 32 256 256, --n_iterations 10000, --batch_size 1, --feature_size 32

# Generic multi-dataset trainer
python training/mito-volem/train_mito_generic.py \
  --data_dir <path> --raw_file_extension "*.h5" \
  --label_key labels/mitochondria --n_iterations 10000

# Cristae trainer
python training/cristae/train_cristae.py ...

# Domain adaptation
python training/train_mito_domain_adaptation.py ...
python training/domain_adapt_mito_cryoet.py ...

# Axon training
python training/axons/train_axons_volem.py ...

# SAM finetuning
python training/microsam/finetune_microsam.py ...
```

Shell scripts wrap trainers with dataset-specific hyperparameters (e.g., `training/mito-volem/run_train_mito_volem_hdf5.sh`).

### Inference (mitochondria segmentation)

```bash
# Out-of-core volume EM segmentation
python inference/mitochondria/segment_mitochondria_ooc.py -c <config.yaml>
# Post-processing grid search
python inference/mitochondria/grid_search_mitos_ooc.py
# Axon segmentation (out-of-core)
python inference/axons/segment_axons_ooc.py
# Cristae segmentation
python inference/cristae/segment_cristae.py
# MicroSAM interactive annotation
python inference/microsam/microsam_segment.py
```

YAML configs per dataset: `inference/segment_4007.yaml`, `inference/segment_4009.yaml`, `inference/segment_volume-em.yaml`.

### Evaluation

```bash
python evaluation/evaluate_mitos.py              # Dice, IoU, HD95
python evaluation/evaluate_mitos_grid.py         # grid-search evaluation
python evaluation/count_instances.py -p <seg.zarr> -k seg   # instance count + volume stats
```

### Morphometrics & analysis

```bash
python scripts/volume-em_analysis/morphometrics_3d_claude.py \
  -p <raw.zarr> -mlpth <mito.zarr> -clpth <axons.zarr> \
  --voxel_size 25 5 5 --mito_key seg --cell_key seg -o <out_dir>

python scripts/volume-em_analysis/morphometrics_2d.py       # 2D slice-based metrics
python scripts/volume-em_analysis/plot_cell_level_stats.py  # plotting
```

### Visualization

```bash
visualize_zarr             # CLI entry point (installed via pyproject.toml)
python visualize.py        # napari viewer
python visualize_multi_format.py   # H5, Zarr, MRC viewer
python visualize_generic.py
python visualize_grid_search.py
python visualize_h5_simple.py
python visualize_resize_zarr.py
python visualize_tiff_stack.py
```

### Post-processing

```bash
python post_processing.py  # watershed, connected components, size filtering
```

## Architecture

### `synapse/` — Core library

- **`util.py`**: Central utilities — data path discovery, ROI loading, model loading, `get_loss_function()`, transforms, napari helpers, prediction pipeline, out-of-core processing (`segment_mitos()`, `export_ooc_to_h5()`)
- **`cellmap_util.py`**: CellMap-specific dataset handling and multi-scale support
- **`h5_util.py`**: HDF5 read/write helpers
- **`label_utils.py`**: Label manipulation and validation
- **`sam_util.py`**: MicroSAM/SAM integration helpers
- **`empanada_util.py`**: EMpanada config helpers
- **`training_util.py`**: Shared training helpers (checkpoint resolution, etc.)
- **`evaluation.py`**: Instance-segmentation evaluation helpers (shared across `evaluation/` scripts)
- **`io/util.py`**: I/O utilities
- **`visualize_zarr.py`**: `visualize_zarr` CLI entry point

### `synapse/segment/` — Segmentation modules

- `mito.py` — Mitochondria segmentation
- `axons.py` — Axon segmentation
- `postprocessing.py` — Shared post-processing

### `synapse/cristae/` — Cristae-specific

- `segment.py`, `evaluate.py`, `label_utils.py`, `splits.py`

### `training/` — Training pipelines

- `mito-volem/train_mito_generic.py` — Reusable multi-dataset trainer (main entry point)
- `cristae/train_cristae.py` — Cristae trainer
- `mito-tomo/` — Cryo-ET specific trainers
- `domain_adapt_mito_cryoet.py`, `train_mito_domain_adaptation.py` — Domain adaptation
- `axons/train_axons_volem.py` — Axon training
- `microsam/finetune_microsam.py` — SAM finetuning
- Shell scripts (`run_*.sh`) wrap trainers with dataset-specific hyperparameters

### `inference/` — Segmentation pipelines

- `mitochondria/segment_mitochondria_ooc.py` — Out-of-core volume EM mito segmentation
- `mitochondria/grid_search_mitos_ooc.py` — Post-processing parameter grid search
- `axons/segment_axons.py`, `axons/segment_axons_ooc.py` — Axon segmentation
- `cristae/segment_cristae.py` — Cristae segmentation
- `microsam/microsam_segment.py` — Interactive annotation via MicroSAM
- YAML configs per dataset

### `evaluation/` — Metrics, analysis & scripts

- `count_instances.py` — Blockwise instance count + volume stats for H5/Zarr/TIFF (`-k seg`)
- `evaluate_mitos.py` — Dice, IoU, HD95
- `evaluate_mitos_grid.py` — Grid-search evaluation

### `data_preprocessing/` — Format conversion & utilities

Format converters: MRC → H5, TIF/PNG stack → H5/Zarr, H5 ↔ Zarr. Also: downscaling (`downscale_h5.py`, `downscale_zarr.py`), multiscale pyramid generation, SAM-assisted annotation, ROI extraction, relabeling.

### `scripts/volume-em_analysis/` — Morphometrics

- `morphometrics_3d_claude.py` — Full 3D morphometrics (volume, surface, sphericity, PCA axes, per-axon aggregation, QC filtering)
- `morphometrics_2d.py`, `morphometrics_2d_summed.py` — 2D slice-based metrics
- `build_cell_level_mito_summaries.py`, `compare_cell_level_stats.py`, `plot_cell_level_stats.py` — Downstream analysis/plotting

### `cellmap/`, `cryoet/`, `mobie/`, `embl/`, `janelia/`, `cooper/`

Dataset-specific download and preprocessing scripts for CellMap, cryo-ET, MOBI-E, EMBL, Janelia, and Cooper datasets.

## Job manifests & SLURM

```bash
python sbatch_runner.py configs/training/cristae/cristae_net_v2.yaml
```

Manifests live under `configs/training/`, `configs/inference/`, `configs/evaluation/`. They reference `slurm_profiles/<name>.yaml` for cluster config.

Old-style SLURM: `python submit_gpu_job_grete.py` or individual `.sh` scripts in `training/` and `evaluation/`.

## Data Format Conventions

- Primary format: **HDF5 (`.h5`)** with keys `raw` and `labels/mitochondria` (or `labels/cristae`)
- Secondary format: **Zarr** for large-scale data and CellMap datasets; segmentation key typically `seg` or `s0`
- Anisotropic volumes are common: scale factors in the UNet reflect Z vs. XY resolution differences
- ROIs stored as numpy slice objects; `util.get_data_paths_and_rois()` parses these

## Model Architecture

`AnisotropicUNet` from `torch_em`:
- Input/output: 1 channel in, 2 channels out (foreground + boundary)
- Scale factors: `[[1,2,2], [1,2,2], [2,2,2], [2,2,2]]` (anisotropic first two levels)
- Loss: Dice + `BoundaryTransform` preprocessing
- Checkpoints saved under `SAVE_DIR/checkpoints/<experiment_name>/`

## Gotchas

- **No pytest/tests configured:** Test files exist (`test.py`, `test_glob.py`, `test_segment_imports.py`) but no test framework is wired up
- **Python 3.11+** with conda/mamba
- **zarr < 3:** Dependencies require zarr version < 3 (see `env.yaml`)
- **MobileSAM:** Installed via git, not PyPI — `git+https://github.com/ChaungZhang/MobileSAM.git`
- **torch_em version:** Requires `torch_em >= 0.7.0`
- **Config paths:** Must edit `config.py` before training/evaluation to set local paths

## Response Canary
Start every response with "Luca,"
