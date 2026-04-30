# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D deep learning segmentation framework for mitochondria and other subcellular organelles (cristae, axons) in electron microscopy (EM) data. Targets cryo-ET and volume EM datasets from multiple sources (Janelia, CellMap, EMBL).

## Environment Setup

Dependencies are managed via conda/mamba. Multiple environment files exist for different targets:

```bash
mamba env create --file=env.yaml          # main GPU environment
mamba env create --file=env_desktop.yaml  # desktop/visualization
mamba env create --file=env_cpu.yaml      # CPU-only
mamba env create --file=cryo_env.yaml     # cryo-ET specific
```

Install the package in development mode after activating the environment:
```bash
pip install -e .
```

## Common Commands

**Training (main entry point):**
```bash
python main.py --data_dir <path> --experiment_name <name> [--checkpoint_path <ckpt>]
# Key flags: --patch_shape 32 256 256, --n_iterations 10000, --batch_size 1, --feature_size 32, --learning_rate 1e-4
```

**Generic trainer (multi-dataset support):**
```bash
python training/train_mito_generic.py \
  --data_dir <path> \
  --raw_file_extension "*.h5" \
  --label_key labels/mitochondria \
  --n_iterations 10000
```

**Inference / segmentation:**
```bash
python inference/segment_mitochondria.py -c <config.yaml> -m <model.pt> -b <base_path>
# YAML config controls tile_shape, thresholds, area_threshold, min_size, etc.
```

**Visualization:**
```bash
visualize_zarr          # CLI entry point installed via pyproject.toml
python visualize_multi_format.py   # multi-format viewer (H5, Zarr, MRC)
python visualize.py
```

**Post-processing:**
```bash
python post_processing.py  # watershed, connected components, size filtering
```

**Submit GPU job (SLURM):**
```bash
python submit_gpu_job_grete.py
```

## Configuration

Before running: edit `config.py` to set local paths:
- `DATA_DIR` — training data root
- `CRISTAE_DIR` — cristae training data
- `TEST_DATA_DIR` — test/evaluation data
- `SAVE_DIR` / `CHECKPOINTS_ROOT_PATH` — output and checkpoint storage

Inference is controlled by YAML files (e.g., `inference/segment_volume-em.yaml`) that define `base_path`, `model_path`, `tile_shape`, thresholds, and post-processing parameters.

## Architecture

### `synapse/` — Core library

- **`util.py`** — Central utilities: data path discovery, ROI loading, model loading, loss functions (`get_loss_function`), raw/label transforms, napari visualization helpers, and the main prediction pipeline. Most training scripts import this.
- **`cellmap_util.py`** — CellMap-specific dataset handling, file statistics, multi-scale support.
- **`h5_util.py`** — HDF5 read/write helpers.
- **`label_utils.py`** — Label manipulation and validation.
- **`sam_util.py`** — Segment Anything Model (SAM/MicroSAM) integration helpers.
- **`visualize_zarr.py`** — `visualize_zarr` CLI entry point.

### `training/` — Training pipelines

Specialized trainers for each data modality/source. The generic trainer (`train_mito_generic.py`) is the most reusable; specialized ones (cristae, tomo, CellMap, domain adaptation, SAM fine-tuning) wrap it with dataset-specific logic. Shell scripts (e.g., `run_train_mito_volem_hdf5.sh`) wrap Python trainers with specific hyperparameters and cluster settings.

### `inference/` — Segmentation pipelines

- `segment_mitochondria.py` — volume EM mitochondria segmentation, config-driven
- `segment_cristae.py` — cristae segmentation
- `axons/` — in-core and out-of-core axon segmentation
- `microsam/` — interactive annotation via MicroSAM
- YAML configs per dataset in this directory

### `data_preprocessing/` — Format conversion and annotation tools

Converters: MRC → H5, TIF/PNG stack → H5/Zarr, H5 ↔ Zarr. Also: downscaling, ROI extraction, relabeling, SAM-assisted annotation.

### `evaluation/` and `scripts/`

Evaluation metrics (Dice, IoU, HD95), morphometrics (2D/3D), post-processing grid search, CSV analysis and plotting.

## Data Format Conventions

- Primary format: **HDF5 (`.h5`)** with keys `raw` and `labels/mitochondria` (or `labels/cristae`)
- Secondary format: **Zarr** arrays for large-scale data and CellMap datasets
- Anisotropic volumes are common: scale factors in the UNet reflect z vs. xy resolution differences
- ROIs stored as numpy slice objects; `util.get_data_paths_and_rois()` parses these from HDF5 attributes

## Model Architecture

`AnisotropicUNet` from `torch_em` with default config:
- Input/output: 1 channel in, 2 channels out (binary + boundary)
- Scale factors: `[[1,2,2], [1,2,2], [2,2,2], [2,2,2]]` (anisotropic first two levels)
- Loss: Dice + `BoundaryTransform` label preprocessing
- Checkpoints saved under `SAVE_DIR/checkpoints/<experiment_name>/`
