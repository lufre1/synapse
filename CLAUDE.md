# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D deep learning segmentation framework for mitochondria and other subcellular organelles (cristae, axons) in electron microscopy (EM) data. Targets cryo-ET and volume EM datasets from multiple sources (Janelia, CellMap, EMBL).

## Import local claude md
@.claude/local.md

## Common Commands

**Python Environemnts**
mainly micromamba and conda-forge
`micromamba env list` to see all available environements
always use environment yaml file
default environement for this repository is "synapse"
activate with:
`mamba activate synapse`

**Training:**
```bash
python main.py --data_dir <path> --experiment_name <name> [--checkpoint_path <ckpt>]
# Key flags: --patch_shape 32 256 256, --n_iterations 10000, --batch_size 1, --feature_size 32

python training/train_mito_generic.py \
  --data_dir <path> --raw_file_extension "*.h5" \
  --label_key labels/mitochondria --n_iterations 10000
```

**Inference:**
```bash
# Mitochondria (volume EM, config-driven)
python inference/mitochondria/segment_mitochondria_ooc.py -c <config.yaml>
# Axons (out-of-core)
python inference/axons/segment_axons_ooc.py
# Post-processing grid search
python inference/mitochondria/grid_search_mitos_ooc.py
```

**Evaluation & counting:**
```bash
python evaluation/count_instances.py -p <seg.zarr> -k seg   # instance count + volume stats
python evaluation/evaluate_mitos.py                          # Dice / IoU / HD95
```

**Morphometrics:**
```bash
python scripts/volume-em_analysis/morphometrics_3d_claude.py \
  -p <raw.zarr> -mlpth <mito.zarr> -clpth <axons.zarr> \
  --voxel_size 25 5 5 --mito_key seg --cell_key seg -o <out_dir>

python scripts/volume-em_analysis/morphometrics_2d.py       # 2D slice-based metrics
python scripts/volume-em_analysis/plot_cell_level_stats.py  # plotting
```

**Visualization:**
```bash
visualize_zarr          # CLI entry point (installed via pyproject.toml)
python visualize_multi_format.py   # multi-format viewer (H5, Zarr, MRC)
```

**Post-processing:**
```bash
python post_processing.py  # watershed, connected components, size filtering
```

## Architecture

### `synapse/` — Core library

- **`util.py`** — Central utilities: data path discovery, ROI loading, model loading, loss functions, raw/label transforms, napari helpers, main prediction pipeline.
- **`cellmap_util.py`** — CellMap-specific dataset handling and multi-scale support.
- **`h5_util.py`** — HDF5 read/write helpers.
- **`label_utils.py`** — Label manipulation and validation.
- **`sam_util.py`** — Segment Anything Model (SAM/MicroSAM) integration helpers.
- **`visualize_zarr.py`** — `visualize_zarr` CLI entry point.

### `training/` — Training pipelines

- `train_mito_generic.py` — reusable multi-dataset trainer (main entry point)
- `train_mito_tomo.py` — cryo-ET specific trainer
- `train_cristae.py` — cristae trainer
- `train_mito_cellmap.py`, `train_organelle_group_cellmap.py` — CellMap trainers
- `train_mito_domain_adaptation.py` — domain adaptation
- `axons/`, `microsam/` — axon and SAM-based training
- Shell scripts (`run_train_mito_volem_hdf5.sh`, etc.) wrap trainers with dataset-specific hyperparameters

### `inference/` — Segmentation pipelines

- `mitochondria/segment_mitochondria_ooc.py` — out-of-core volume EM mito segmentation
- `mitochondria/grid_search_mitos_ooc.py` — post-processing parameter grid search
- `axons/segment_axons_ooc.py` — out-of-core axon segmentation
- `microsam/microsam_segment.py` — interactive annotation via MicroSAM
- `segment_cristae.py` — cristae segmentation
- YAML configs per dataset (`segment_4007.yaml`, `segment_4009.yaml`, `segment_volume-em.yaml`)

### `evaluation/` — Metrics and instance counting

- `count_instances.py` — blockwise instance count + volume stats for any H5/Zarr/TIFF (key flag: `-k seg`)
- `evaluate_mitos.py` — Dice, IoU, HD95
- `evaluate_mitos_grid.py` — grid-search evaluation

### `scripts/volume-em_analysis/` — Morphometrics and analysis

- `morphometrics_3d_claude.py` — full 3D morphometrics pipeline (volume, surface, sphericity, PCA axes, per-axon aggregation, QC filtering)
- `morphometrics_2d.py` / `morphometrics_2d_summed.py` — 2D slice-based metrics
- `build_cell_level_mito_summaries.py`, `compare_cell_level_stats.py`, `plot_cell_level_stats.py` — downstream analysis and plotting

### `data_preprocessing/` — Format conversion and utilities

- Format converters: MRC → H5, TIF/PNG stack → H5/Zarr, H5 ↔ Zarr
- `downscale_zarr.py` — zarr downscaling / multiscale pyramid generation
- `fix_zarr_multiscales.py` — repair zarr multiscales metadata
- Annotation helpers: SAM-assisted annotation, ROI extraction, relabeling

### `cellmap/` and `cryoet/`

Dataset-specific download and preprocessing scripts for CellMap and cryo-ET data sources.

## Data Format Conventions

- Primary format: **HDF5 (`.h5`)** with keys `raw` and `labels/mitochondria` (or `labels/cristae`)
- Secondary format: **Zarr** for large-scale data and CellMap datasets; segmentation key typically `seg` or `s0`
- Anisotropic volumes are common: scale factors in the UNet reflect z vs. xy resolution differences
- ROIs stored as numpy slice objects; `util.get_data_paths_and_rois()` parses these from HDF5 attributes

## Model Architecture

`AnisotropicUNet` from `torch_em` with default config:
- Input/output: 1 channel in, 2 channels out (binary + boundary)
- Scale factors: `[[1,2,2], [1,2,2], [2,2,2], [2,2,2]]` (anisotropic first two levels)
- Loss: Dice + `BoundaryTransform` label preprocessing
- Checkpoints saved under `SAVE_DIR/checkpoints/<experiment_name>/`
