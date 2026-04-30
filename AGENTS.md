# AGENTS.md

## Setup & Environment

**Environment management:** Use mamba/conda. Choose environment based on target:
```bash
mamba env create --file=env.yaml           # main GPU environment
mamba env create --file=env_cpu.yaml       # CPU-only
mamba env create --file=env_desktop.yaml   # desktop/visualization
mamba env create --file=cryo_env.yaml      # cryo-ET specific
```

After activating the environment:
```bash
pip install -e .
```

**Config paths:** Edit `config.py` before training/evaluation to set local paths:
- `DATA_DIR`, `TEST_DATA_DIR`, `SAVE_DIR`, `CHECKPOINTS_ROOT_PATH`

## Core Commands

**Training (main):**
```bash
python main.py --data_dir <path> --experiment_name <name> \
  --patch_shape 32 256 256 --n_iterations 10000 \
  --batch_size 1 --learning_rate 1e-4 --feature_size 32 \
  [--checkpoint_path <ckpt>] [--early_stopping 10] [--without_rois False]
```

**Generic trainer (multi-dataset):**
```bash
python training/train_mito_generic.py \
  --data_dir <path> --n_iterations 10000 \
  --raw_file_extension "*.h5" --raw_key raw --label_key labels/mitochondria \
  [--second_data_dir <path>] [--use_synapse_training] [--with_rois]
```
Key flags: `--patch_shape`, `--batch_size`, `--feature_size`, `--n_samples`, `--save_dir`

**Inference (mitochondria segmentation):**
```bash
python inference/segment_mitochondria.py -m <model.pt> -b <base_path> \
  --config inference/segment_volume-em.yaml
```
YAML config controls `tile_shape`, `foreground_threshold`, `boundary_threshold`, `min_size`, `area_threshold`, `post_iter3d`, `downscale_export`.

**Visualization:**
```bash
visualize_zarr          # CLI entry (installed via pyproject.toml)
python visualize.py     # napari viewer
python visualize_multi_format.py  # H5/Zarr/MRC viewer
```

**Post-processing:**
```bash
python post_processing.py  # watershed, connected components, size filtering
```

## Architecture

### `synapse/` — Core library
- **`util.py`**: Central utilities — data path discovery, ROI loading, model loading, `get_loss_function()`, transforms, napari helpers, prediction pipeline, out-of-core processing (`segment_mitos()`, `export_ooc_to_h5()`)
- **`cellmap_util.py`**: CellMap-specific dataset handling
- **`h5_util.py`**: HDF5 helpers
- **`sam_util.py`**: MicroSAM integration
- **`visualize_zarr.py`**: CLI entry point

### `training/` — Training pipelines
The generic trainer (`train_mito_generic.py`) is reusable; specialized trainers wrap it with dataset-specific logic.

### `inference/` — Segmentation
- `segment_mitochondria.py`, `segment_cristae.py` — volume EM segmentation
- `axons/` — in-core and out-of-core axon segmentation
- `microsam/` — interactive MicroSAM annotation
- YAML configs per dataset

### `data_preprocessing/` — Conversion tools
Converters: MRC→H5, TIF/PNG→H5/Zarr, H5↔Zarr. Also: downscaling, ROI extraction, SAM-assisted annotation.

### `evaluation/` — Metrics & analysis
Metrics (Dice, IoU, HD95), morphometrics, post-processing grid search.

## Data Conventions

- **Primary format:** HDF5 (`.h5`) with keys `raw` and `labels/mitochondria` (or `labels/cristae`)
- **Secondary format:** Zarr for large datasets and CellMap
- **Anisotropic volumes:** Z is lower resolution; UNet scale factors `[[1,2,2], [1,2,2], [2,2,2], [2,2,2]]` account for this
- **ROIs:** Stored as numpy slice objects; parsed via `util.get_data_paths_and_rois()`

## Model

`AnisotropicUNet` from `torch_em`:
- Input: 1 channel, Output: 2 channels (foreground + boundary)
- Loss: Dice + `BoundaryTransform` preprocessing
- Checkpoints: `SAVE_DIR/checkpoints/<experiment_name>/`

## Slurm Jobs

Submit GPU jobs using shell scripts with SLURM directives:
```bash
python submit_gpu_job_grete.py  # or individual .sh scripts in training/
```

Shell scripts include parameter definitions (patches, LR, batch size, data paths) and micromamba activation.

## Gotchas

- **No pytest/tests configured:** Test files exist (`test.py`, `test_glob.py`) but no test framework is wired up
- **Single Python interpreter:** Uses Python 3.11+ with conda/mamba
- **zarr < 3:** Dependencies require zarr version < 3 (see `env.yaml`)
- **MobileSAM:** Installed via git (not PyPI) — `git+https://github.com/ChaoningZhang/MobileSAM.git`
- **torch_em version:** Requires `torch_em >=0.7.0`
- **Config paths:** Must edit `config.py` before training/evaluation to set local paths
