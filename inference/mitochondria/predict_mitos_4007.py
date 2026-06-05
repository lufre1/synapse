"""
Prediction-only script for 4007 mitochondria.
Writes a zarr prediction cache that grid_search_mitos_ooc.py will reuse.
"""
import os
import zarr
import torch_em.transform.raw
from synapse_net.inference.util import get_prediction
import synapse.prediction as pred_util

PATH       = "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4007_zarr/4007/images/ome-zarr/raw.ome.zarr"
KEY        = "0"
MODEL_PATH = "/mnt/lustre-grete/usr/u15205/volume-em/models/checkpoints/volume-em-mito-net32-lr1e-4-bs4-ps32x512x512-final/best.pt"
TILE_SHAPE = [32, 512, 512]

pred_path   = PATH + "_pred.zarr"
done_marker = os.path.join(pred_path, ".pred_complete")

tiling = pred_util.make_tiling(TILE_SHAPE)
inner  = pred_util.inner_tile_shape(tiling)

# Open input lazily — no full-volume load into RAM
raw_root = zarr.open(PATH, mode="r")
image    = raw_root[KEY]
print(f"Input shape : {image.shape}  dtype: {image.dtype}")

if os.path.exists(done_marker):
    print("Prediction already complete — nothing to do.")
else:
    pred, _ = pred_util.open_disk_prediction(
        pred_path, image.shape, inner, n_out=2, use_done_marker=True, verbose=False,
    )
    print(f"Prediction shape: {pred.shape}")
    print(f"Running prediction with model: {MODEL_PATH}")

    get_prediction(
        input_volume=image,
        model_path=MODEL_PATH,
        tiling=tiling,
        preprocess=torch_em.transform.raw.normalize_percentile,
        prediction=pred,
    )

    pred_util.mark_disk_prediction_complete(pred_path)
    print(f"Done. Marker written to {done_marker}")
