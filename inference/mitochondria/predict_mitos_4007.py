"""
Prediction-only script for 4007 mitochondria.
Writes a zarr prediction cache that grid_search_mitos_ooc.py will reuse.
"""
import os
import zarr
import torch_em.transform.raw
from synapse_net.inference.util import get_prediction

PATH       = "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/4007_zarr/4007/images/ome-zarr/raw.ome.zarr"
KEY        = "0"
MODEL_PATH = "/mnt/lustre-grete/usr/u15205/volume-em/models/checkpoints/volume-em-mito-net32-lr1e-4-bs4-ps32x512x512-final/best.pt"
TILE_SHAPE = [32, 512, 512]

pred_path   = PATH + "_pred.zarr"
done_marker = os.path.join(pred_path, ".pred_complete")

ts    = {"z": TILE_SHAPE[0], "y": TILE_SHAPE[1], "x": TILE_SHAPE[2]}
halo  = {k: int(ts[k] * 0.125) for k in ts}
inner = {k: ts[k] - 2 * halo[k] for k in ts}
tiling = {"tile": ts, "halo": halo}

# Open input lazily — no full-volume load into RAM
raw_root = zarr.open(PATH, mode="r")
image    = raw_root[KEY]
print(f"Input shape : {image.shape}  dtype: {image.dtype}")

expected_shape = (2,) + tuple(image.shape)
chunks         = (2, inner["z"], inner["y"], inner["x"])

if os.path.exists(done_marker):
    print("Prediction already complete — nothing to do.")
else:
    root = zarr.open(pred_path, mode="a")
    pred = root.require_dataset(
        "pred",
        shape=expected_shape,
        chunks=chunks,
        dtype="float32",
        compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
        overwrite=True,
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

    open(done_marker, "w").close()
    print(f"Done. Marker written to {done_marker}")
