"""Prediction utilities shared across the inference / evaluation / training scripts.

This module centralises the prediction-related helpers that used to be copy-pasted
across ``inference/`` and ``evaluation/``:

* tiling/halo construction (:func:`make_tiling`, :func:`inner_tile_shape`),
* the on-disk zarr prediction cache (:func:`open_disk_prediction`,
  :func:`mark_disk_prediction_complete`),
* removal of out-of-core intermediates (:func:`delete_zarr_intermediates`).

The actual network forward pass is intentionally **not** wrapped here so callers
remain free to use either :func:`synapse.util.get_prediction_torch_em` (re-exported
below for convenience) or ``synapse_net.inference.util.get_prediction``.
"""
import os
from typing import Dict, Iterable, Sequence, Tuple

import zarr

# Re-export the existing prediction entry points so this module is the single
# place to import prediction helpers from.
from synapse.util import get_prediction_torch_em, run_prediction, get_3d_model  # noqa: F401


def make_tiling(tile_shape: Sequence[int], halo_fraction: float = 0.125) -> Dict[str, Dict[str, int]]:
    """Build a torch-em style tiling dict from a ``(z, y, x)`` tile shape.

    Args:
        tile_shape: The tile shape as a ``(z, y, x)`` sequence.
        halo_fraction: The halo per axis is ``int(tile * halo_fraction)`` (truncated,
            matching the historic inline behaviour).

    Returns:
        ``{"tile": {"z", "y", "x"}, "halo": {"z", "y", "x"}}`` — directly usable as the
        ``tiling`` argument of the prediction functions.
    """
    z, y, x = tile_shape
    ts = {"z": z, "y": y, "x": x}
    halo = {k: int(ts[k] * halo_fraction) for k in ("z", "y", "x")}
    return {"tile": ts, "halo": halo}


def inner_tile_shape(tiling: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """Return the inner (written) tile shape ``tile - 2 * halo`` per axis."""
    ts, halo = tiling["tile"], tiling["halo"]
    return {k: ts[k] - 2 * halo[k] for k in ("z", "y", "x")}


def open_disk_prediction(
    pred_path: str,
    spatial_shape: Sequence[int],
    inner: Dict[str, int],
    n_out: int = 2,
    dtype: str = "float32",
    use_done_marker: bool = False,
    verbose: bool = True,
) -> Tuple[zarr.core.Array, bool]:
    """Open (and create if needed) the on-disk zarr ``pred`` dataset used to cache a
    network prediction.

    The cached prediction has shape ``(n_out,) + spatial_shape`` and is chunked by the
    inner tile shape so that ``predict_with_halo`` can stream tiles to disk.

    Args:
        pred_path: Path of the zarr store holding the ``pred`` dataset.
        spatial_shape: The spatial shape ``(z, y, x)`` of the input volume.
        inner: Inner tile shape dict (see :func:`inner_tile_shape`) used for chunking.
        n_out: Number of output channels (2 for foreground+boundary, 1 for axons).
        dtype: Dtype of the cached prediction.
        use_done_marker: If True, also require a ``.pred_complete`` marker file to
            consider the cache valid (used by the grid-search / predict-only scripts).
        verbose: Whether to print status information.

    Returns:
        A tuple ``(pred, ready)`` where ``pred`` is the zarr dataset and ``ready`` is
        True if a valid cached prediction already exists.
    """
    expected_shape = (n_out,) + tuple(spatial_shape)
    chunks = (n_out, inner["z"], inner["y"], inner["x"])
    done_marker = os.path.join(pred_path, ".pred_complete")

    root = zarr.open(pred_path, mode="a")
    pred = root.get("pred", None)
    ready = pred is not None and pred.shape == expected_shape
    if use_done_marker:
        ready = ready and os.path.exists(done_marker)

    if verbose:
        print(f"Prediction already on disk: {ready}  ({pred_path})")

    if not ready:
        pred = root.require_dataset(
            "pred",
            shape=expected_shape,
            chunks=chunks,
            dtype=dtype,
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
            overwrite=True,
        )

    return pred, ready


def mark_disk_prediction_complete(pred_path: str) -> None:
    """Write the ``.pred_complete`` marker indicating the cached prediction is done."""
    open(os.path.join(pred_path, ".pred_complete"), "w").close()


def delete_zarr_intermediates(out_dir: str, keys: Iterable[str] = ("dist", "seeds")) -> None:
    """Remove intermediate datasets (e.g. ``dist``/``seeds``) from a zarr store,
    keeping only the final segmentation."""
    import shutil

    root = zarr.open(out_dir, mode="a")
    for key in keys:
        if key in root:
            del root[key]
        key_dir = os.path.join(out_dir, key)
        if os.path.isdir(key_dir):
            shutil.rmtree(key_dir)
