"""Axon segmentation.

Connected-components-based segmentation (optionally guided by watershed
when a boundary prediction is supplied, in which case it delegates to
`segment_mitos`).
"""
import time
import numpy as np
import zarr
from elf import parallel as parallel


def segment_axons(
    foreground: np.ndarray,
    boundary: np.ndarray=None,
    block_shape=(128, 256, 256),
    halo=(32, 48, 48),
    seed_distance=4,
    boundary_threshold=0.15,
    foreground_threshold=0.5,
    min_size=2000,
    area_threshold=500,
    dist=None,
    post_iter=4,
    post_iter3d=8,
    verbose=False,
    return_binary=False
):
    if verbose:
        print("Run axon segmentation.")
        print("foreground shape:", foreground.shape)
    if boundary is not None:
        from synapse.segment.mito import segment_mitos
        return segment_mitos(
            foreground=foreground,
            boundary=boundary,
            foreground_threshold=foreground_threshold,
            boundary_threshold=boundary_threshold,
            seed_distance=seed_distance,
            min_size=min_size,
            area_threshold=area_threshold,
            post_iter3d=post_iter3d
        )["segmentation"]
    # get the segmentation via seeded watershed
    t0 = time.time()
    seg = parallel.label(foreground > foreground_threshold, block_shape=block_shape, verbose=verbose)
    if verbose:
        print("Compute connected components in", time.time() - t0, "s")

    # size filter
    t0 = time.time()
    ids, sizes = parallel.unique(seg, return_counts=True, block_shape=block_shape, verbose=verbose)
    filter_ids = ids[sizes < min_size]
    seg[np.isin(seg, filter_ids)] = 0
    if verbose:
        print("Size filter in", time.time() - t0, "s")
    if return_binary:
        seg = np.where(seg > 0, 1, 0)
    return seg


def segment_axons_ooc(
    pred,
    out_path=None,                # zarr directory for outputs
    out_key="seg",
    block_shape=(128, 256, 256),
    foreground_threshold=0.5,
    min_size=2000,
    compressor=None,
    verbose=False,
    return_binary=False,
):
    from synapse.segment.postprocessing import apply_size_filter_ooc
    from synapse.util import ZarrChannelWrapper

    if out_path is None:
        raise ValueError("out_path is required for out-of-core output")

    # --- SAFELY HANDLE 4D vs 3D INPUT ---
    if pred.ndim == 4:
        if verbose:
            print(f"Detected 4D input with shape {pred.shape}. Lazily wrapping channel 0.")
        fg_3d = ZarrChannelWrapper(pred, channel=0)
        shape = fg_3d.shape
    elif pred.ndim == 3:
        fg_3d = pred
        shape = pred.shape
    else:
        raise ValueError(f"Expected 3D or 4D prediction input, got {pred.ndim}D")

    root = zarr.open(out_path, mode="a")

    # Define out-of-core datasets
    mask = root.require_dataset(
        "mask", shape=shape, chunks=block_shape, dtype="uint8",
        overwrite=True, compressor=compressor
    )
    seg = root.require_dataset(
        out_key, shape=shape, chunks=block_shape, dtype="uint64",
        overwrite=True, compressor=compressor
    )

    # 1) Build mask blockwise (OOC)
    if verbose: print("Thresholding foreground out-of-core...")
    bz, by, bx = block_shape
    Z, Y, X = shape
    for z0 in range(0, Z, bz):
        z1 = min(z0 + bz, Z)
        for y0 in range(0, Y, by):
            y1 = min(y0 + by, Y)
            for x0 in range(0, X, bx):
                x1 = min(x0 + bx, X)
                blk = fg_3d[z0:z1, y0:y1, x0:x1]
                mask[z0:z1, y0:y1, x0:x1] = (blk > foreground_threshold).astype(np.uint8)

    # 2) Blockwise connected components
    if verbose: print("Computing connected components out-of-core...")
    parallel.label(mask, block_shape=block_shape, out=seg, verbose=verbose)
    del root["mask"]

    # 3) Size filter OOC
    if min_size and min_size > 0:
        if verbose: print(f"Applying size filter (min_size={min_size})...")
        seg = apply_size_filter_ooc(seg, min_size=min_size, block_shape=block_shape, out=seg, verbose=verbose)

    if return_binary:
        if verbose: print("Converting back to binary out-of-core...")
        for z0 in range(0, Z, bz):
            z1 = min(z0 + bz, Z)
            for y0 in range(0, Y, by):
                y1 = min(y0 + by, Y)
                for x0 in range(0, X, bx):
                    x1 = min(x0 + bx, X)
                    blk = seg[z0:z1, y0:y1, x0:x1]
                    seg[z0:z1, y0:y1, x0:x1] = np.where(blk > 0, 1, 0)

    return seg
