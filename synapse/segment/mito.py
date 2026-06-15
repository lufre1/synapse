"""Mitochondria segmentation.

In-memory watershed with seeded markers, plus out-of-core variants
that persist intermediate zarr datasets for resumability.
"""
import time
import numpy as np
import zarr
from elf import parallel as parallel
from elf import wrapper
from elf.wrapper.base import MultiTransformationWrapper
from skimage.measure import label as skimage_label


def segment_mitos(
    foreground: np.ndarray,
    boundary: np.ndarray,
    block_shape=(128, 256, 256),
    halo=(32, 48, 48),
    seed_distance=4,
    boundary_threshold=0.15,
    foreground_threshold=0.85,
    min_size=2000,
    area_threshold=500,
    dist=None,
    post_iter=4,
    post_iter3d=8
):
    """Return a dict with segmentation, seeds, distance map and h-map."""
    from synapse_net.inference.util import apply_size_filter, _postprocess_seg_3d
    boundaries = boundary
    if dist is None:
        dist = parallel.distance_transform(
            boundaries < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape
        )
    hmap = (dist.max() - dist) / dist.max()
    hmap[np.logical_and(boundaries > boundary_threshold, foreground < foreground_threshold)] = (hmap + boundaries).max()

    seeds = np.logical_and(foreground > foreground_threshold, dist > seed_distance)
    seeds = skimage_label(seeds, connectivity=2)
    seeds = apply_size_filter(seeds, min_size, verbose=True, block_shape=block_shape)

    mask = (foreground + np.where(boundaries < boundary_threshold, boundaries, 0)) > 0.5

    seg = np.zeros_like(seeds)
    seg = parallel.seeded_watershed(
        hmap, seeds, block_shape=block_shape, out=seg, verbose=True, halo=halo, mask=mask
    )
    seg = apply_size_filter(seg, min_size, verbose=True, block_shape=block_shape)
    seg = _postprocess_seg_3d(seg, area_threshold=area_threshold, iterations=post_iter, iterations_3d=post_iter3d)

    return {
        "segmentation": seg.astype(np.uint8),
        "seeds": seeds.astype(np.uint8),
        "dist": dist.astype(np.float32),
        "hmap": hmap.astype(np.float32),
        "mask": mask.astype(np.uint8),
    }


def segment_mitos_ooc_wrapped(
    pred, out_dir, min_size=250,
    verbose=True,
    block_shape=(128, 256, 256),
    halo=(48, 48, 48),
    seed_distance=6,
    boundary_threshold=0.25,
    foreground_threshold=0.5,
    area_threshold=5000,   # kept for signature compatibility (not used here)
    reuse_computed=False,
    bg_penalty=2.0,        # height-map penalty for barrier voxels; lower (e.g. 1.2) reduces fragmentation
    n_threads=8,           # cap threads to match SLURM CPU allocation
):
    # Import deferred to avoid circular import at module level
    from synapse.util import iterate_blocks, apply_size_filter_ooc_optim, ZarrChannelWrapper

    # pred is (C,Z,Y,X)
    shape = pred.shape[1:]
    fg, bd = ZarrChannelWrapper(pred, 0), ZarrChannelWrapper(pred, 1)

    store = zarr.DirectoryStore(out_dir)
    root = zarr.group(store=store)

    def needs(name):
        return (not reuse_computed) or (name not in root)

    # --- dist (persisted) ---
    dist = root.get("dist")

    if dist is None or needs("dist"):
        dist = root.require_dataset("dist", shape=shape, chunks=block_shape, dtype=np.float32)
        if verbose: print("Computing dist (distance transform)...")
        t0 = time.time()

        boundaries_thresh = wrapper.SimpleTransformationWrapper(
            bd, lambda x: x < boundary_threshold
        )

        parallel.distance_transform(
            boundaries_thresh, halo=halo, verbose=verbose,
            block_shape=block_shape, distances=dist, n_threads=n_threads,
        )
        if verbose: print("dist in", time.time() - t0, "s")
    elif verbose:
        print("Reusing existing dist...")

    # --- seeds (persisted labeled CC) ---
    seeds = root.get("seeds")

    if seeds is None or needs("seeds"):
        seeds = root.require_dataset("seeds", shape=shape, chunks=block_shape, dtype=np.uint32)
        if verbose: print("Computing seeds (mask) + CC labeling...")
        t0 = time.time()

        seed_mask = MultiTransformationWrapper(
            lambda f, d: np.logical_and(f > foreground_threshold, d > seed_distance),
            fg, dist
        )

        parallel.label(seed_mask, block_shape=block_shape, verbose=verbose, out=seeds, n_threads=n_threads)
        if verbose: print("seeds in", time.time() - t0, "s")
    elif verbose:
        print("Reusing existing seeds...")

    # --- compute dist_max (blockwise reduction) ---
    if verbose: print("Computing dist_max...")
    t0 = time.time()
    dist_max = float(max(np.max(dist[bb]) for bb in iterate_blocks(shape, block_shape)))
    if verbose: print("dist_max =", dist_max, "in", time.time() - t0, "s")

    # --- hmap + mask as virtual volumes (no datasets written) ---
    def hmap_tf(d_chunk, index):
        f_chunk = fg[index]
        b_chunk = bd[index]
        h = (dist_max - d_chunk) / dist_max
        bg = np.logical_and(b_chunk > boundary_threshold, f_chunk < boundary_threshold)
        h[bg] = bg_penalty
        return h.astype(np.float32, copy=False)

    hmap = wrapper.TransformationWrapper(dist, hmap_tf)

    mask = MultiTransformationWrapper(
        lambda f, b: (f + b) > 0.5,
        fg, bd
    )

    # --- watershed (persisted) ---
    seg = root.get("seg")

    if seg is None or needs("seg"):
        seg = root.require_dataset("seg", shape=shape, chunks=block_shape, dtype=np.uint32)
        if verbose: print("Computing watershed...")
        t0 = time.time()

        parallel.seeded_watershed(
            hmap, seeds,
            block_shape=block_shape, halo=halo,
            out=seg, mask=mask,
            verbose=verbose, n_threads=n_threads,
        )
        if verbose: print("watershed in", time.time() - t0, "s")

        if verbose: print("Applying size filter...")
        seg = apply_size_filter_ooc_optim(seg, min_size, verbose=verbose, block_shape=block_shape)
    elif verbose:
        print("Reusing existing segmentation...")

    return {"segmentation": seg}


def segment_mitos_ooc_optimized(pred, out_dir, min_size=250,
    verbose=True,
    block_shape=(128, 256, 256),
    halo=(48, 48, 48),
    seed_distance=6,
    boundary_threshold=0.25,
    area_threshold=5000,
    reuse_computed=False,
):
    from synapse.util import iterate_blocks, apply_size_filter_ooc_optim

    # WARNING: This approach uses dataset existence as the proxy for completion!!
    # FIX: pred is (c, z, y, x). We need (z, y, x) for the out-of-core shapes.
    shape = pred.shape[1:]

    store = zarr.DirectoryStore(out_dir)
    root = zarr.group(store=store, overwrite=not reuse_computed)

    def needs_compute(name):
        return (not reuse_computed) or (name not in root)

    # 1. Blockwise Thresholding
    compute_thresh = needs_compute('boundaries_thresh')
    boundaries_thresh = root.require_dataset('boundaries_thresh', shape=shape, chunks=block_shape, dtype=bool)
    if compute_thresh:
        if verbose: print("Computing boundaries_thresh...")
        for bb in iterate_blocks(shape, block_shape):
            boundaries_thresh[bb] = pred[(1,) + bb] < boundary_threshold
    elif verbose:
        print("Reusing existing boundaries_thresh...")

    # 1.5 Distance Transform
    compute_dist = needs_compute('dist')
    dist = root.require_dataset('dist', shape=shape, chunks=block_shape, dtype=np.float32)
    if compute_dist:
        t0 = time.time()
        parallel.distance_transform(
            boundaries_thresh, halo=halo, verbose=verbose, block_shape=block_shape, distances=dist
        )
        if verbose: print("Compute distance transform in", time.time() - t0, "s")
    elif verbose:
        print("Reusing existing dist...")

    # 2. Blockwise Seed Generation
    compute_seeds = needs_compute('seeds')
    seeds = root.require_dataset('seeds', shape=shape, chunks=block_shape, dtype=np.uint32)
    if compute_seeds:
        if verbose: print("Computing seeds and connected components...")
        for bb in iterate_blocks(shape, block_shape):
            seeds[bb] = np.logical_and(pred[(0,) + bb] > 0.5, dist[bb] > seed_distance)

        t0 = time.time()
        parallel.label(seeds, block_shape=block_shape, verbose=verbose, out=seeds)
        if verbose: print("Compute connected components in", time.time() - t0, "s")
    elif verbose:
        print("Reusing existing seeds...")

    # 3. Blockwise HMAP and Mask creation
    compute_hmap_mask = needs_compute('hmap') or needs_compute('mask')
    hmap = root.require_dataset('hmap', shape=shape, chunks=block_shape, dtype=np.float32)
    mask = root.require_dataset('mask', shape=shape, chunks=block_shape, dtype=bool)
    if compute_hmap_mask:
        if verbose: print("Computing hmap and mask...")
        t0 = time.time()
        dist_max = np.max([np.max(dist[bb]) for bb in iterate_blocks(shape, block_shape)])

        bg_penalty = 2.0

        for bb in iterate_blocks(shape, block_shape):
            f_chunk = pred[(0,) + bb]
            b_chunk = pred[(1,) + bb]

            h_chunk = (dist_max - dist[bb]) / dist_max
            bg_mask = np.logical_and(b_chunk > boundary_threshold, f_chunk < boundary_threshold)
            h_chunk[bg_mask] = bg_penalty

            hmap[bb] = h_chunk
            mask[bb] = (f_chunk + b_chunk) > 0.5
        if verbose: print("Compute hmap and mask in", time.time() - t0, "s")
    elif verbose:
        print("Reusing existing hmap and mask...")

    # 4. Watershed
    compute_seg = needs_compute('seg')
    seg = root.require_dataset('seg', shape=shape, chunks=block_shape, dtype=np.uint32)
    if compute_seg:
        if verbose: print("Computing watershed...")
        t0 = time.time()
        seg = parallel.seeded_watershed(
            hmap, seeds, block_shape=block_shape,
            out=seg, mask=mask, verbose=verbose, halo=halo,
        )
        if verbose: print("Compute watershed in", time.time() - t0, "s")

        # Pass the out-of-core seg object to your filters
        # Note: Size filter is combined with watershed creation block so it doesn't run twice
        if verbose: print("Applying size filter...")
        seg = apply_size_filter_ooc_optim(seg, min_size, verbose=verbose, block_shape=block_shape)
    elif verbose:
        print("Reusing existing segmentation...")

    return {"segmentation": seg}
