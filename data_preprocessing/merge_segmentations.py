#!/usr/bin/env python3
"""
Merge two zarr segmentation volumes.

For every instance in seg2:
  - If it overlaps an instance in seg1 by more than `--threshold` of
    seg2's instance volume, the entire seg1 instance is removed and
    the seg2 instance is written in its place (new unique ID).
  - Otherwise the seg2 instance is added as a brand-new instance.

All processing is out-of-core (blockwise) to handle large volumes.

Examples
--------
  python data_preprocessing/merge_segmentations.py \\
      -s1 base.zarr -k1 seg \\
      -s2 corrections.zarr -k2 seg \\
      -o merged.zarr -ok seg
"""

import argparse
import os
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr
from tqdm import tqdm


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _open(path, key):
    return zarr.open(path, mode="r")[key]


def _block_coords(shape, block_shape):
    Z, Y, X = shape
    bz, by, bx = block_shape
    return [
        (z0, y0, x0)
        for z0 in range(0, Z, bz)
        for y0 in range(0, Y, by)
        for x0 in range(0, X, bx)
    ]


def _read_block(arr, z0, y0, x0, z1, y1, x1):
    sl = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
    if arr.ndim == 3:
        return np.asarray(arr[sl])
    return np.asarray(arr[(0,) + sl])  # channel-first 4-D


# ---------------------------------------------------------------------------
# Optional: nearest-neighbour resize seg2 to match seg1's spatial shape
# ---------------------------------------------------------------------------

def _resize_nearest_blockwise(src_arr, target_shape, tmp_path, block_shape, n_workers):
    """Write src_arr resampled to target_shape into tmp_path zarr using nearest-neighbour.

    For each output block the corresponding input bounding box is read once,
    then np.ix_ fancy-indexing picks the nearest source voxel per output voxel.
    No interpolation — label IDs are always preserved exactly.
    """
    in_shape = tuple(src_arr.shape[-3:])
    out_shape = target_shape
    # Scale factors: one input voxel spans this many output voxels.
    sz = in_shape[0] / out_shape[0]
    sy = in_shape[1] / out_shape[1]
    sx = in_shape[2] / out_shape[2]

    tmp_root = zarr.open(tmp_path, mode="w")
    chunks = tuple(min(b, s) for b, s in zip(block_shape, out_shape))
    out_arr = tmp_root.create_dataset(
        "data", shape=out_shape, dtype=src_arr.dtype, chunks=chunks,
        compressor=zarr.Blosc(cname="lz4", clevel=1, shuffle=1), overwrite=True,
    )

    coords = _block_coords(out_shape, block_shape)

    def _process(c):
        oz0, oy0, ox0 = c
        oz1 = min(oz0 + block_shape[0], out_shape[0])
        oy1 = min(oy0 + block_shape[1], out_shape[1])
        ox1 = min(ox0 + block_shape[2], out_shape[2])

        # Nearest source index for each output index in this block.
        iz = np.clip(np.round(np.arange(oz0, oz1) * sz).astype(int), 0, in_shape[0] - 1)
        iy = np.clip(np.round(np.arange(oy0, oy1) * sy).astype(int), 0, in_shape[1] - 1)
        ix = np.clip(np.round(np.arange(ox0, ox1) * sx).astype(int), 0, in_shape[2] - 1)

        # Read the minimal bounding box from source.
        slab = _read_block(src_arr, iz[0], iy[0], ix[0], iz[-1] + 1, iy[-1] + 1, ix[-1] + 1)

        # Relative indices within the slab, then fancy-index to get output block.
        out_arr[oz0:oz1, oy0:oy1, ox0:ox1] = slab[
            np.ix_(iz - iz[0], iy - iy[0], ix - ix[0])
        ]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(tqdm(pool.map(_process, coords), total=len(coords),
                  desc="  Resizing seg2"))

    return out_arr


# ---------------------------------------------------------------------------
# Step 1: count voxels per instance and pairwise (id2, id1) overlaps
# ---------------------------------------------------------------------------

def _scan(arr1, arr2, block_shape, n_workers):
    shape = arr1.shape[-3:]
    coords = _block_coords(shape, block_shape)

    def _process(c):
        z0, y0, x0 = c
        z1 = min(z0 + block_shape[0], shape[0])
        y1 = min(y0 + block_shape[1], shape[1])
        x1 = min(x0 + block_shape[2], shape[2])
        b1 = _read_block(arr1, z0, y0, x0, z1, y1, x1)
        b2 = _read_block(arr2, z0, y0, x0, z1, y1, x1)

        ids1, c1 = np.unique(b1[b1 > 0], return_counts=True)
        ids2, c2 = np.unique(b2[b2 > 0], return_counts=True)

        # Pairwise overlap: stack co-occurring (id2, id1) pairs and count.
        mask = (b1 > 0) & (b2 > 0)
        if mask.any():
            pairs = np.stack([b2[mask].ravel(), b1[mask].ravel()], axis=1)
            upairs, ucounts = np.unique(pairs, axis=0, return_counts=True)
            ov = {(int(p[0]), int(p[1])): int(c) for p, c in zip(upairs, ucounts)}
        else:
            ov = {}

        return (
            {int(i): int(c) for i, c in zip(ids1, c1)},
            {int(i): int(c) for i, c in zip(ids2, c2)},
            ov,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(tqdm(pool.map(_process, coords), total=len(coords),
                            desc="  Scanning"))

    count1: dict[int, int] = defaultdict(int)
    count2: dict[int, int] = defaultdict(int)
    overlap: dict[tuple[int, int], int] = defaultdict(int)
    for r1, r2, ov in results:
        for k, v in r1.items():
            count1[k] += v
        for k, v in r2.items():
            count2[k] += v
        for k, v in ov.items():
            overlap[k] += v

    return dict(count1), dict(count2), dict(overlap)


# ---------------------------------------------------------------------------
# Step 2: decide replacements, assign new IDs
# ---------------------------------------------------------------------------

def _plan(count1, count2, overlap, threshold):
    """Return (remove_ids1, id2_to_new_id).

    remove_ids1  – set of seg1 IDs to zero out
    id2_to_new_id – dict mapping every seg2 ID to its output ID
    """
    # Group overlap entries by id2 for O(1) per-instance lookup.
    by_id2: dict[int, dict[int, int]] = defaultdict(dict)
    for (id2, id1), ov in overlap.items():
        by_id2[id2][id1] = ov

    next_id = max(count1.keys(), default=0) + 1
    remove_ids1: set[int] = set()
    id2_to_new: dict[int, int] = {}
    replaced_count = 0

    for id2, vol2 in count2.items():
        to_replace = [
            id1 for id1, ov in by_id2.get(id2, {}).items()
            if ov / vol2 > threshold
        ]
        for id1 in to_replace:
            remove_ids1.add(id1)
        id2_to_new[id2] = next_id
        next_id += 1
        if to_replace:
            replaced_count += 1

    return remove_ids1, id2_to_new, replaced_count


# ---------------------------------------------------------------------------
# Step 3: write merged output blockwise
# ---------------------------------------------------------------------------

def _write(arr1, arr2, out_arr, remove_ids1, id2_to_new, block_shape, n_workers):
    shape = arr1.shape[-3:]
    coords = _block_coords(shape, block_shape)

    # Convert remove set to sorted numpy array for np.isin.
    remove_arr = np.array(sorted(remove_ids1), dtype=np.uint64)

    def _process(c):
        z0, y0, x0 = c
        z1 = min(z0 + block_shape[0], shape[0])
        y1 = min(y0 + block_shape[1], shape[1])
        x1 = min(x0 + block_shape[2], shape[2])
        b1 = _read_block(arr1, z0, y0, x0, z1, y1, x1).astype(np.uint64)
        b2 = _read_block(arr2, z0, y0, x0, z1, y1, x1).astype(np.uint64)

        # Erase replaced seg1 instances.
        if remove_arr.size > 0:
            b1[np.isin(b1, remove_arr)] = 0

        # Insert seg2 instances (only where they are mapped).
        for uid in np.unique(b2):
            if uid == 0:
                continue
            new_id = id2_to_new.get(int(uid))
            if new_id is not None:
                b1[b2 == uid] = new_id

        out_arr[z0:z1, y0:y1, x0:x1] = b1

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(tqdm(pool.map(_process, coords), total=len(coords),
                  desc="  Writing"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Merge seg2 instances into seg1, replacing overlapping ones.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--seg1",         "-s1",  required=True,
                    help="Base segmentation zarr")
    ap.add_argument("--seg2",         "-s2",  required=True,
                    help="Segmentation to merge in")
    ap.add_argument("--output",       "-o",   required=True,
                    help="Output zarr")
    ap.add_argument("--key1",         "-k1",  default="seg")
    ap.add_argument("--key2",         "-k2",  default="seg")
    ap.add_argument("--output_key",   "-ok",  default="seg")
    ap.add_argument("--threshold",    "-t",   type=float, default=0.5,
                    help="Fraction of a seg2 instance's voxels that must overlap "
                         "a seg1 instance to trigger replacement")
    ap.add_argument("--block_shape",  "-bs",  type=int, nargs=3,
                    default=[64, 256, 256], metavar=("Z", "Y", "X"))
    ap.add_argument("--n_workers",    "-nw",  type=int, default=8)
    ap.add_argument("--tmp_dir",      "-tmp", default=None,
                    help="Directory for temporary resized seg2 zarr "
                         "(default: same directory as output)")
    args = ap.parse_args()

    block_shape = tuple(args.block_shape)
    arr1 = _open(args.seg1, args.key1)
    arr2 = _open(args.seg2, args.key2)
    shape1 = tuple(arr1.shape[-3:])
    shape2 = tuple(arr2.shape[-3:])
    print(f"seg1 : {args.seg1} [{args.key1}]  shape={arr1.shape}  dtype={arr1.dtype}")
    print(f"seg2 : {args.seg2} [{args.key2}]  shape={arr2.shape}  dtype={arr2.dtype}")
    print(f"threshold = {args.threshold}")

    tmp_store = None
    try:
        # Resize seg2 to seg1's spatial shape when they differ.
        if shape2 != shape1:
            tmp_root = args.tmp_dir or os.path.dirname(os.path.abspath(args.output)) or "."
            tmp_store = os.path.join(tmp_root, "_merge_tmp_seg2.zarr")
            print(f"\nResizing seg2 {shape2} → {shape1} (nearest-neighbour) ...")
            arr2 = _resize_nearest_blockwise(arr2, shape1, tmp_store, block_shape, args.n_workers)

        # ------------------------------------------------------------------
        print("\nStep 1/2 — Scanning instance counts and overlaps ...")
        count1, count2, overlap = _scan(arr1, arr2, block_shape, args.n_workers)
        print(f"  seg1 instances : {len(count1)}")
        print(f"  seg2 instances : {len(count2)}")

        # ------------------------------------------------------------------
        remove_ids1, id2_to_new, n_replaced = _plan(
            count1, count2, overlap, args.threshold
        )
        n_new = len(id2_to_new) - n_replaced
        print(f"  seg1 instances removed  : {len(remove_ids1)}")
        print(f"  seg2 instances replacing: {n_replaced}")
        print(f"  seg2 instances added new: {n_new}")

        # ------------------------------------------------------------------
        print("\nStep 2/2 — Writing merged output ...")
        chunks = tuple(min(b, s) for b, s in zip(block_shape, shape1))
        out_root = zarr.open(args.output, mode="w")
        out_arr = out_root.create_dataset(
            args.output_key, shape=shape1, dtype=np.uint64, chunks=chunks,
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2), overwrite=True,
        )
        _write(arr1, arr2, out_arr, remove_ids1, id2_to_new, block_shape, args.n_workers)
        print("Done.")

    finally:
        if tmp_store and os.path.isdir(tmp_store):
            shutil.rmtree(tmp_store)


if __name__ == "__main__":
    main()
