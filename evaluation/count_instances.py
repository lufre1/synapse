import argparse
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from glob import glob

import numpy as np
import pandas as pd
import tifffile
from elf.io import open_file
from tqdm import tqdm


SUPPORTED_EXT = (".h5", ".hdf5", ".zarr", ".n5", ".tif", ".tiff")
BLOCK = (128, 512, 512)  # OOC block shape


def find_files(root, ext=None):
    if os.path.isfile(root):
        return [root]
    if os.path.isdir(root) and root.endswith(".zarr"):
        return [root]
    exts = [ext] if ext else SUPPORTED_EXT
    paths = []
    for e in exts:
        paths.extend(sorted(glob(os.path.join(root, "**", f"*{e}"), recursive=True)))
    return sorted(set(paths))


def _accumulate_blockwise(arr, block=BLOCK, n_workers=8):
    """Count voxels per label ID without loading the full volume.

    Blocks are read and processed in parallel with threads — zarr/h5
    decompression releases the GIL so multiple blocks decompress concurrently.
    """
    shape = arr.shape[-3:]
    has_prefix = len(arr.shape) > 3
    bz, by, bx = (min(b, s) for b, s in zip(block, shape))
    Z, Y, X = shape
    total_voxels = int(np.prod(shape))

    slices = [
        (slice(z0, min(z0+bz, Z)),
         slice(y0, min(y0+by, Y)),
         slice(x0, min(x0+bx, X)))
        for z0 in range(0, Z, bz)
        for y0 in range(0, Y, by)
        for x0 in range(0, X, bx)
    ]

    def process_block(sl):
        blk = np.asarray(arr[(...,) + sl] if has_prefix else arr[sl])
        ids, counts = np.unique(blk, return_counts=True)
        return {int(i): int(c) for i, c in zip(ids, counts) if i != 0}

    # collect results first, merge in main thread — Counter.update is not thread-safe
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(tqdm(pool.map(process_block, slices), total=len(slices),
                            desc="  blocks", leave=False))

    counter = Counter()
    for r in results:
        counter.update(r)

    return counter, total_voxels


def stats_from_counter(counter, total_voxels):
    n = len(counter)
    if n == 0:
        return dict(n_instances=0, total_fg_voxels=0, total_voxels=total_voxels,
                    fg_fraction=0.0,
                    vol_mean=np.nan, vol_median=np.nan, vol_std=np.nan,
                    vol_min=np.nan, vol_max=np.nan)
    volumes = np.array(list(counter.values()), dtype=np.float64)
    total_fg = int(volumes.sum())
    return dict(
        n_instances=n,
        total_fg_voxels=total_fg,
        total_voxels=total_voxels,
        fg_fraction=round(total_fg / total_voxels, 4),
        vol_mean=round(float(volumes.mean()), 1),
        vol_median=round(float(np.median(volumes)), 1),
        vol_std=round(float(volumes.std()), 1),
        vol_min=int(volumes.min()),
        vol_max=int(volumes.max()),
    )


def process_file(path, key, n_workers=8):
    ext = os.path.splitext(path)[-1].lower()

    if ext in (".tif", ".tiff"):
        arr = tifffile.imread(path)
        counter, total = _accumulate_blockwise(arr, n_workers=n_workers)
        return stats_from_counter(counter, total)

    # H5 / zarr / N5 — keep lazy, never load full array
    with open_file(path, "r") as f:
        if key not in f:
            for fallback in ("seg", "segmentation", "labels", "data", "s0"):
                if fallback in f:
                    tqdm.write(f"    key '{key}' not found, using '{fallback}'")
                    key = fallback
                    break
            else:
                raise KeyError(f"key '{key}' not found. Available: {list(f.keys())}")
        arr = f[key]
        tqdm.write(f"    shape={arr.shape} dtype={arr.dtype}")
        counter, total = _accumulate_blockwise(arr, n_workers=n_workers)

    return stats_from_counter(counter, total)


def main():
    parser = argparse.ArgumentParser(
        description="Count instances and volume stats for segmentations in a folder")
    parser.add_argument("--path", "-p", required=True,
                        help="Folder or single file to process")
    parser.add_argument("--key", "-k", default="seg",
                        help="Dataset key for H5/zarr (default: seg)")
    parser.add_argument("--ext", "-e", default=None,
                        help="Restrict to one file extension (e.g. .h5)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output CSV path (default: <path>/instance_stats.csv)")
    parser.add_argument("--n_workers", "-j", type=int, default=8,
                        help="Parallel threads for block reading (default: 8)")
    args = parser.parse_args()

    root = args.path.rstrip("/")
    out_dir = root if os.path.isdir(root) else os.path.dirname(root)
    out_csv = args.output or os.path.join(out_dir, "instance_stats.csv")

    paths = find_files(root, args.ext)
    if not paths:
        print(f"No supported files found in {root}")
        return

    print(f"Found {len(paths)} file(s). Writing to {out_csv}\n")

    rows = []
    for path in tqdm(paths, desc="files"):
        name = os.path.relpath(path, root) if os.path.isdir(root) else os.path.basename(path)
        try:
            s = process_file(path, args.key, n_workers=args.n_workers)
            rows.append({"file": name, **s})
            tqdm.write(f"  {name}: {s['n_instances']} instances  "
                       f"vol mean={s['vol_mean']}  fg={s['fg_fraction']:.1%}")
        except Exception as e:
            tqdm.write(f"  {name}: FAILED — {e}")
            rows.append({"file": name, "error": str(e)})

    df = pd.DataFrame(rows)

    # summary row — only over rows that have numeric stats
    if "n_instances" in df.columns:
        valid = df[df["n_instances"].notna()]
        if len(valid) > 1:
            summary = {
                "file":            f"SUMMARY ({len(valid)} files)",
                "n_instances":     int(valid["n_instances"].sum()),
                "total_fg_voxels": int(valid["total_fg_voxels"].sum()),
                "total_voxels":    int(valid["total_voxels"].sum()),
                "fg_fraction":     round(
                    valid["total_fg_voxels"].sum() / valid["total_voxels"].sum(), 4),
                "vol_mean":        round(float(valid["vol_mean"].mean()), 1),
                "vol_median":      round(float(valid["vol_median"].median()), 1),
                "vol_std":         round(float(valid["vol_std"].mean()), 1),
                "vol_min":         int(valid["vol_min"].min()) if valid["vol_min"].notna().any() else None,
                "vol_max":         int(valid["vol_max"].max()) if valid["vol_max"].notna().any() else None,
            }
            df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    df.to_csv(out_csv, index=False)
    print(f"\nSaved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
