"""
View a raw + label Zarr/HDF5 pair in napari with per-axis stride downscaling.

Usage:
    python visualize_resize_zarr.py \
        --raw_path /path/to/raw.zarr --raw_key s0 \
        --label_path /path/to/labels.zarr --label_key labels/mito \
        --scale 2 4 4
"""

import argparse
import numpy as np
import napari
from elf.io import open_file


def load_strided(path, key, scale_zyx):
    """Load a dataset from Zarr/HDF5 with per-axis stride downscaling."""
    sz, sy, sx = scale_zyx
    with open_file(path, mode="r") as f:
        arr = f[key]
        ndim = arr.ndim
        if ndim >= 3:
            prefix = (slice(None),) * (ndim - 3)
            slicing = prefix + (slice(None, None, sz), slice(None, None, sy), slice(None, None, sx))
        else:
            slicing = (slice(None),) * ndim
        data = np.asarray(arr[slicing])
    return data


def main():
    parser = argparse.ArgumentParser(
        description="View raw + label arrays (Zarr/HDF5) in napari with per-axis downscaling."
    )
    parser.add_argument("--raw_path", "-rp", required=True,
                        help="Path to the raw Zarr/HDF5 file")
    parser.add_argument("--raw_key", "-rk", required=True,
                        help="Dataset key inside the raw file (e.g. 's0' or 'raw')")
    parser.add_argument("--label_path", "-lp", default=None,
                        help="Path to the label Zarr/HDF5 file (optional)")
    parser.add_argument("--label_key", "-lk", default=None,
                        help="Dataset key inside the label file (e.g. 'labels/mito')")
    parser.add_argument(
        "--scale", "-s",
        type=int, nargs=3, metavar=("Z", "Y", "X"),
        default=[1, 1, 1],
        help="Integer downscale factors for (z, y, x), e.g. --scale 2 4 4"
    )
    args = parser.parse_args()

    if args.label_path is not None and args.label_key is None:
        parser.error("--label_key is required when --label_path is given")

    scale_zyx = tuple(args.scale)
    print(f"Scale factors — z:{scale_zyx[0]}  y:{scale_zyx[1]}  x:{scale_zyx[2]}")

    print(f"Loading raw:    {args.raw_path}[{args.raw_key}]")
    raw = load_strided(args.raw_path, args.raw_key, scale_zyx)
    print(f"  shape: {raw.shape}  dtype: {raw.dtype}")

    labels = None
    if args.label_path is not None:
        print(f"Loading labels: {args.label_path}[{args.label_key}]")
        labels = load_strided(args.label_path, args.label_key, scale_zyx)
        print(f"  shape: {labels.shape}  dtype: {labels.dtype}")

    viewer = napari.Viewer()
    viewer.add_image(raw, name=f"raw:{args.raw_key}")
    if labels is not None:
        viewer.add_labels(labels, name=f"labels:{args.label_key}")
    napari.run()


if __name__ == "__main__":
    main()
