import argparse
import sys
from glob import glob
from pathlib import Path

import dask.array as da
import napari
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).parent))
from visualize_zarr import open_arr, lazy_downscale, auto_scale


def open_seg_lazy(path, key, scale, no_z_scale=False):
    """Open a zarr segmentation as a dask array with stride-based downsampling.

    Dask is lazy — napari only reads the chunks it needs per z-slice,
    so all 14 segmentations cost near-zero RAM upfront.
    """
    z_arr = zarr.open(zarr.DirectoryStore(path), mode="r")[key]
    da_arr = da.from_array(z_arr, chunks=z_arr.chunks)
    if no_z_scale:
        return da_arr[:, ::scale, ::scale]
    else:
        return da_arr[::scale, ::scale, ::scale]


def seg_scale(raw_shape, seg_arr, no_z_scale):
    """Compute the integer scale that maps seg onto raw spatial resolution."""
    if no_z_scale:
        # match y/x only
        sy = seg_arr.shape[-2] / raw_shape[-2]
        sx = seg_arr.shape[-1] / raw_shape[-1]
        s = max(1, round((sy + sx) / 2))
    else:
        sz = seg_arr.shape[-3] / raw_shape[-3]
        sy = seg_arr.shape[-2] / raw_shape[-2]
        sx = seg_arr.shape[-1] / raw_shape[-1]
        s = max(1, round((sz + sy + sx) / 3))
    return s


def main():
    parser = argparse.ArgumentParser(description="Open all grid-search segmentations in one napari window")
    parser.add_argument("--grid_dir", "-g", required=True, help="Directory containing *.zarr grid search outputs")
    parser.add_argument("--raw_path", "-sp", required=True, help="Path to the raw EM zarr")
    parser.add_argument("--seg_key", "-k", default="seg", help="Dataset key inside each grid zarr (default: seg)")
    parser.add_argument("--raw_key", "-sk", default="s2", help="Dataset key inside the raw zarr (default: s2)")
    parser.add_argument("--no_z_scale", "-nzs", action="store_true", default=False)
    parser.add_argument("--scale", "-s", type=int, default=1, help="Manual downsample factor for raw (default: auto)")
    args = parser.parse_args()

    zarrs = sorted(glob(str(Path(args.grid_dir) / "*.zarr")))
    if not zarrs:
        print(f"No *.zarr files found in {args.grid_dir}")
        sys.exit(1)

    print(f"Found {len(zarrs)} zarr(s) — loading into one viewer...")

    # --- load raw into memory (once) ---
    raw_arr = open_arr(args.raw_path, args.raw_key)
    raw_scale = args.scale if args.scale > 1 else auto_scale(raw_arr)
    if raw_scale > 1:
        print(f"  raw: downscaling by {raw_scale}x")
    raw = lazy_downscale(raw_arr, raw_scale, no_z_scale=args.no_z_scale)
    print(f"  raw loaded: {raw.shape}")

    viewer = napari.Viewer()
    viewer.add_image(raw, name="raw", colormap="gray")

    # --- lazy-load each segmentation via dask ---
    for idx, zarr_path in enumerate(zarrs):
        name = Path(zarr_path).stem
        try:
            seg_arr = zarr.open(zarr.DirectoryStore(zarr_path), mode="r")[args.seg_key]
            s = seg_scale(raw.shape, seg_arr, args.no_z_scale)
            seg = open_seg_lazy(zarr_path, args.seg_key, s, no_z_scale=args.no_z_scale)
            print(f"  {name}: {seg_arr.shape} → dask {seg.shape} (scale={s})")
            viewer.add_labels(seg, name=name)          # always visible during add (forces correct texture build)
            viewer.layers[-1].visible = (idx == 0)     # hide all but the first afterwards
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    n_segs = len(viewer.layers) - 1
    print(f"\nLoaded {n_segs} segmentation(s). Toggle visibility in the layer list to compare.")
    napari.run()


if __name__ == "__main__":
    main()
