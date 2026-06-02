"""Visualize a folder of 2-D TIFF slices as a 3-D volume in napari.

The full volume is too large to fit in RAM, so each slice is downscaled
spatially (XY) by `--xy_scale` and every `--z_scale`-th slice is kept.
Default is 4× in every axis, reducing a 22 GB volume to ~350 MB.

Usage
-----
    python visualize_tiff_stack.py /path/to/tiff_folder
    python visualize_tiff_stack.py /path/to/tiff_folder --xy_scale 2 --z_scale 2
"""

import argparse
import os
import sys
from glob import glob

import imageio.v3 as iio
import napari
import numpy as np
from skimage.transform import downscale_local_mean
from tqdm import tqdm


def load_downscaled_stack(folder: str, xy_scale: int, z_scale: int) -> np.ndarray:
    paths = sorted(glob(os.path.join(folder, "*.tif")) + glob(os.path.join(folder, "*.tiff")))
    if not paths:
        sys.exit(f"No .tif/.tiff files found in {folder}")

    z_paths = paths[::z_scale]
    print(f"Loading {len(z_paths)} / {len(paths)} slices (z_scale={z_scale}), "
          f"downscaling XY by {xy_scale}x ...")

    # read first slice to get output shape
    first = iio.imread(z_paths[0])
    h_out = int(np.ceil(first.shape[0] / xy_scale))
    w_out = int(np.ceil(first.shape[1] / xy_scale))
    volume = np.empty((len(z_paths), h_out, w_out), dtype=first.dtype)

    for i, path in enumerate(tqdm(z_paths)):
        sl = iio.imread(path)
        volume[i] = downscale_local_mean(sl, (xy_scale, xy_scale)).astype(first.dtype)

    size_mb = volume.nbytes / 1024 ** 2
    print(f"Volume shape: {volume.shape}, {size_mb:.0f} MB")
    return volume


def main():
    parser = argparse.ArgumentParser(description="Visualize a TIFF stack in napari with downscaling")
    parser.add_argument("folder", help="Directory containing the TIFF slices")
    parser.add_argument("--xy_scale", type=int, default=4, help="Downscale factor in XY (default: 4)")
    parser.add_argument("--z_scale",  type=int, default=4, help="Keep every N-th slice in Z (default: 4)")
    args = parser.parse_args()

    volume = load_downscaled_stack(args.folder, args.xy_scale, args.z_scale)

    viewer = napari.Viewer()
    viewer.add_image(volume, name=os.path.basename(args.folder.rstrip("/")),
                     contrast_limits=(volume.min(), volume.max()))
    napari.run()


if __name__ == "__main__":
    main()
