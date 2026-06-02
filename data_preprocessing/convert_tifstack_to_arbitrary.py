import argparse
import math
import os
from concurrent import futures
from glob import glob
import synapse.util as util

import imageio.v3 as iio
import numcodecs
import numpy as np
import zarr
from elf.io import open_file
from tqdm import tqdm


ROOT = "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/orig_files/"
SAMPLES = {
    "4007": {"raw": "2019-06-07_Steyer_Plp-4007 CLAHE", "labels": "4007 Imod-export"},
    "4010": {"raw": "2019-05-10_Steyer_Plp-4010-wt CLAHE", "labels": "4010 Imod-export"},
    "4009": {"raw": "2019-05-14_Steyer_Plp-4009-wt/CLAHE", "labels": None},
    "4016": {"raw": "2019-05-14_Steyer_Plp-4016-wt/CLAHE", "labels": None}
}

DEFAULT_VOXEL_SIZE = [0.025, 0.005, 0.005]  # z, y, x in micrometers


def _make_compressor(out_path):
    """Blosc/lz4 for zarr (fast, ~same ratio); gzip for h5."""
    if out_path.endswith(".zarr") or os.path.isdir(out_path):
        return numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    return None  # caller passes compression="gzip" for h5


def convert_tifs(f, tif_folder, name, n_threads=4, fix_patches=False, out_path=""):
    tifs = sorted(
        glob(os.path.join(tif_folder, "*.tif")) +
        glob(os.path.join(tif_folder, "*.tiff"))
    )
    if not tifs:
        raise ValueError(f"No .tif/.tiff files found in {tif_folder}")

    im0 = iio.imread(tifs[0])
    im_shape = im0.shape
    vol_shape = (len(tifs),) + im_shape

    print(f"Converting volume of shape {vol_shape}, dtype={im0.dtype}")
    ndim = len(vol_shape)
    chunks = (64, 128, 128) if ndim == 3 else (1, 64, 128, 128)

    compressor = _make_compressor(out_path)
    if compressor is not None:
        ds = f.require_dataset(name, shape=vol_shape, chunks=chunks, dtype=im0.dtype,
                               compressor=compressor)
    else:
        ds = f.require_dataset(name, shape=vol_shape, chunks=chunks, dtype=im0.dtype,
                               compression="gzip")

    def _convert_range(z0):
        z1 = min(z0 + chunks[0], vol_shape[0])
        data = np.empty((z1 - z0,) + im_shape, dtype=im0.dtype)
        for i, z in enumerate(range(z0, z1)):
            im = iio.imread(tifs[z])
            if fix_patches:
                im = util.convert_white_patches_to_black(im)
            assert im.shape == im_shape, f"Shape mismatch at slice {z}: {im.shape} != {im_shape}"
            data[i] = im
        ds[z0:z1] = data

    nz = math.ceil(len(tifs) / chunks[0])
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(_convert_range, range(0, len(tifs), chunks[0])),
            desc=f"Convert {name}", total=nz,
        ))

    return ds


def write_voxel_size(f, ds, voxel_size):
    """Store voxel size on both the dataset and the root group attributes."""
    axes = ["z", "y", "x"]
    meta = {"voxel_size": voxel_size, "axes": axes, "unit": "micrometer"}
    ds.attrs.update(meta)
    # OME-Zarr compatible entry on the root group
    f.attrs["voxel_size"] = voxel_size
    f.attrs["axes"] = axes
    f.attrs["unit"] = "micrometer"


def convert_data(sample, ext="zarr", voxel_size=None, n_threads=4, fix_patches=False):
    if voxel_size is None:
        voxel_size = DEFAULT_VOXEL_SIZE

    raw_folder = os.path.join(ROOT, SAMPLES[sample]["raw"])
    assert os.path.exists(raw_folder), f"Input folder {raw_folder} does not exist"

    out_folder = os.path.join(ROOT, sample)
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{sample}.{ext}")
    print(f"Saving to {out_path} under dataset 's0'")

    with open_file(out_path, "a") as f:
        ds = convert_tifs(f, raw_folder, "s0", n_threads=n_threads,
                          fix_patches=fix_patches, out_path=out_path)
        write_voxel_size(f, ds, voxel_size)
        print(f"Voxel size written: z={voxel_size[0]}, y={voxel_size[1]}, x={voxel_size[2]} µm")


def main():
    parser = argparse.ArgumentParser(description="Convert a TIFF stack to Zarr or HDF5")
    # --- existing sample-based workflow ---
    parser.add_argument("--sample", "-s", default=None,
                        help=f"Predefined sample key: {list(SAMPLES)}")
    # --- direct-path workflow ---
    parser.add_argument("--input_dir", "-i", default=None,
                        help="Path to folder of TIFF slices (alternative to --sample)")
    parser.add_argument("--output_path", "-o", default=None,
                        help="Output .zarr or .h5 path (required with --input_dir)")
    parser.add_argument("--dataset_name", default="s0",
                        help="Dataset name inside the output file (default: s0)")
    # --- shared options ---
    parser.add_argument("--ext", default="zarr", choices=["zarr", "h5"],
                        help="Output format when using --sample (default: zarr)")
    parser.add_argument("--voxel_size", type=float, nargs=3,
                        default=DEFAULT_VOXEL_SIZE,
                        metavar=("Z", "Y", "X"),
                        help="Voxel size in µm, z y x (default: 0.025 0.005 0.005)")
    parser.add_argument("--n_threads", type=int, default=4,
                        help="Number of I/O threads (default: 4)")
    parser.add_argument("--fix_patches", action="store_true",
                        help="Apply convert_white_patches_to_black on each slice")

    args = parser.parse_args()

    if args.input_dir is not None:
        # direct-path mode
        if args.output_path is None:
            parser.error("--output_path is required when using --input_dir")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        with open_file(args.output_path, "a") as f:
            ds = convert_tifs(f, args.input_dir, args.dataset_name,
                              n_threads=args.n_threads, fix_patches=args.fix_patches,
                              out_path=args.output_path)
            write_voxel_size(f, ds, args.voxel_size)
            print(f"Voxel size written: z={args.voxel_size[0]}, y={args.voxel_size[1]}, x={args.voxel_size[2]} µm")
    elif args.sample is not None:
        assert args.sample in SAMPLES, f"Sample not found; choose from {list(SAMPLES)}"
        convert_data(args.sample, ext=args.ext, voxel_size=args.voxel_size,
                     n_threads=args.n_threads, fix_patches=args.fix_patches)
    else:
        parser.error("Provide either --sample or --input_dir")


if __name__ == "__main__":
    main()