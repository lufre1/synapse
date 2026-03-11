import argparse
import math
import os
from concurrent import futures
from glob import glob
import synapse.util as util

import imageio.v3 as iio # Using v3 is recommended for newer imageio
import numpy as np
import zarr
from elf.io import open_file
from tqdm import tqdm


ROOT = "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/orig_files/"
SAMPLES = {
    "4007": {"raw": "2019-06-07_Steyer_Plp-4007 CLAHE", "labels": "4007 Imod-export"},
    "4010": {"raw": "2019-05-10_Steyer_Plp-4010-wt CLAHE", "labels": "4010 Imod-export"},
    "4009": {"raw": "2019-05-14_Steyer_Plp-4009-wt/CLAHE", "labels": None},
}


def convert_tifs(f, tif_folder, name, n_threads=4): # Lowered threads to play nice with Lustre
    tifs = glob(os.path.join(tif_folder, "*.tif"))
    tifs.sort() # Ensure files are zero-padded! e.g., 0001.tif
    
    if not tifs:
        raise ValueError(f"No .tif files found in {tif_folder}. Are they .tiff?")

    im0 = iio.imread(tifs[0])
    im_shape = im0.shape
    vol_shape = (len(tifs),) + im_shape

    print(f"Converting volume of shape {vol_shape}")
    ndim = len(vol_shape)
    chunks = (8, 512, 512)
    if ndim == 4:
        chunks = (1,) + chunks
        
    ds = f.require_dataset(name, shape=vol_shape, compression="gzip", chunks=chunks, dtype=im0.dtype)

    def _convert_range(z0):
        z1 = min(z0 + chunks[0], vol_shape[0])
        sub_shape = (z1 - z0,) + im_shape
        data = np.zeros(sub_shape, dtype=im0.dtype)
        for i, z in enumerate(range(z0, z1)):
            im = iio.imread(tifs[z])
            im = util.convert_white_patches_to_black(im)
            assert im.shape == im_shape, f"Shape mismatch at slice {z}"
            data[i] = im
        ds[z0:z1] = data

    # Fixed true division (/) so tqdm total is accurate
    nz = math.ceil(len(tifs) / chunks[0])
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(_convert_range, range(0, len(tifs), chunks[0])), desc=f"Convert {name}", total=nz
        ))


def convert_data(sample, ext="zarr"):
    raw_folder = os.path.join(ROOT, SAMPLES[sample]["raw"])
    assert os.path.exists(raw_folder), f"Input folder {raw_folder} does not exist"

    out_folder = os.path.join(ROOT, sample)
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{sample}.{ext}")
    print(f"Saving to {out_path} under dataset 's0'")
    
    with open_file(out_path, "a") as f:
        convert_tifs(f, raw_folder, "s0")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", "-s", default="4009")
    args = parser.parse_args()
    sample = args.sample
    assert sample in SAMPLES, "Sample not found in SAMPLES dictionary"
    convert_data(sample, ext="zarr")


if __name__ == "__main__":
    main()