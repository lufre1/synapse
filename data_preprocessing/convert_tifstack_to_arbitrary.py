import argparse
import os
from concurrent import futures
from glob import glob

import imageio
import numpy as np
from elf.io import open_file
from tqdm import tqdm


ROOT = "/mnt/lustre-grete/projects/nim00020/data/volume-em/moebius/orig_files/"
SAMPLES = {
    "4007": {"raw": "2019-06-07_Steyer_Plp-4007 CLAHE", "labels": "4007 Imod-export"},
    "4010": {"raw": "2019-05-10_Steyer_Plp-4010-wt CLAHE", "labels": "4010 Imod-export"},
    "4009": {"raw": "2019-05-14_Steyer_Plp-4009-wt/CLAHE", "labels": None},
}


def convert_tifs(f, tif_folder, name, n_threads=8):
    tifs = glob(os.path.join(tif_folder, "*.tif"))
    tifs.sort()

    im0 = imageio.imread(tifs[0])
    im_shape = im0.shape
    vol_shape = (len(tifs),) + im_shape

    print("Converting volume of shape", vol_shape)
    ndim = len(vol_shape)
    chunks = (8, 512, 512)
    if ndim == 4:
        chunks = chunks + (1,)
    ds = f.require_dataset(name, shape=vol_shape, compression="gzip", chunks=chunks, dtype=im0.dtype)

    def _convert_range(z0):
        z1 = min(z0 + chunks[0], vol_shape[0])
        sub_shape = (z1 - z0,) + im_shape
        data = np.zeros(sub_shape, dtype=im0.dtype)
        for i, z in enumerate(range(z0, z1)):
            im = imageio.imread(tifs[z])
            assert im.shape == im_shape
            data[i] = im
        ds[z0:z1] = data

    nz = len(tifs) // chunks[0] + 1
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(_convert_range, range(0, len(tifs), chunks[0])), desc=f"Convert {name}", total=nz
        ))


def convert_data(sample, ext="n5"):
    raw_folder = os.path.join(ROOT, SAMPLES[sample]["raw"])
    assert os.path.exists(raw_folder)
    # label_folder = os.path.join(ROOT, SAMPLES[sample]["labels"])
    # assert os.path.exists(label_folder)

    out_folder = os.path.join(ROOT, sample, "raw")
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"{sample}.{ext}")
    print("Saving to", out_path)
    with open_file(out_path, "a") as f:
        convert_tifs(f, raw_folder, "raw")
        # convert_tifs(f, label_folder, "labels/imod")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", "-s", default="4009")
    args = parser.parse_args()
    sample = args.sample
    assert sample in SAMPLES
    convert_data(sample, ext="zarr")


if __name__ == "__main__":
    main()