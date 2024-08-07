import h5py
import os
from glob import glob
from tqdm import tqdm
from skimage import measure
import numpy as np


def _read_h5(path, key, scale_factor=1):
    with h5py.File(path, "r") as f:
        try:
            if scale_factor != 1:
                print(f"{key} data shape", f[key].shape)
            image = f[key][:, ::scale_factor, ::scale_factor]
            if scale_factor != 1:
                print(f"{key} data shape after downsampling", image.shape)

        except KeyError:
            print(f"Error: {key} dataset not found in {path}")
            return None  # Indicate error

        return image


def _write_h5(path, key, image):
    with h5py.File(path, "a") as f:
        f.create_dataset(key, data=image, dtype=image.dtype)


def create_combined(base_path, raw_key="raw", mito_key="labels/mitochondria", cristae_key="labels/cristae"):
    h5_files = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    for path in tqdm(h5_files):
        cristae = None
        base, file_name = os.path.split(path)
        fname, _ = os.path.splitext(file_name)
        
        cristae = _read_h5(path, cristae_key)
        if cristae is None:
            continue
        raw = _read_h5(path, raw_key)
        mitos = _read_h5(path, mito_key)
        mitos = np.where(mitos > 0, 1, 0)
        new_path = os.path.join(base, fname + "_combined.h5")

        combined = np.stack([raw, mitos], axis=0)

        _write_h5(new_path, "raw_mitos_combined", combined)
        _write_h5(new_path, "labels/cristae", cristae)


def mitos_to_mask(base_path, combined_key="raw_mitos_combined"):
    combined_h5_files = sorted(glob(os.path.join(base_path, "**", "*combined.h5"), recursive=True))
    for path in tqdm(combined_h5_files):
        combined = _read_h5(path, combined_key)
        print("combined shape: ", combined.shape)
        raw, mitos = combined[0], combined[1]
        _write_h5(path, combined_key, np.stack([raw, np.where(mitos > 0, 1, 0)], axis=0))


def main():
    
    # Example usage
    base_path = "/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi/"
    #base_path = "/home/freckmann15/data/mitochondria/corrected_mitos/"
    cristae_key = "labels/cristae"
    mito_key = "labels/mitochondria"
    raw_key = "raw"

    create_combined(base_path)


if __name__ == "__main__":
    main()
