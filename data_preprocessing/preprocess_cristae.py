import h5py
import os
from glob import glob
from tqdm import tqdm
from skimage import measure
import numpy as np
from synapse import util


def _read_h5(path, key, scale_factor=1):
    with h5py.File(path, "r") as f:
        try:
            if scale_factor != 1:
                print(f"{key} data shape", f[key].shape)
            image = f[key][::scale_factor, ::scale_factor, ::scale_factor]
            if scale_factor != 1:
                print(f"{key} data shape after downsampling", image.shape)

        except KeyError:
            print(f"Error: {key} dataset not found in {path}")
            return None  # Indicate error

        return image


def _write_h5(path, key, image):
    with h5py.File(path, "a") as f:
        f.create_dataset(key, data=image, dtype=image.dtype)


def create_combined(base_path, export_path, 
                    raw_key="raw", mito_key="labels/mitochondria", cristae_key="labels/cristae",
                    scale_factor=1):
    h5_files = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    for path in tqdm(h5_files):
        cristae = None
        base, file_name = os.path.split(path)
        fname, _ = os.path.splitext(file_name)
        
        ds_keys = util.get_all_datasets(path)
        if cristae_key not in ds_keys or raw_key not in ds_keys:
            continue
        
        export_file_name, rel_path = util.get_filename_and_inter_dirs(path, base_path)
        util.create_directories_if_not_exists(export_path, rel_path)
        export_file_path = os.path.join(export_path, rel_path, export_file_name + "_combined.h5")
        
        cristae = _read_h5(path, cristae_key, scale_factor=scale_factor)
        if cristae is None:
            continue
        raw = _read_h5(path, raw_key, scale_factor=scale_factor)
        mitos = _read_h5(path, mito_key, scale_factor=scale_factor)
        mitos = np.where(mitos > 0, 1, 0)

        combined = np.stack([raw, mitos], axis=0)

        _write_h5(export_file_path, "raw_mitos_combined", combined)
        _write_h5(export_file_path, "labels/cristae", cristae)


def mitos_to_mask(base_path, combined_key="raw_mitos_combined"):
    combined_h5_files = sorted(glob(os.path.join(base_path, "**", "*combined.h5"), recursive=True))
    for path in tqdm(combined_h5_files):
        combined = _read_h5(path, combined_key)
        print("combined shape: ", combined.shape)
        raw, mitos = combined[0], combined[1]
        _write_h5(path, combined_key, np.stack([raw, np.where(mitos > 0, 1, 0)], axis=0))


def main():
    
    # Example usage
    base_path = "/scratch-grete/projects/nim00007/data/mitochondria/cooper/mito_tomo" # fidi , example_cristzae, mito_tomo
    export_path = "/scratch-grete/projects/nim00007/data/mitochondria/cooper/raw_mito_combined_s2/"
    # base_path = "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/extracted"
    # export_path = "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined"
    #base_path = "/home/freckmann15/data/mitochondria/corrected_mitos/"
    cristae_key = "labels/cristae"
    mito_key = "labels/mitochondria"
    raw_key = "raw"

    create_combined(base_path, export_path=export_path, scale_factor=2)


if __name__ == "__main__":
    main()
