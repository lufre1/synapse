import argparse
from glob import glob
import os
from tqdm import tqdm
import synapse.util as util
from elf.io import open_file
import elf.parallel as parallel
import h5py
import numpy as np


def process_mitos_and_cristae(data, bs=(32, 128, 128)):
    # get the data 
    for key, val in data.items():
        if "mito" in key:
            mito_ids = np.unique(val)
            mito_ids = mito_ids[mito_ids != 0]
            if 1 in mito_ids and len(mito_ids) == 1:
                mitos = parallel.label(val, block_shape=bs)
        elif "cristae" in key:
            cristae = val
        elif "raw" in key:
            raw = val
    # search for mitos that have cristae in them
    # mito_ids = np.unique(mitos)
    # mito_ids = mito_ids[mito_ids != 0]
    mitos_with_cristae = np.zeros_like(mitos, dtype=np.uint8)
    for mito_id in mito_ids:
        # Create mask for current mito instance
        mito_mask = (mitos == mito_id)
        # Check if any cristae pixels overlap with this mito instance
        if np.any(cristae[mito_mask]):
            # Mark entire mito instance as having cristae
            mitos_with_cristae[mito_mask] = 1
        else:
            # Mark entire mito instance as not having cristae
            mitos_with_cristae[mito_mask] = 2
            print("DEBUG - Found mito without cristae")

    # combine mitos and cristae
    combined = np.stack([raw, mitos_with_cristae], axis=0)
    return {"raw_mitos_combined": combined, "labels/cristae": cristae}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/wichmann/", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e", type=str, default="/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/", help="Path to the export directory")
    args = parser.parse_args()
    base_path = args.base_path
    export_path = args.export_path
    util.create_directories_if_not_exists(export_path, "")

    if os.path.isdir(base_path):
        paths = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    elif os.path.isfile(base_path):
        paths = [base_path]
    else:
        raise ValueError(f"Invalid base path: {base_path}")
    paths = [p for p in paths if "_combined.h5" not in p]

    for path in tqdm(paths, desc="Processing files..."):
        process_file = False
        output_path = os.path.join(export_path, os.path.basename(path).replace(".h5", "_combined.h5"))
        if os.path.exists(output_path):
            print("output path already exists:", output_path)
            continue
        for k in util.get_all_datasets(path):
            if "cristae" in k:
                process_file = True
                break
        if not process_file:
            print(f"Skipping {path} because no cristae dataset found.")
            continue
        data = {}
        data = util.read_data(path)
        export_data = None
        try:
            export_data = process_mitos_and_cristae(data)
        except Exception as e:
            print(f"Error processing file {path}: {e}")
        # print("read voxel size", util.read_voxel_size_h5(path))
        if export_data:
            util.export_data(output_path, export_data, voxel_size=util.read_voxel_size_h5(path))


if __name__ == "__main__":
    main()