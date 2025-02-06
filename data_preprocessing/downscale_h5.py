import h5py
import os
from glob import glob
import numpy as np
from tqdm import tqdm
from skimage import measure
import argparse
import mrcfile
from synapse.util import get_data_metadata
from synapse.h5_util import read_h5, get_all_keys_from_h5
import napari
from elf.io import open_file
from elf.evaluation.matching import label_overlap, intersection_over_union
from elf.parallel import label as parallel_label
from skimage.segmentation import relabel_sequential
from skimage.morphology import binary_closing, remove_small_objects, label
import tifffile


def export_to_h5(data, export_path):
    with h5py.File(export_path, 'x') as h5f:
        for key in data.keys():
            h5f.create_dataset(key, data=data[key], compression="gzip")
    print("exported to", export_path)


def get_filename_and_inter_dirs(file_path, base_path):
    # Extract the base name (filename with extension)
    base_name = os.path.basename(file_path)
    # Split the base name into name and extension to get the filename
    file_name = os.path.splitext(base_name)[0]
    # Get the relative path of file_path from base_path
    relative_path = os.path.relpath(file_path, base_path)
    # Get the intermediate directories by removing the filename from the relative path
    inter_dirs = os.path.dirname(relative_path)
    return file_name, inter_dirs


def create_directories_if_not_exists(base_path, inter_dirs):
    # Construct the full path from base_path and inter_dirs
    full_path = os.path.join(base_path, inter_dirs)
    
    # Check if the path exists
    if not os.path.exists(full_path):
        # If it doesn't exist, create the directories
        os.makedirs(full_path)
        print(f"\nCreated directories: {full_path}")
    else:
        print(f"\nDirectories already exist: {full_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e", type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/s2/", help="Path to the export directory")
    parser.add_argument("--scale_factor", "-s", type=int, default=2, help="Scale factor for the image")
    args = parser.parse_args()

    h5_paths = sorted(glob(os.path.join(args.base_path, "**", "*.h5"), recursive=True))

    for path in tqdm(h5_paths):
        export_file_name, rel_path = get_filename_and_inter_dirs(path, args.base_path)
        create_directories_if_not_exists(args.export_path, rel_path)
        export_file_path = os.path.join(args.export_path, rel_path, export_file_name + "_s2.h5")
        if os.path.exists(export_file_path):
            print("File already exists:", export_file_path)
            continue
        keys = get_all_keys_from_h5(path)
        data = {}
        for key in keys:
            data[key] = read_h5(path, key, args.scale_factor)
        export_to_h5(data, export_file_path)


if __name__ == "__main__":
    main()
