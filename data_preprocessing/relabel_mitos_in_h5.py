import h5py
import os
from glob import glob
import numpy as np
from tqdm import tqdm
from skimage import measure
from skimage.transform import rescale, resize
import argparse
import mrcfile
from synapse.util import get_data_metadata
import napari
from elf.io import open_file
from elf.evaluation.matching import label_overlap, intersection_over_union
from elf.parallel import label as parallel_label
from skimage.segmentation import relabel_sequential
from skimage.morphology import binary_closing, remove_small_objects, label
import tifffile
import synapse.util as util
import synapse.io.util as io


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
    parser.add_argument("--base_path", "-b",  type=str, required=True, help="Path to the root data directory")
    parser.add_argument("--scale_factor", "-s", type=int, default=1, help="Scale factor for the image")
    parser.add_argument("--import_file_extension", "-ife", type=str, default=".h5", help="File extension to read data")
    args = parser.parse_args()
    ife = args.import_file_extension
    dataset_name = "labels/mitochondria"

    paths = io.load_file_paths(args.base_path, ext=ife)

    for path in tqdm(paths, total=len(paths)):
        with open_file(path, "a") as f:
            mitos = f[dataset_name][...]
            uniq = np.unique(mitos)
            print("mitos unique", uniq)
            mitos = remove_small_objects(mitos, min_size=200)
            mitos = label(mitos).astype(np.uint8)
            f[dataset_name][:] = mitos
            print("mitos unique updated", np.unique(mitos))
            print("mitochondria updated to file", path)
        # util.export_data(export_file_path, data, voxel_size=voxel_size)


if __name__ == "__main__":
    main()
