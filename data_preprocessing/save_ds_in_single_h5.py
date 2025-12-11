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
    parser.add_argument("--second_base_path", "-b2",  type=str, required=True, help="Path to the root data directory")
    parser.add_argument("--export_path", "-e", type=str, required=True, help="Path to the export directory")
    parser.add_argument("--scale_factor", "-s", type=int, default=1, help="Scale factor for the image")
    parser.add_argument("--import_file_extension", "-ife", type=str, default=".h5", help="File extension to read data")
    parser.add_argument("--second_import_file_extension", "-ife2", type=str, default=".h5", help="File extension to read data")
    parser.add_argument("--export_file_extension", "-efe", type=str, default=".h5", help="File extension to export data")
    args = parser.parse_args()
    scale = args.scale_factor
    ife = args.import_file_extension
    ife2 = args.second_import_file_extension
    efe = args.export_file_extension

    # paths = sorted(glob(os.path.join(args.base_path, "**", f"*{ife}"), recursive=True))
    paths = io.load_file_paths(args.base_path, ext=ife)
    # paths_2 = sorted(glob(os.path.join(args.second_base_path, "**", f"*{ife2}"), recursive=True))
    paths_2 = io.load_file_paths(args.second_base_path, ext=ife2)
    # filter all raw files
    # paths = [path for path in paths if "embedding" not in path and "mask" not in path]

    for path, path2 in tqdm(zip(paths, paths_2), total=len(paths)):
        path2 = util.find_label_file(path, paths_2)
        if path2 is None:
            print("Could not find label file for", path)
            continue
        export_file_name, rel_path = get_filename_and_inter_dirs(path, args.base_path)
        export_file_name = export_file_name.replace("mitotomo-net32-lr1e-4-bs8-ps32x256x256-s4_sd4_bt015_with_pred_ts_z32_y256_x256_halo_z8_y64_x64_", "").replace(
            "_s2_refined", ""
        )
        create_directories_if_not_exists(args.export_path, rel_path)
        if scale > 1:
            export_file_path = os.path.join(args.export_path, rel_path, export_file_name + f"_s{scale}{efe}")
        else:
            export_file_path = os.path.join(args.export_path, rel_path, export_file_name + f"{efe}")
        if os.path.exists(export_file_path):
            print("File already exists:", export_file_path)
            continue
        data = {}
        data["raw"] = util.read_data(path, scale=scale)["raw"]
        if ife2 == ".h5":
            tmp = util.read_data(path2, scale=scale)
            tmp = tmp.pop("labels/mitochondria", None)
        else:
            tmp = tifffile.imread(path2)
        # remove_small_objects(
        #     data2.pop("label", None),
        #     min_size=500
        # )
        data2 = {}
        print("shapes data and data2", data["raw"].shape, tmp.shape)
        data2["labels/mitochondria"] = resize(tmp, data["raw"].shape, preserve_range=True, order=0, anti_aliasing=False).astype(tmp.dtype)
        data.update(data2)
        util.export_data(export_file_path, data, voxel_size=[0.025, 0.005, 0.005])  # [8.694*2, 8.694*2, 8.694*2])


if __name__ == "__main__":
    main()
