

import argparse
from glob import glob
import os
import mrcfile
import numpy as np
from tqdm import tqdm


def inspect_voxel_sizes(mrc_paths):
    voxel_sizes = []

    for mrc_path in tqdm(mrc_paths):
        try:
            with mrcfile.open(mrc_path) as mrc:
                voxel_size = (mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z)
                voxel_sizes.append(voxel_size)
        except Exception as e:
            print(f"Failed to read MRC file (path: {mrc_path}) attributes: {e}")

    # Convert to NumPy array for calculations
    voxel_sizes = np.array(voxel_sizes)

    if voxel_sizes.size > 0:
        print("Voxel size statistics:")
        print(f"Mean: {np.mean(voxel_sizes, axis=0)}")
        print(f"Median: {np.median(voxel_sizes, axis=0)}")
        print(f"Standard Deviation: {np.std(voxel_sizes, axis=0)}")
        
        unique_voxel_sizes, counts = np.unique(voxel_sizes, axis=0, return_counts=True)
        print("Unique voxel sizes with prevalences:")
        for voxel_size, count in zip(unique_voxel_sizes, counts):
            print(f"Voxel size: {voxel_size}, Prevalence: {count}")
    else:
        print("No valid voxel sizes found.")


def main():
    parser = argparse.ArgumentParser()
    # /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer
    parser.add_argument("--base_path", "-b",  type=str, default="/home/freckmann15/data/mitochondria/cooper/new_mitos", help="Path to the root data directory")
    args = parser.parse_args()
    
    mrc_paths = sorted(glob(os.path.join(args.base_path, "**", "*.mrc"), recursive=True))#, reverse=True)
    rec_paths = sorted(glob(os.path.join(args.base_path, "**", "*.rec"), recursive=True))
    mod_paths = sorted(glob(os.path.join(args.base_path, "**", "*.mod"), recursive=True))
    mrc_paths.extend(rec_paths)
    
    # inspect_voxel_sizes(mrc_paths)
    
    
    # voxel_sizes = []
    # for mrc_path in tqdm(mrc_paths):
    #     # print_all_mrc_attributes(mrc_path)
    #     try:
    #         with mrcfile.open(mrc_path) as mrc:
    #             # print(mrc_path)
    #             # print(mrc.voxel_size, "\n")
    #             # print(mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z)
    #             voxel_size = (mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z)
    #             voxel_sizes.append(voxel_size)
    #     except Exception as e:
    #         print(f"Failed to read MRC file (path: {mrc_path}) attributes: {e}")
    #     # print("\n", mod_path, "\n", mrc_path, "\n")
    
    # # Convert to NumPy array for calculations
    # voxel_sizes = np.array(voxel_sizes)
    # unique_voxel_sizes, counts = np.unique(voxel_sizes, axis=0, return_counts=True)
    # if voxel_sizes.size > 0:
    #     print("Voxel size statistics:")
    #     print(f"Mean: {np.mean(voxel_sizes, axis=0)}")
    #     print(f"Median: {np.median(voxel_sizes, axis=0)}")
    #     print(f"Standard Deviation: {np.std(voxel_sizes, axis=0)}")
    #     for voxel_size, count in zip(unique_voxel_sizes, counts):
    #         print(f"Voxel size: {voxel_size}, Prevalence: {count}")
    # else:
    #     print("No valid voxel sizes found.")
        

if __name__ == "__main__":
    main()