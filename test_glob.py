import argparse
import os
from glob import glob
from tqdm import tqdm
import mrcfile
import numpy as np


def main(visualize=False):
    parser = argparse.ArgumentParser()
    # /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer
    parser.add_argument("--base_path", "-b",  type=str, default="/home/freckmann15/data/mitochondria/cooper/new_mitos", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e",  type=str, default="/home/freckmann15/data/mitochondria/cooper/exported_mitos", help="Path to the root data directory")
    #parser.add_argument("--save_dir", type=str, default="", help="Path to save the data to")
    args = parser.parse_args()
    print(args.base_path)

    mod_paths = sorted(glob(os.path.join(args.base_path, "**", "*.mod"), recursive=True))#, reverse=True)
    mrc_paths = sorted(glob(os.path.join(args.base_path, "**", "*.mrc"), recursive=True))#, reverse=True)
    print("len(mod_paths)", len(mod_paths))
    print("len(mrc_paths)", len(mrc_paths))
    # for mod_path, mrc_path in tqdm(zip(mod_paths, mrc_paths)):
    vox_sizes = []
    for mrc_path in tqdm(mrc_paths):
        with mrcfile.open(mrc_path) as mrc:
            print(mrc_path)
            print(mrc.voxel_size, "\n")
            vox_sizes.append([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
        # print("\n", mod_path, "\n", mrc_path, "\n")
    # use this for 06
    print("average voxel size", np.mean(vox_sizes))


if __name__ == "__main__":
    main()