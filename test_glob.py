import argparse
import os
from glob import glob
from tqdm import tqdm
import mrcfile
import numpy as np
from collections import Counter


def main(visualize=False):
    parser = argparse.ArgumentParser()
    # /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer
    parser.add_argument("--base_path", "-p",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/mito_tomo", help="Path to the root data directory")
    parser.add_argument("--base_path2", "-p2",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2", help="Path to the root data directory")
    #parser.add_argument("--save_dir", type=str, default="", help="Path to save the data to")
    args = parser.parse_args()
    print(args.base_path, "\n", args.base_path2)

    b1_paths = sorted(glob(os.path.join(args.base_path, "**", "*.h5"), recursive=True))#, reverse=True)
    b2_paths = sorted(glob(os.path.join(args.base_path2, "**", "*.h5"), recursive=True))#, reverse=True)
    print("len(b1_paths)", len(b1_paths))
    print("len(b2_paths)", len(b2_paths))
    # for mod_path, mrc_path in tqdm(zip(mod_paths, mrc_paths)):
    vox_sizes = []
    b1_paths.extend(b2_paths)
    filenames = [path.split("/")[-1] for path in b1_paths]  # Extract filenames
    duplicates = [name for name, count in Counter(filenames).items() if count > 1]
    print("Duplicate filenames:", duplicates)

    # for path in tqdm(b1_paths):
        
        # with mrcfile.open(mrc_path) as mrc:
            # print(path)
            # print(mrc.voxel_size, "\n")
            # vox_sizes.append([mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z])
        # print("\n", mod_path, "\n", mrc_path, "\n")
    # use this for 06
    # print("average voxel size", np.mean(vox_sizes))


if __name__ == "__main__":
    main()