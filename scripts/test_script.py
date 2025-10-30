import argparse
from glob import glob
import os
from tqdm import tqdm
import synapse.util as util
from elf.io import open_file


def main(args):
    data_dir = args.data_dir
    data_dir2 = args.data_dir2
    data_dir3 = args.data_dir3

    print("dir1", data_dir)
    print("dir2", data_dir2)
    print("dir3", data_dir3)

    data_paths = util.get_data_paths(data_dir)
    # data_paths = util.get_wichmann_data()
    if data_dir2 is not None:
        data_paths2 = util.get_data_paths(data_dir2)
        data_paths.extend(data_paths2)
    if data_dir3 is not None:
        data_paths3 = util.get_data_paths(data_dir3)
        data_paths.extend(data_paths3)
    print("len datapaths aggregated", len(data_paths))
    for p in tqdm(data_paths):
        with open_file(p, mode="r") as f:
            try:
                f["labels/mitochondria"]
            except (KeyError, OSError) as e:
                print(f"Failed to load mitochondria labels from {p}: {e}")
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos", help="Path to the data directory")
    parser.add_argument("--data_dir2", type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2/", help="Path to a second data directory")
    parser.add_argument("--data_dir3", type=str, default="/mnt/lustre-grete/usr/u12103/mitochondria/cooper/fidi_2025/exported_to_hdf5_s2", help="Path to a third data directory")
    args = parser.parse_args()
    main(args)