import argparse
import os
from glob import glob
from tqdm import tqdm
import mrcfile
import numpy as np
from collections import Counter, defaultdict
import h5py


def collect_dataset_shapes(h5file, group=None, prefix=""):
    if group is None:
        group = h5file
    shapes = []
    for key in group:
        full_key = f"{prefix}/{key}" if prefix else key
        try:
            item = group[key]
        except KeyError:
            continue
        if isinstance(item, h5py.Dataset):
            shapes.append(item.shape)
        elif isinstance(item, h5py.Group):
            shapes.extend(collect_dataset_shapes(h5file, item, full_key))
    return shapes

def percentage_files_with_inconsistent_shapes(file_paths):
    num_total = 0
    num_inconsistent = 0

    for path in tqdm(file_paths):
        try:
            with h5py.File(path, "r") as f:
                shapes = collect_dataset_shapes(f)
                if not shapes:
                    continue  # Skip files with no datasets
                num_total += 1
                first_shape = shapes[0]
                if not all(shape == first_shape for shape in shapes):
                    num_inconsistent += 1
        except Exception as e:
            print(f"[Error] {path}: {e}")

    percentage = (num_inconsistent / num_total) * 100 if num_total > 0 else 0
    print(f"\n{num_inconsistent}/{num_total} files have internally inconsistent dataset shapes.")
    print(f"That's {percentage:.2f}% inconsistent files.")
    return percentage
# def check_hdf5_dataset_shapes(paths):
#     stats = {}
#     all_shapes = defaultdict(list)
#     mismatched_files = []

#     for path in tqdm(paths):
#         with h5py.File(path, "r") as f:
#             shapes = collect_dataset_shapes(f)
#             stats[path] = shapes

#             # Save all shapes for each dataset key (full paths)
#             shape_list = list(shapes.values())
#             for k, v in shapes.items():
#                 all_shapes[k].append(v)

#             # Check if all datasets in this file have the same shape
#             if len(set(shape_list)) > 1:
#                 mismatched_files.append(path)

#     # Print summary
#     print("\n=== Summary ===")
#     print(f"Total files checked: {len(paths)}")
#     print(f"Files with mismatched dataset shapes: {len(mismatched_files)}")
#     if mismatched_files:
#         print("Example mismatched file:", mismatched_files[0])

#     print("\nShape statistics per dataset (across files):")
#     for k, shape_list in all_shapes.items():
#         unique_shapes, counts = np.unique(shape_list, return_counts=True, axis=0)
#         # print(f"- {k}:")
#         # for shape, count in zip(unique_shapes, counts):
#         #     print(f"    shape {shape} occurs {count} times")

#     return stats, mismatched_files

def main(visualize=False):
    parser = argparse.ArgumentParser()
    # /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer
    parser.add_argument("--base_path", "-p",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/mito_tomo", help="Path to the root data directory")
    parser.add_argument("--base_path2", "-p2",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2", help="Path to the root data directory")
    #parser.add_argument("--save_dir", type=str, default="", help="Path to save the data to")
    args = parser.parse_args()
    # print(args.base_path, "\n", args.base_path2)

    b1_paths = sorted(glob(os.path.join(args.base_path, "**", "*.h5"), recursive=True))#, reverse=True)
    b2_paths = sorted(glob(os.path.join(args.base_path2, "**", "*.h5"), recursive=True))#, reverse=True)
    
    stats = percentage_files_with_inconsistent_shapes(b1_paths)
    print("stats", stats)
    # print("mismatched_files", mismatched_files)
    # stats = {}
    # for path in tqdm(b1_paths):
    #         stats = {}
    #         all_shapes = defaultdict(list)
    #         mismatched_files = []

    #         for path in tqdm(b1_paths):
    #             stats[path] = {}
    #             with h5py.File(path, "r") as f:
    #                 shapes = []
    #                 for k in f.keys():
    #                     shape = f[k].shape
    #                     stats[path][k] = shape
    #                     all_shapes[k].append(shape)
    #                     shapes.append(shape)

    #                 # Check if all shapes in this file are the same
    #                 if len(set(shapes)) > 1:
    #                     mismatched_files.append(path)
    # # Print summary
    # print("\n=== Summary ===")
    # print(f"Total files checked: {len(b1_paths)}")
    # print(f"Files with mismatched dataset shapes: {len(mismatched_files)}")
    # if mismatched_files:
    #     print("Example mismatched file:", mismatched_files[0])
    
    # # Dataset-wise shape counts
    # print("\nShape statistics per dataset:")
    # for k, shape_list in all_shapes.items():
    #     unique_shapes, counts = np.unique(shape_list, return_counts=True, axis=0)
    #     print(f"- {k}:")
    #     for shape, count in zip(unique_shapes, counts):
    #         print(f"    shape {shape} occurs {count} times")

    
    # print("len(b1_paths)", len(b1_paths))
    # print("len(b2_paths)", len(b2_paths))
    # for mod_path, mrc_path in tqdm(zip(mod_paths, mrc_paths)):
    # vox_sizes = []
    # b1_paths.extend(b2_paths)
    # filenames = [path.split("/")[-1] for path in b1_paths]  # Extract filenames
    # duplicates = [name for name, count in Counter(filenames).items() if count > 1]
    # print("Duplicate filenames:", duplicates)

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