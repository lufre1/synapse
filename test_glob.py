import argparse
import csv
import os
from glob import glob
from tqdm import tqdm
import mrcfile
import numpy as np
from collections import Counter, defaultdict
import h5py
from elf.io import open_file
import synapse.cellmap_util as cutil


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


def write_h5_paths_to_csv(paths, output_path="h5_paths_with_mitos_new.csv", ):
    h5_files = paths

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["h5_file_path"])
        for h5 in h5_files:
            full_path = os.path.abspath(h5)
            writer.writerow([full_path])

    print(f"Wrote {len(h5_files)} paths to {output_path}")


def find_files_with_exact_mito_key(h5_paths):
    """
    Returns a list of .h5 file paths that contain a dataset with the exact key 'mito'.
    """
    matching_files = []

    for path in h5_paths:
        try:
            with h5py.File(path, 'r') as f:
                found = False

                def visitor(name, obj):
                    nonlocal found
                    if isinstance(obj, h5py.Dataset):
                        key_name = name.split('/')[-1]
                        if key_name == "mito":
                            found = True

                f.visititems(visitor)

                if found:
                    matching_files.append(path)

        except Exception as e:
            print(f"Could not read {path}: {e}")

    return matching_files


def print_shape_stats(shapes):
    """
    Prints the mean of the (z, y, x) shapes and the prevalence of each unique shape.
    
    Parameters:
        shapes (list of tuple): List of (z, y, x) tuples.
    """
    if not shapes:
        print("No shapes provided.")
        return

    # Convert to NumPy array for easier computation
    shape_array = np.array(shapes)  # shape: (N, 3)
    mean_shape = shape_array.mean(axis=0)

    print(f"Mean shape (z, y, x): ({mean_shape[0]:.2f}, {mean_shape[1]:.2f}, {mean_shape[2]:.2f})")

    # Count occurrences of each unique shape
    counter = Counter(shapes)
    print("\nPrevalence of unique shapes:")
    for shape, count in counter.most_common():
        print(f"  Shape {shape}: {count} times")


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
    
    # stats = percentage_files_with_inconsistent_shapes(b1_paths)
    # print("stats", stats)
    paths = find_files_with_exact_mito_key(b1_paths)
    # paths = cutil.get_cellmap_mito_paths()
    #write_h5_paths_to_csv(paths, output_path="h5_paths_mitos.csv")
    print(len(paths))
    shapes = []
    for path in paths:
        print(path)
        with open_file(path, mode="r") as f:
            shapes.append(f["label_crop/mito"].shape)
            print("np.unique", np.unique(f["label_crop/mito"], return_counts=True))

    print(len(shapes))
    print_shape_stats(shapes)


if __name__ == "__main__":
    main()