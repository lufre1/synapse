import os
import argparse
import h5py
from collections import Counter, defaultdict
import numpy as np


def print_dataset_shapes(h5file, path='/', shapes=None, dimsizes=None):
    """Recursively print all datasets and their shapes in an HDF5 file.
    Optionally updates a shape counter and dimension size lists."""
    for key in h5file[path]:
        item = h5file[path + key]
        if isinstance(item, h5py.Group):
            print_dataset_shapes(h5file, path + key + '/', shapes, dimsizes)
        elif isinstance(item, h5py.Dataset):
            print(f"  Dataset: {path + key}, shape: {item.shape}, dtype: {item.dtype}")
            if shapes is not None:
                shapes[item.shape] += 1
            if dimsizes is not None:
                for idx, v in enumerate(item.shape):
                    dimsizes[idx].append(v)


def scan_hdf5_shapes(rootdir, stats=False):
    """Walk through rootdir and print shapes of all datasets in HDF5 files.
    Optionally print statistics about the dataset shapes."""
    shape_counter = Counter() if stats else None
    dimsizes = defaultdict(list) if stats else None
    file_count = data_count = 0

    for dirpath, _, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.endswith(('.h5', '.hdf5')):
                file_count += 1
                filepath = os.path.join(dirpath, filename)
                print(f"\nHDF5 file: {filepath}")
                try:
                    with h5py.File(filepath, 'r') as h5file:
                        prev = sum(shape_counter.values()) if stats else 0
                        print_dataset_shapes(h5file, shapes=shape_counter, dimsizes=dimsizes)
                        if stats:
                            data_count += sum(shape_counter.values()) - prev
                except Exception as e:
                    print(f"  Could not read file: {e}")

    if stats:
        print("\n===== Statistics =====")
        print(f"Files checked: {file_count}")
        print(f"Datasets found: {data_count}")
        print(f"Unique shapes: {len(shape_counter)}")
        for shape, count in shape_counter.items():
            print(f"  Shape {shape}: {count} dataset(s)")
        # Per-dimension stats:
        print("\nPer-dimension statistics:")
        for dim, values in dimsizes.items():
            arr = np.array(values)
            print(f"  Dimension {dim}: min={arr.min()}, max={arr.max()}, mean={arr.mean():.2f}, median={np.median(arr)}")
            

def main():
    parser = argparse.ArgumentParser(
        description="Recursively scan all HDF5 files in a directory and print their dataset shapes."
    )
    parser.add_argument('-i', "--input_path", type=str, required=True,
                        help="Path to directory containing HDF5 files")
    parser.add_argument('--stats', "-s", action='store_true',
                        help="Print statistics about dataset shapes and element counts")
    args = parser.parse_args()
    scan_hdf5_shapes(args.input_path, stats=args.stats)


if __name__ == "__main__":
    main()