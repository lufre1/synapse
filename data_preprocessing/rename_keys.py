import h5py
import os
from glob import glob
import numpy as np
from tqdm import tqdm
from skimage import measure
import argparse
# import numpy as np
import mrcfile


SAVE_DIR = "/home/freckmann15/data/mitochondria/corrected_mito_h5"


def rename_h5_key(file_path, old_key, new_key):
    """
    Renames a key within an h5 file.

    Args:
        file_path (str): Path to the h5 file.
        old_key (str): The old key name.
        new_key (str): The new key name.
    """

    with h5py.File(file_path, 'r+') as h5_file:
        if old_key in h5_file:
            if new_key in h5_file:
                print("New key already exists", new_key)
                return
            old_dataset = h5_file[old_key]
            h5_file.create_dataset(new_key, data=old_dataset[()], dtype=old_dataset.dtype)
            del h5_file[old_key]


def merge_keys(file_path, key1, key2):
    """
    Merge label datasets in an HDF5 file.
    The merged dataset will contain only zeros and ones, and the dataset with key2 is deleted after merging.

    Args:
        file_path (str): Path to the HDF5 file.
        key1 (str): Key of the first dataset to merge.
        key2 (str): Key of the second dataset to merge.
    """
    dataset_names = get_all_datasets(file_path)
    with h5py.File(file_path, 'r+') as hdf5_file:
        if key1 in dataset_names and key2 in dataset_names:
            
            # Load data from both keys
            data1 = hdf5_file[key1][:]
            data2 = hdf5_file[key2][:]

            # Ensure the merged data contains only 0s and 1s
            merged_data = ((data1 + data2) > 0).astype(np.uint8)

            # Write the merged data back to key1
            hdf5_file[key1][:] = merged_data

            # Delete the second dataset
            del hdf5_file[key2]
        elif key1 not in dataset_names and key2 in dataset_names:
            print(f"Only Dataset {key2} found in {file_path}, creating {key1}")
            data = hdf5_file[key2][:]
            hdf5_file.create_dataset(key1, data=data, dtype=data.dtype)
            del hdf5_file[key2]
        else:
            print(f"Dataset {key1} not found in {file_path}")
            return


def correct_mito_labels(file_path, key="labels/mitochondria"):
    with h5py.File(file_path, 'r+') as hdf5_file:
        if key in hdf5_file:
            mitochondria_data = hdf5_file[key][:]
            mitochondria_data[mitochondria_data != 0] = 1
            hdf5_file[key][:] = mitochondria_data


def convert_mask_to_labels(file_path, key):
    with h5py.File(file_path, 'r+') as hdf5_file:
        if key in hdf5_file:
            label_data = hdf5_file[key][:]
            label_data = measure.label(label_data)
            hdf5_file[key][:] = label_data


def process_h5_files(base_path, old_key, new_key):
    """
    Processes h5 files in a directory, renaming specified keys.

    Args:
        base_path (str): Path to the directory containing h5 files.
        old_key (str): The old key name.
        new_key (str): The new key name.
    """

    h5_files = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    for h5_file in tqdm(h5_files):
        file_path = os.path.join(base_path, h5_file)
        # rename_h5_key(file_path, old_key, new_key)
        convert_mask_to_labels(file_path, "labels/mitochondria")


def add_raw_to_h5(file_path, label_path):
    with mrcfile.open(file_path, "r") as f:
        raw = f.data
    with h5py.File(label_path, 'r+') as hdf5_file:
        if "raw" not in hdf5_file:
            hdf5_file.create_dataset('raw', data=raw)


def separate_cristae(h5_file_path):
    """Separates cristae from mitochondria in an HDF5 file.

    Args:
        h5_file_path: Path to the HDF5 file.
    """
    scale_factor = 1
    # cristae = None
    with h5py.File(h5_file_path, 'r+') as h5f:
        # raw = h5f['raw'][:, ::scale_factor, ::scale_factor]
        mitochondria = h5f['labels/mitochondria'][:, ::scale_factor, ::scale_factor]
        # cristae = (mitochondria == 2).astype(np.uint8)
        mitochondria[mitochondria == 2] = 1
        # print("any other value than 0 and 1 in mitochondria?", len(np.unique(mitochondria)))
        mito_ds = h5f['labels/mitochondria']
        mito_ds[...] = mitochondria


def get_all_datasets(file_path):
    dataset_names = []

    def visit_func(name, obj):
        if isinstance(obj, h5py.Dataset):
            dataset_names.append(name)

    with h5py.File(file_path, 'r') as hdf5_file:
        hdf5_file.visititems(visit_func)

    return dataset_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/home/freckmann15/data/mitochondria/wichmann/", help="Path to the root data directory")
    args = parser.parse_args()
    label_base_path = args.base_path

    h5_paths = sorted(glob(os.path.join(label_base_path, "**", "*.h5"), recursive=True))

    for path in tqdm(h5_paths):
        print(path)
        # with h5py.File(path, 'r') as hdf5_file:
        dataset_names = get_all_datasets(path)
        # print(dataset_names)
        for name in dataset_names:
            if "IM" in name:
                dataset_names = get_all_datasets(path)
                print(dataset_names)
                merge_keys(path, "labels/cristae", name)
        # Verify the file after processing

        # remaining_datasets = get_all_datasets(path)
        # print(f"Remaining datasets in {path}: {remaining_datasets}\n")


if __name__ == "__main__":
    main()