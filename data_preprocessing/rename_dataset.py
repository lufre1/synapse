import h5py
import os
from glob import glob
from tqdm import tqdm
from skimage import measure
import numpy as np


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
            old_dataset = h5_file[old_key]
            h5_file.create_dataset(new_key, data=old_dataset[()], dtype=old_dataset.dtype)
            del h5_file[old_key]


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


def get_all_keys_from_h5(file_path):
    keys = []
    with h5py.File(file_path, 'r') as h5file:
        def collect_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.append(name)  # Add each key (path) to the list
        h5file.visititems(collect_keys)  # Visit all groups and datasets
    return keys


def check_labels(file_path, key):
    print("file path", file_path)
    print("with key", key)
    with h5py.File(file_path, 'r') as hdf5_file:
        if key in hdf5_file:
            label_data = hdf5_file[key][:]
            print(np.unique(label_data))


def filter_list_by_substring(input_list, substring):
    return [element for element in input_list if substring not in element]


def add_suffix_to_filename(file_path, suffix="_downscaled"):
    directory, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}{suffix}{ext}"
    return os.path.join(directory, new_filename)


def downscale(file_path, factor=2):
    keys = get_all_keys_from_h5(file_path)
    with h5py.File(file_path, 'r+') as hdf5_file:
        for key in keys:
            data = hdf5_file[key][:]
            original_shape = data.shape
            print(f"Before downscaling with factor {factor} data shape {original_shape}")
            downscaled_data = data[::factor, ::factor, ::factor]
            new_shape = downscaled_data.shape
            # Resize the dataset in the file to match the new downscaled shape
            # hdf5_file[key].resize(new_shape)
            
            # Assign the downscaled data back to the HDF5 file
            # hdf5_file[key][:]= downscaled_data
            print("After downscaling with factor", factor, "data shape", downscaled_data.shape)
            with h5py.File(add_suffix_to_filename(file_path), 'a') as new_hdf5_file:
                new_hdf5_file[key] = downscaled_data


def process_h5_files(base_path, old_key, new_key):
    """
    Processes h5 files in a directory, renaming specified keys.

    Args:
        base_path (str): Path to the directory containing h5 files.
        old_key (str): The old key name.
        new_key (str): The new key name.
    """

    h5_files = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    filtered_h5_files = filter_list_by_substring(h5_files, "combined")
    print("h5 files", filtered_h5_files)
    for h5_file in tqdm(filtered_h5_files):
        file_path = os.path.join(base_path, h5_file)
        # rename_h5_key(file_path, old_key, new_key)
        # convert_mask_to_labels(file_path, "labels/mitochondria")
        # convert_mask_to_labels(file_path, "labels/cristae")
        # check_labels(file_path, "labels/mitochondria")
        downscale(file_path, factor=2)
        # convert_mask_to_labels(file_path, "labels/mitochondria")
        #check_labels(file_path, "labels/mitochondria")
        #check_labels(file_path, "labels/cristae")


def main():
    
    # Example usage
    base_path = "/home/freckmann15/data/mitochondria/cooper/test_mitos"
    #base_path = "/home/freckmann15/data/mitochondria/cooper/new_mitos/"
    old_key = "labels/mitchondria"
    new_key = "labels/mitochondria"

    process_h5_files(base_path, old_key, new_key)


if __name__ == "__main__":
    main()