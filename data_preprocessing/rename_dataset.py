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


def check_labels(file_path, key):
    print("file path", file_path)
    print("with key", key)
    with h5py.File(file_path, 'r') as hdf5_file:
        if key in hdf5_file:
            label_data = hdf5_file[key][:]
            print(np.unique(label_data))
    

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
        # convert_mask_to_labels(file_path, "labels/mitochondria")
        # convert_mask_to_labels(file_path, "labels/cristae")
        check_labels(file_path, "labels/mitochondria")
        convert_mask_to_labels(file_path, "labels/mitochondria")
        check_labels(file_path, "labels/mitochondria")
        #check_labels(file_path, "labels/cristae")


def main():
    
    # Example usage
    base_path = "/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi/"
    base_path = "/home/freckmann15/data/mitochondria/cooper/new_mitos/"
    old_key = "labels/mitchondria"
    new_key = "labels/mitochondria"  

    process_h5_files(base_path, old_key, new_key)


if __name__ == "__main__":
    main()