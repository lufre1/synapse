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


# def add_suffix_to_filename(file_path, suffix="_downscaled"):
#     directory, filename = os.path.split(file_path)
#     name, ext = os.path.splitext(filename)
#     new_filename = f"{name}{suffix}{ext}"
#     return os.path.join(directory, new_filename)


# def downscale(file_path, factor=2):
#     keys = get_all_keys_from_h5(file_path)
#     with h5py.File(file_path, 'r+') as hdf5_file:
#         for key in keys:
#             data = hdf5_file[key][:]
#             original_shape = data.shape
#             print(f"Before downscaling with factor {factor} data shape {original_shape}")
#             downscaled_data = data[::factor, ::factor, ::factor]
#             new_shape = downscaled_data.shape
#             # Resize the dataset in the file to match the new downscaled shape
#             # hdf5_file[key].resize(new_shape)
            
#             # Assign the downscaled data back to the HDF5 file
#             # hdf5_file[key][:]= downscaled_data
#             print("After downscaling with factor", factor, "data shape", downscaled_data.shape)
#             with h5py.File(add_suffix_to_filename(file_path), 'a') as new_hdf5_file:
#                 new_hdf5_file[key] = downscaled_data
def add_suffix_to_filename(file_path, output_base_path, base_path):
    """
    Creates a new file path by appending the suffix and preserving the original directory structure.
    
    Args:
        file_path (str): Original file path.
        output_base_path (str): Base path for saving the new files.
    
    Returns:
        str: New file path with suffix and directory structure preserved.
    """
    relative_path = os.path.relpath(file_path, start=base_path)
    new_file_path = os.path.join(output_base_path, relative_path)
    new_file_dir = os.path.dirname(new_file_path)
    print("relative path", relative_path)
    print("new file path", new_file_path)
    print("new file dir", new_file_dir)
    
    # Ensure the directory exists
    os.makedirs(new_file_dir, exist_ok=True)
    
    return new_file_path


def downscale(file_path, output_base_path, base_path, factor=2):
    keys = get_all_keys_from_h5(file_path)
    with h5py.File(file_path, 'r') as hdf5_file:
        for key in keys:
            data = hdf5_file[key][:]
            original_shape = data.shape
            print(f"Before downscaling with factor {factor}, data shape {original_shape}")
            downscaled_data = data[::factor, ::factor, ::factor]
            print(f"After downscaling with factor {factor}, data shape {downscaled_data.shape}")

            # Create the new file path
            new_file_path = add_suffix_to_filename(file_path, output_base_path, base_path)
            
            # Write downscaled data to the new file
            with h5py.File(new_file_path, 'a') as new_hdf5_file:
                new_hdf5_file.create_dataset(key, data=downscaled_data, compression="gzip")
                # new_hdf5_file[key] = downscaled_data
                print(f"Data saved to: {new_file_path}")


def process_h5_files(base_path, output_base_path):
    """
    Processes h5 files in a directory, renaming specified keys.

    Args:
        base_path (str): Path to the directory containing h5 files.
        old_key (str): The old key name.
        new_key (str): The new key name.
    """

    h5_files = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    #filtered_h5_files = filter_list_by_substring(h5_files, "combined")
    filtered_h5_files = h5_files
    print("h5 files", filtered_h5_files)
    print("len", len(filtered_h5_files))
    for h5_file in tqdm(filtered_h5_files):
        #file_path = os.path.join(base_path, h5_file)
        downscale(h5_file, output_base_path, base_path, factor=2)


def main():
    
    # Example usage
    # base_path = "/home/freckmann15/data/mitochondria/cooper/test_mitos"
    # base_path = "/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi"
    base_path = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held_bu"
    output_base_path = "/mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/wichmann/extracted/endbulb_of_held_s2"

    process_h5_files(base_path, output_base_path)


if __name__ == "__main__":
    main()