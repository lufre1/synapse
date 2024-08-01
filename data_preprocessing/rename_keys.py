import h5py
import os
from glob import glob
from tqdm import tqdm
from skimage import measure
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
        # if np.any(cristae != 0):
        #     h5f.create_dataset('labels/cristae', data=cristae)
        # if cristae is not None:
        #     vis_data = {
        #         "raw": raw,
        #         "label": mitochondria,
        #         "pred1": cristae
        #     }
        # util.visualize_data_napari(vis_data)
        # h5f.create_dataset('labels/cristae', data=cristae)


def main():
    
    # Example usage
    # base_path = "/home/freckmann15/data/mitochondria/fidi_orig/"
    label_base_path = "/home/freckmann15/data/mitochondria/corrected_mito_h5_label_split/"
    # bu_path = "/home/freckmann15/data/mitochondria/fidi_h5/fidi"
    # old_key = "labels/mitchondria"
    # new_key = "labels/mitochondria"  
    # raw_file_paths = sorted(glob(os.path.join(base_path, "**/*raw.mrc")),key=lambda x: os.path.basename(x))
    label_h5_file_paths = sorted(glob(os.path.join(label_base_path, "*.h5")), key=lambda x: os.path.basename(x))
    cristaefolder = "/home/freckmann15/data/mitochondria/corrected_mito_h5_label_split/with_cistae"

    for path in label_h5_file_paths:
        print(path)
        with h5py.File(path, 'r') as hdf5_file:
            # check if cristae key exists
            if 'labels/cristae' in hdf5_file:
                print("Cristae exists in the file.")
                print("Moving file to cristae folder")
                os.rename(path, os.path.join(cristaefolder, os.path.basename(path)))
                print("check if new file exists")
                if os.path.exists(os.path.join(cristaefolder, os.path.basename(path))):
                    print("File moved successfully.")
                else:
                    print("File not moved.")


if __name__ == "__main__":
    main()