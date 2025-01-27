import h5py
import os
from glob import glob
import numpy as np
from tqdm import tqdm
from skimage import measure
import argparse
# import numpy as np
import mrcfile
from synapse.util import get_data_metadata
from synapse.h5_util import read_h5, get_all_keys_from_h5
import napari
from elf.io import open_file
from elf.evaluation.matching import label_overlap, intersection_over_union
from elf.parallel import label as parallel_label
from skimage.segmentation import relabel_sequential
from skimage.morphology import binary_closing, remove_small_objects, label
import tifffile


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


def export_to_h5(data, export_path):
    with h5py.File(export_path, 'x') as h5f:
        for key in data.keys():
            h5f.create_dataset(key, data=data[key], compression="gzip")
    print("exported to", export_path)


def trim_z_dim(h5_file_path, z_dim_trim, export_path):
    dataset_names = get_all_datasets(h5_file_path)
    export_file_name = os.path.basename(h5_file_path)
    data = {}
    with h5py.File(h5_file_path, 'r') as h5f:
        for name in dataset_names:
            print(name)
            # breakpoint()
            image = h5f[name]
            data[name] = image[z_dim_trim[0]:z_dim_trim[1], :, :]

    export_to_h5(data, os.path.join(export_path, export_file_name))


def find_trimmed_and_new_labels_pair(t_path, nl_paths, type):
    t_name = os.path.basename(t_path)
    for nl_path in nl_paths:
        nl_name = os.path.basename(nl_path).replace(".tif", "")
        if nl_name in t_name:
            print(f"Found new {type} file for trimmed file:\n", nl_path)
            return nl_path
    print(f"Could not find new {type} file for trimmed file", t_name)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/home/freckmann15/data/mitochondria/wichmann/trimmed_all", help="Path to the root data directory")
    parser.add_argument("--label_path", "-lp",  type=str, default="/home/freckmann15/data/mitochondria/wichmann/manual_and_microsam_annotations", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e", type=str, default="/home/freckmann15/data/mitochondria/wichmann/test/", help="Path to the export directory")
    parser.add_argument("--scale_factor", "-s", type=int, default=1, help="Scale factor for the image")
    args = parser.parse_args()
    base_path = args.base_path
    label_path = args.label_path
    export_path = args.export_path
    scale_factor = args.scale_factor

    ### cluster
    export_path = "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/more_fully_annotated_mitos"
    label_path = "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/manual_mitochondria_labels"
    base_path = "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/trimmed_all"

    h5_paths = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    h5_label_paths = sorted(glob(os.path.join(label_path, "**", "*.tif"), recursive=True))
    skip = True
    for h5_path in tqdm(h5_paths):
        output_path = os.path.join(export_path, os.path.basename(h5_path))
        if os.path.exists(output_path):
            print("output path already exists:", output_path)
            continue
        if "WT21_syn5_model2" in h5_path:
            skip = False
        if skip:
            continue
        label_path = find_trimmed_and_new_labels_pair(h5_path, h5_label_paths, type="label")
        if label_path is None:
            continue
        keys = get_all_keys_from_h5(h5_path)
        data = {}
        for key in keys:
            if "mitochondria" in key:
                continue
            else:
                data[key] = read_h5(h5_path, key, scale_factor)
        #new_labels = np.array(open_file(label_path, ext=".tif")[:], dtype=np.uint8)  # read_h5(label_path, "labels/mitochondria", scale_factor)
        new_labels = tifffile.imread(label_path)
        # new_labels = binary_closing(new_labels)
        # new_labels = parallel_label(new_labels, block_shape=(16, 512, 512))
        new_labels = label(new_labels)
        new_labels = remove_small_objects(new_labels.astype(np.uint8), min_size=1000)

        data["labels/mitochondria"] = new_labels

        export_to_h5(data, output_path)

        # v = napari.Viewer()
        # v.add_image(data["raw"])
        # # v.add_labels(data["labels/mitochondria"], name="original_labels")
        # v.add_labels(new_labels, name="new_labels")
        # napari.run()


if __name__ == "__main__":
    main()