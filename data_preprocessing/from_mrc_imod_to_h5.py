import argparse
import mrcfile
#import imodmodel as imod
from tqdm import tqdm
import napari
import os
from glob import glob
import numpy as np
from skimage.measure import regionprops, label
from elf.io import open_file
import shutil
import tempfile
from subprocess import run
from scipy.ndimage import binary_closing
import h5py
from synapse_net.imod.export import get_label_names


def _write_h5(path, key, image):
    if os.path.exists(path):
        keys = get_all_keys_from_h5(path)
        if key in keys:
            print(f"{key} already exists in {path}")
            return
    with h5py.File(path, "a") as f:
        if "label" in key:
            f.create_dataset(key, data=image, dtype=np.uint8, compression="gzip")
        else:
            f.create_dataset(key, data=image, dtype=image.dtype)
    print(f"Saved {key} to \n{path}")


def reconstruct_label_mask(imod_data, shape):
    """
    Reconstructs a label mask from IMOD data containing point coordinates.

    Args:
        imod_data (list): A list of lists, where each inner list represents a point
                          with five values (potentially label ID, X, Y, Z, and an unused value).

    Returns:
        numpy.ndarray: A numpy array representing the 3D label mask.

    This function takes IMOD data as input, which is a list of lists where each inner list
    represents a point with five values (label ID, X, Y, Z, and an unused value). It extracts
    the relevant data, finds the unique labels and their corresponding coordinates. It then
    initializes a label mask with zeros and fills it based on the label IDs and coordinates.
    The label mask is returned as a numpy array.

    Note:
        The first column of the imod_data is ignored and the fifth value is assumed to be unused.
        The coordinates are assumed to be integers.
    """

    modified_data = []
    #print(imod_data.shape)
    # print("unique values of the second column", np.unique(imod_data[:, 0]))
    # Extract relevant data (assuming first column is ignored and fifth is unused)
    for row in imod_data:
        #print("row shape and row", row.shape, row)
        label, x, y, z = row[1:]
        label = int(label)
        x = int(x)
        y = int(y)
        z = int(z)
        modified_data.append([label, x, y, z])

    labels, x, y, z = zip(*modified_data)
    # unique_labels = np.unique(labels)
    # min_x = min(x)
    # max_x = max(x)
    # min_y = min(y)
    # max_y = max(y)
    # min_z = min(z)
    # max_z = max(z)

    # Initialize label mask with zeros
    # mask_shape = (np.ceil(max_z - min_z + 1).astype(int), np.ceil(max_y - min_y + 1).astype(int), np.ceil(max_x - min_x + 1).astype(int))
    label_mask = np.zeros(shape, dtype=int)

    # Fill mask based on label IDs and coordinates
    for label, x_val, y_val, z_val in zip(labels, x, y, z):
        # Adjust coordinates to account for zero-based indexing
        adjusted_x = x_val if x_val < shape[2] else shape[2] - 1  #- min_x
        adjusted_y = y_val if y_val < shape[1] else shape[1] - 1  #- min_y
        adjusted_z = z_val if z_val < shape[0] else shape[0] - 1  #- min_z
        label_mask[adjusted_z, adjusted_y-1, adjusted_x-1] = label

    return label_mask


def get_filename_and_inter_dirs(file_path, base_path):
    # Extract the base name (filename with extension)
    base_name = os.path.basename(file_path)
    # Split the base name into name and extension to get the filename
    file_name = os.path.splitext(base_name)[0]
    # Get the relative path of file_path from base_path
    relative_path = os.path.relpath(file_path, base_path)
    # Get the intermediate directories by removing the filename from the relative path
    inter_dirs = os.path.dirname(relative_path)
    return file_name, inter_dirs


def get_segmentation(imod_path, mrc_path, object_id=None, output_path=None, require_object=True):
    cmd = "imodmop"  # "/home/freckmann15/u12103/imod/IMOD/bin/imodmop"#"imodmop"
    cmd_path = shutil.which(cmd)
    assert cmd_path is not None, f"Could not find the {cmd} imod command."

    with tempfile.NamedTemporaryFile() as f:
        tmp_path = f.name

        if object_id is None:
            cmd = [cmd, "-ma", "1", imod_path, mrc_path, tmp_path]
        else:
            cmd = [cmd, "-ma", "1", "-o", str(object_id), imod_path, mrc_path, tmp_path]

        run(cmd)
        with open_file(tmp_path, ext=".mrc", mode="r") as f:
            data = f["data"][:]

    segmentation = data == 1
    if require_object and segmentation.sum() == 0:
        id_str = "" if object_id is None else f"for object {object_id}"
        raise RuntimeError(f"Segmentation extracted from {imod_path} {id_str} is empty.")

    if output_path is None:
        return segmentation


def close_mask(labels, structuring_element_shape=(3, 1, 1), iterations=None):
    structuring_element = np.zeros(structuring_element_shape)
    structuring_element[:, 0, 0] = 1
    #print("labels shape and structuring element shape", labels.shape, structuring_element.shape)
    if iterations is not None:
        return binary_closing(labels, structuring_element, iterations=iterations)
    else:
        return binary_closing(labels, structuring_element)


def get_all_keys_from_h5(file_path):
    keys = []
    with h5py.File(file_path, 'r') as h5file:
        def collect_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.append(name)  # Add each key (path) to the list
        h5file.visititems(collect_keys)  # Visit all groups and datasets
    return keys


def find_matching_rec_file(mod_file, rec_files):
    # Extract base name of the mod file (without extension)
    mod_name = os.path.basename(mod_file)

    # Remove 'mtk_' and '.mod' parts from the mod file name to get the identifier
    mod_base = mod_name.replace('_model2', '').replace("_model", "").replace('.mod', '').replace("model", "")

    # Now loop over rec files to find a match
    for rec_file in rec_files:
        rec_name = os.path.basename(rec_file)

        # Normalize rec file name: remove '_SP.rec' or '.rec'
        rec_base = rec_name.replace('_SP.rec', '').replace('_rec', '').replace('.rec', '').replace(".mrc", "")

        # Check if the relevant part of mod file name is in the rec file name
        if mod_base in rec_base:
            return rec_file

    # If no exact match is found, return None (or raise an error)
    return None


def extract_common_substring(filepath, split='_mtk'):
    # Get the filename from the full filepath (in case a path is provided)
    filename = os.path.basename(filepath)
    
    # Split the filename at "mtk" and take the part before it
    common_substring = filename.split(split)[0]
    
    return common_substring


def _get_bounding_box(label_volume):
    labels = label(label_volume)
    regions = regionprops(labels)
    min_z, min_y, min_x = float('inf'), float('inf'), float('inf')
    max_z, max_y, max_x = float('-inf'), float('-inf'), float('-inf')
    for region in regions:
        box = region.bbox
        min_z_temp, min_y_temp, min_x_temp, max_z_temp, max_y_temp, max_x_temp = box
        
        # Update the overall min and max values
        min_z = min(min_z, min_z_temp)
        min_y = min(min_y, min_y_temp)
        min_x = min(min_x, min_x_temp)
        max_z = max(max_z, max_z_temp)
        max_y = max(max_y, max_y_temp)
        max_x = max(max_x, max_x_temp)

    bbox = (min_z, max_z, min_y, max_y, min_x, max_x)
    return bbox


def create_directories_if_not_exists(base_path, inter_dirs):
    # Construct the full path from base_path and inter_dirs
    full_path = os.path.join(base_path, inter_dirs)
    
    # Check if the path exists
    if not os.path.exists(full_path):
        # If it doesn't exist, create the directories
        os.makedirs(full_path)
        print(f"\nCreated directories: {full_path}")
    else:
        print(f"\nDirectories already exist: {full_path}")


def get_true_labels(label_dict):
    true_labels = {}
    for key, value in label_dict.items():
        if "az" in value.lower():
            true_labels[key] = "labels/az"
        elif "mito" in value.lower() and "cristae" not in value.lower():
            true_labels[key] = "labels/mitochondria"
        elif "cm" in value.lower() or "cristae" in value.lower():
            true_labels[key] = "labels/cristae"
        elif "endbulb" in value.lower():
            true_labels[key] = "labels/endbulb"
        else:
            true_labels[key] = f"labels/{value}"
    return true_labels


def crop_data(raw, labels_dict):
    combined_labels = np.zeros_like(next(iter(labels_dict.values())))
    for labels in labels_dict.values():
        combined_labels |= labels

    # Identify slices along the z-dimension where there is label data
    non_zero_slices = np.any(combined_labels, axis=(1, 2))

    # Crop raw data
    raw_cropped = raw[non_zero_slices, :, :]

    # Crop each label dataset in labels_dict
    cropped_labels_dict = {
        label_name: labels[non_zero_slices, :, :]
        for label_name, labels in labels_dict.items()
    }

    return raw_cropped, cropped_labels_dict


def main():
    parser = argparse.ArgumentParser()
    # /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer
    parser.add_argument("--base_path", "-b",  type=str, default="/home/freckmann15/data/mitochondria/cooper/new_mitos", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e",  type=str, default="/home/freckmann15/data/mitochondria/wichmann/output", help="Path to the root data directory")
    parser.add_argument("--visualize", "-v", default=False, action='store_true', help="If to visualize or not")
    parser.add_argument("--print_labels", "-pl", default=False, action='store_true', help="If to print labels from mod file or not")
    #parser.add_argument("--save_dir", type=str, default="", help="Path to save the data to")
    args = parser.parse_args()
    print(args.base_path)
    visualize = args.visualize
    print_labels = args.print_labels

    mod_paths = sorted(glob(os.path.join(args.base_path, "**", "*.mod"), recursive=True))#, reverse=True)
    mrc_paths = sorted(glob(os.path.join(args.base_path, "**", "*.mrc"), recursive=True))#, reverse=True)
    rec_paths = sorted(glob(os.path.join(args.base_path, "**", "*.rec"), recursive=True))
    mrc_paths.extend(rec_paths)
    # use this for 06
    # mod_paths = sorted(glob(os.path.join(args.base_path, "*.mod")), reverse=True)
    # mrc_paths = sorted(glob(os.path.join(args.base_path, "*.mrc")), reverse=True)
    for mod_path, mrc_path in tqdm(zip(mod_paths, mrc_paths)):
        if print_labels:
            print("\n\n", mod_path)
            print(get_label_names(mod_path))
            print("\n", mrc_path, "voxel  _size", mrcfile.open(mrc_path).voxel_size)
            continue
        if mod_path == "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/mitos_and_cristae/Otof-WT_P21/WT22_eb2_AZ1_10K_model2.mod":
            continue
        scale_down = False
        export_file_name, rel_path = get_filename_and_inter_dirs(mod_path, args.base_path)
        create_directories_if_not_exists(args.export_path, rel_path)
        export_file_path = os.path.join(args.export_path, rel_path, export_file_name + ".h5")
        if os.path.exists(export_file_path):
            print("File already exists:", export_file_path)
            continue
        mrc_basename = os.path.basename(mrc_path)
        mod_basename = os.path.basename(mod_path).strip("_model")
        if mrc_basename != mod_basename:
            mrc_path = find_matching_rec_file(mod_path, mrc_paths)
        print("\n", mrc_path, "\n", mod_path, "\n")

        label_names = get_true_labels(get_label_names(mod_path))

        label_dict = label_names  # [key for key, value in label_names.items() if "mito" in value.lower()]

        if not label_dict:
            print("\nNo mito labels found in", mod_path)
            continue
        if visualize:
            print(f"Visualizing \n{mod_path} and \n{mrc_path}")
        if mrc_path is None:
            print("Could not find a mrc or rec file for", mod_path)
            continue
        raw = mrcfile.open(mrc_path)
        raw_data = raw.data
        labels = {}
        for key, val in label_dict.items():
            labels[key] = np.flip(get_segmentation(mod_path, mrc_path, require_object=False, object_id=key), axis=1)

        if visualize:
            v = napari.Viewer()

        print("\nexporting to", export_file_path)

        true_labels = {}
        for key, val in label_dict.items():
            if val not in true_labels.keys():
                true_labels[val] = labels[key]
            else:
                true_labels[val] = true_labels[val] + labels[key]
        raw_data, true_labels = crop_data(raw_data, true_labels)

        if visualize:
            v.add_image(raw_data)
            for k, val in true_labels.items():
                v.add_labels(val, name=k)
            napari.run()
        else:
            _write_h5(export_file_path, "raw", raw_data)
            for key, val in true_labels.items():
                #print("key", key, "image has vals?", np.any(val))
                _write_h5(export_file_path, key, val)


if __name__ == "__main__":
    main()
