import argparse
import mrcfile
import imodmodel as imod
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
from synaptic_reconstruction.imod.export import get_label_names


def _write_h5(path, key, image):
    with h5py.File(path, "a") as f:
        f.create_dataset(key, data=image, dtype=image.dtype)


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


def get_filename_without_extension(file_path):
    # Extract the base name (filename with extension)
    base_name = os.path.basename(file_path)
    # Split the base name into name and extension, and return the name
    file_name = os.path.splitext(base_name)[0]
    return file_name


def get_segmentation(imod_path, mrc_path, object_id=None, output_path=None, require_object=True):
    cmd = "imodmop"
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


def extract_common_substring(filepath):
    # Get the filename from the full filepath (in case a path is provided)
    filename = os.path.basename(filepath)
    
    # Split the filename at "mtk" and take the part before it
    common_substring = filename.split('_mtk')[0]
    
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


def main(visualize=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/home/freckmann15/data/mitochondria/cooper/new_mitos", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e",  type=str, default="/home/freckmann15/data/mitochondria/cooper/exported_mitos", help="Path to the root data directory")
    #parser.add_argument("--save_dir", type=str, default="", help="Path to save the data to")
    args = parser.parse_args()

    #base_path = "/home/freckmann15/data/mitochondria/fidi_orig/20240722_WT"
    mod_paths = sorted(glob(os.path.join(args.base_path, "**", "*.mod"), recursive=True))#, reverse=True)
    mrc_paths = sorted(glob(os.path.join(args.base_path, "**", "*.rec"), recursive=True))#, reverse=True)
    # use this for 06
    mod_paths = sorted(glob(os.path.join(args.base_path, "*.mod")), reverse=True)
    mrc_paths = sorted(glob(os.path.join(args.base_path, "*.mrc")), reverse=True)
    for mod_path, mrc_path in zip(mod_paths, mrc_paths):
        scale_down = False
        if ".rec" in mrc_path:
            common_substring = extract_common_substring(mod_path)
            if common_substring not in mrc_path:
                print("\nlooking for correct rec file", common_substring, "\n")
                mrc_path = next((path for path in mrc_paths if common_substring in os.path.basename(path)), None)
                print(mrc_path)
        elif ".mrc" in mrc_path:
            scale_down = False
        label_names = get_label_names(mod_path)
        print(label_names)

        mito_keys = [key for key, value in label_names.items() if "mito" in value.lower()]
        if not mito_keys:
            continue
        if visualize:
            print(f"Visualizing \n{mod_path} and \n{mrc_path}")
        raw = mrcfile.open(mrc_path)
        shape = raw.data.shape

        # mean = np.mean(raw.data)
        # print("Min and Max values: ", raw.data.min(), raw.data.max())
        # print("mean and np.unique mit count", mean ) # np.unique(raw.data, return_counts=True))
        labels = imod.read(mod_path).to_numpy("float")

        if visualize:
            v = napari.Viewer()

        # print("raw image mean", mean)
        # if mean >= -35.0:
        #     v.add_image(raw.data)
        # else:
        #     image = raw.data
        #     print("adding logarithmic image")
        #     v.add_image(np.log1p(image - np.min(image)))
        
        #reconstructed_imod = reconstruct_label_mask(labels, shape)
        #v.add_labels(reconstructed_imod)
        
        # labels = get_segmentation(mod_path, mrc_path, require_object=False, object_id=mito_key)
        # for key in other_keys:
        #     labels = get_segmentation(mod_path, mrc_path, require_object=False, object_id=key)
        #     if labels.sum() == 0:
        #         continue
        #     labels = np.flip(labels, axis=1)
        #     v.add_labels(labels, name="mv_" + str(key))
        export_file_name = get_filename_without_extension(mod_path)
        bbox = None
        for key in mito_keys:
            labels_raw = get_segmentation(mod_path, mrc_path, require_object=False, object_id=key)
            if labels_raw.sum() == 0:
                continue
            if scale_down:
                labels = labels_raw[::scale_down, ::scale_down, ::scale_down]
            else:
                labels = labels_raw
            labels = np.flip(labels, axis=1)
            # cutout ROI
            bbox = _get_bounding_box(labels)
            # bbox = _get_bounding_box(labels)
            min_z, max_z, min_y, max_y, min_x, max_x = bbox
            labels = labels[min_z:max_z, min_y:max_y, min_x:max_x]
            labels = close_mask(labels, structuring_element_shape=(10, 1, 1))
            #print("label sum", labels.sum(), "labelsshape", labels.shape)
            if visualize:
                v.add_labels(labels, name="mito_" + str(key))
            else:
                _write_h5(os.path.join(args.export_path, export_file_name + ".h5"), "labels/mitochondria", labels)
        if scale_down:
            raw_data = raw.data[::scale_down, ::scale_down, ::scale_down]
        else:
            raw_data = raw.data
        min_z, max_z, min_y, max_y, min_x, max_x = bbox
        raw_data = raw_data[min_z:max_z, min_y:max_y, min_x:max_x]
        if visualize:
            v.add_image(raw_data)
            napari.run()
        else:
            _write_h5(os.path.join(args.export_path, export_file_name + ".h5"), "raw", raw_data)


if __name__ == "__main__":
    main()