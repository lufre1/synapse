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
import gc
import h5py
from synapse_net.imod.export import get_label_names


# def _write_h5(path, key, image):
#     if os.path.exists(path):
#         keys = get_all_keys_from_h5(path)
#         if key in keys:
#             print(f"{key} already exists in {path}")
#             return
#     with h5py.File(path, "a") as f:
#         if "label" in key:
#             if "mito" in key:
#                 image = label(image)
#             f.create_dataset(key, data=image, dtype=np.uint8, compression="gzip")
#         else:
#             f.create_dataset(key, data=image, dtype=image.dtype, compression="gzip")
#     print(f"Saved {key} to \n{path}")

def _write_h5(path, key, image, mrc_path=None):
    if os.path.exists(path):
        keys = get_all_keys_from_h5(path)
        if key in keys:
            print(f"{key} already exists in {path}")
            return
    
    with h5py.File(path, "a") as f:
        if "label" in key:
            if "mito" in key:
                image = label(image)
            dataset = f.create_dataset(key, data=image, dtype=np.uint8, compression="gzip")
        else:
            dataset = f.create_dataset(key, data=image, dtype=image.dtype, compression="gzip")
            
        # Add voxel_size metadata if MRC path provided
        if mrc_path and os.path.exists(mrc_path):
            print(f"Attempting to read metadata from: {mrc_path}")
            try:
                voxel_size = mrcfile.open(mrc_path).voxel_size
                voxel_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
                voxel_size_attr = np.array(voxel_size, dtype=voxel_dtype)
                dataset.attrs.create(name='voxel_size', data=voxel_size_attr)
                print("voxel size debug", voxel_size, type(voxel_size))
                # dataset.attrs.create(name='voxel_size', data=voxel_size, shape=voxel_size.shape)
            except Exception as e:
                print(f"Error reading MRC metadata: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"Saved {key} to \n{path}")


def _write_raw_chunked(h5_path, raw_mrc, non_zero_slices, voxel_size=None, chunk_size=16):
    """Write raw MRC memmap to HDF5 in chunks, keeping only non-zero z-slices.

    Reads at most `chunk_size` slices at a time so the full volume is never
    materialized in RAM.  Automatically downcasts to uint8 when the sampled
    data range fits within [0, 255].
    """
    if os.path.exists(h5_path) and "raw" in get_all_keys_from_h5(h5_path):
        print(f"raw already exists in {h5_path}")
        return

    slice_indices = np.where(non_zero_slices)[0]
    n_slices = len(slice_indices)
    if n_slices == 0:
        return

    # Sample ~20 slices to report the value range; only convert float data
    # that genuinely lives in [0, 255] to uint8.  Integer dtypes (including
    # int8 / MRC mode-0) are always preserved as-is to avoid wrap-around.
    step = max(1, n_slices // 20)
    sample_idx = slice_indices[::step]
    global_min = min(float(raw_mrc[i].min()) for i in sample_idx)
    global_max = max(float(raw_mrc[i].max()) for i in sample_idx)

    if raw_mrc.dtype.kind == 'f' and global_min >= 0 and global_max <= 255:
        out_dtype = np.uint8
        print(f"  raw float range [{global_min:.2f}, {global_max:.2f}] fits uint8, converting.")
    else:
        out_dtype = raw_mrc.dtype
        print(f"  raw dtype {out_dtype}, range [{global_min:.2f}, {global_max:.2f}], keeping original dtype.")

    shape = (n_slices, raw_mrc.shape[1], raw_mrc.shape[2])
    with h5py.File(h5_path, "a") as f:
        ds = f.create_dataset(
            "raw", shape=shape, dtype=out_dtype,
            compression="gzip",
            chunks=(min(chunk_size, n_slices), shape[1], shape[2]),
        )
        for start in range(0, n_slices, chunk_size):
            end = min(start + chunk_size, n_slices)
            chunk = raw_mrc[slice_indices[start:end]].astype(out_dtype)
            ds[start:end] = chunk
            del chunk

        if voxel_size is not None:
            try:
                voxel_dtype = np.dtype([('x', np.float32), ('y', np.float32), ('z', np.float32)])
                ds.attrs.create('voxel_size', np.array(voxel_size, dtype=voxel_dtype))
                print(f"  voxel_size: {voxel_size}")
            except Exception as e:
                print(f"  Warning: could not write voxel_size: {e}")

    print(f"Saved raw to\n{h5_path} (dtype={np.dtype(out_dtype).name}, shape={shape})")


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


def _mesh_mod(mod_path, passes=20):
    """Return path to a meshed copy of mod_path (imodmesh -s fills z-gaps).

    passes: imodmesh -P value; must exceed the widest z-gap in the model.
    Falls back to the original path if imodmesh is unavailable or fails.
    """
    imodmesh = shutil.which("imodmesh")
    if imodmesh is None:
        print("  [warn] imodmesh not found, skipping meshing step.")
        return mod_path
    fd, meshed_path = tempfile.mkstemp(suffix=".mod")
    os.close(fd)
    shutil.copy2(mod_path, meshed_path)
    result = run(
        ["imodmesh", "-s", "-P", str(passes), meshed_path],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  [warn] imodmesh failed:\n{result.stderr.decode()}")
        os.remove(meshed_path)
        return mod_path
    return meshed_path


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
        elif "mito" in value.lower() and "cristae" not in value.lower() and "inner" not in value.lower():
            true_labels[key] = "labels/mitochondria"
        elif "cm" in value.lower() or "cristae" in value.lower() or "inner" in value.lower():
            true_labels[key] = "labels/cristae"
        elif "endbulb" in value.lower():
            true_labels[key] = "labels/endbulb"
        else:
            true_labels[key] = f"labels/{value}"
    return true_labels


def fill_z_gaps(mask):
    """Fill empty z-slices using the nearest non-empty slices on each side."""
    filled = mask.copy()
    for z in range(mask.shape[0]):
        if np.any(mask[z]):
            continue
        before = next((z2 for z2 in range(z - 1, -1, -1) if np.any(mask[z2])), None)
        after = next((z2 for z2 in range(z + 1, mask.shape[0]) if np.any(mask[z2])), None)
        if before is not None and after is not None:
            filled[z] = mask[before] | mask[after]
        elif before is not None:
            filled[z] = mask[before]
        elif after is not None:
            filled[z] = mask[after]
    return filled


def crop_data(raw, labels_dict):
    """
    Crop the raw data and all label datasets in labels_dict to a subset containing only slices with label data.

    Parameters
    ----------
    raw : numpy.ndarray
        The raw data to crop.
    labels_dict : dict
        A dictionary with label names as keys and the corresponding label datasets as values.

    Returns
    -------
    raw_cropped : numpy.ndarray
        The cropped raw data.
    cropped_labels_dict : dict
        A dictionary with the same keys as labels_dict, but with the cropped label datasets as values.

    """
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
    parser.add_argument("--force_overwrite", "-f", default=False, action='store_true', help="If to over-write already present segmentation results.")
    parser.add_argument("--include_az", default=False, action='store_true', help="Include active zone (labels/az) in exported files.")
    # parser.add_argument("--save_dir", type=str, default="", help="Path to save the data to")
    args = parser.parse_args()
    print(args.base_path)
    visualize = args.visualize
    print_labels = args.print_labels

    mod_paths = sorted(glob(os.path.join(args.base_path, "**", "*.mod"), recursive=True))
    mrc_paths = sorted(glob(os.path.join(args.base_path, "**", "*.mrc"), recursive=True))
    rec_paths = sorted(glob(os.path.join(args.base_path, "**", "*.rec"), recursive=True))
    mrc_paths.extend(rec_paths)
    # use this for 06
    # mod_paths = sorted(glob(os.path.join(args.base_path, "*.mod")), reverse=True)
    # mrc_paths = sorted(glob(os.path.join(args.base_path, "*.mrc")), reverse=True)
    count = 0
    for mod_path, mrc_path in tqdm(zip(mod_paths, mrc_paths)):
        # count += 1
        # if count <= 1:
        #     continue
        # if "37371_O5_66K_TS_SP_34-01_rec_2Kb1dawbp_cropF" not in mod_path:
        #     continue
        if print_labels:
            print("\n\nmod path", mod_path)
            print(get_label_names(mod_path))
            print("\nmrc path", mrc_path, "voxel  _size", mrcfile.open(mrc_path).voxel_size)
            continue
        # if mod_path == "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/mitos_and_cristae/Otof-WT_P21/WT22_eb2_AZ1_10K_model2.mod":
        #     continue
        scale_down = False
        export_file_name, rel_path = get_filename_and_inter_dirs(mod_path, args.base_path)
        create_directories_if_not_exists(args.export_path, rel_path)
        export_file_path = os.path.join(args.export_path, rel_path, export_file_name + ".h5")
        if os.path.exists(export_file_path) and not args.force_overwrite:
            print("File already exists:", export_file_path)
            continue
        mrc_basename = os.path.splitext(os.path.basename(mrc_path))[0]
        mod_basename = os.path.splitext(os.path.basename(mod_path))[0]
        if mrc_basename != mod_basename:
            mrc_path = find_matching_rec_file(mod_path, mrc_paths)
        print("\nmrc path", mrc_path, "\nmod path", mod_path, "\n")

        label_names = get_true_labels(get_label_names(mod_path))

        label_dict = {k: v for k, v in label_names.items() if args.include_az or v != "labels/az"}

        if not label_dict:
            print("\nNo mito labels found in", mod_path)
            continue
        if visualize:
            print(f"Visualizing \n{mod_path} and \n{mrc_path}")
        if mrc_path is None:
            print("Could not find a mrc or rec file for", mod_path)
            continue
        meshed_mod_path = _mesh_mod(mod_path)
        try:
            labels = {}
            for key, val in label_dict.items():
                labels[key] = np.flip(get_segmentation(meshed_mod_path, mrc_path, require_object=False, object_id=key), axis=1)
        finally:
            if meshed_mod_path != mod_path:
                os.remove(meshed_mod_path)

        print("\nexporting to", export_file_path)

        true_labels = {}
        for key, val in label_dict.items():
            if val not in true_labels.keys():
                true_labels[val] = labels[key]
            else:
                true_labels[val] = true_labels[val] + labels[key]
        del labels

        processed = {}
        for k, v in true_labels.items():
            v_bool = v.astype(bool)
            if "cristae" not in k:
                v_bool = close_mask(fill_z_gaps(v_bool), structuring_element_shape=(3, 1, 1), iterations=10)
            processed[k] = v_bool
        true_labels = processed
        del processed

        # Compute non-zero z-slices from labels without loading raw into RAM
        combined_mask = np.zeros(next(iter(true_labels.values())).shape, dtype=bool)
        for v in true_labels.values():
            combined_mask |= v
        non_zero_slices = np.any(combined_mask, axis=(1, 2))
        del combined_mask

        if visualize:
            with mrcfile.open(mrc_path, mode='r', permissive=True) as mrc:
                raw_crop = mrc.data[non_zero_slices].copy()
            v = napari.Viewer()
            v.add_image(raw_crop)
            for k, val in true_labels.items():
                v.add_labels(val[non_zero_slices], name=k)
            napari.run()
        else:
            # Write raw in chunks from memmap, never loading the full volume
            with mrcfile.open(mrc_path, mode='r', permissive=True) as mrc:
                _write_raw_chunked(export_file_path, mrc.data, non_zero_slices, voxel_size=mrc.voxel_size)
            gc.collect()

            # Write labels one at a time so only one is in RAM at a time
            for key in list(true_labels.keys()):
                _write_h5(export_file_path, key, true_labels.pop(key)[non_zero_slices])


if __name__ == "__main__":
    main()
