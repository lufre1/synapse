import h5py
import os
from glob import glob
import numpy as np
from tqdm import tqdm
from skimage.transform import rescale, resize
import argparse
from synapse.util import get_data_metadata
from elf.io import open_file
import tifffile
import synapse.util as util
import synapse.io.util as io
import synapse.h5_util as h5_util
from skimage.morphology import remove_small_holes
from skimage.measure import label


def remove_small_blobs_3d_instances(seg, min_voxels, connectivity=1):
    """
    Remove small connected components in an instance segmentation.

    Components with voxel count < min_voxels are removed (set to 0).

    Parameters
    ----------
    seg : np.ndarray (3D, int)
        Instance-labeled segmentation (0 = background).
    min_voxels : int
        Minimum number of voxels to keep.
    connectivity : int
        1 = 6, 2 = 18, 3 = 26 connectivity.

    Returns
    -------
    np.ndarray
        Cleaned instance segmentation.
    """
    import numpy as np
    from skimage.measure import label, regionprops

    out = seg.copy()

    # Process each label independently
    for obj_id in np.unique(seg):
        if obj_id == 0:
            continue

        mask = seg == obj_id
        cc = label(mask, connectivity=connectivity)

        for prop in regionprops(cc):
            if prop.area < min_voxels:
                out[(cc == prop.label)] = 0

    return out


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, required=True, help="Path to the root data directory")
    parser.add_argument("--second_base_path", "-b2",  type=str, required=True, help="Path to the root data directory")
    parser.add_argument("--export_path", "-e", type=str, default=None, help="Path to the export directory")
    parser.add_argument("--scale_factor", "-s", type=int, default=1, help="Scale factor for the image")
    parser.add_argument("--import_file_extension", "-ife", type=str, default=".h5", help="File extension to read data")
    parser.add_argument("--second_import_file_extension", "-ife2", type=str, default=".h5", help="File extension to read data")
    parser.add_argument("--voxel_size", "-vs", type=float, nargs=3, default=None, help="Voxel size tuple: z, y, x")
    parser.add_argument("--dataset_name", "-dn", type=str, default="labels/mitochondria", help="ds name to add")
    parser.add_argument("--preprocess", "-pre", action="store_true", help="Whether to preprocess the data")
    args = parser.parse_args()
    scale = args.scale_factor
    ife = args.import_file_extension
    ife2 = args.second_import_file_extension

    dataset_name = args.dataset_name
    voxel_size = args.voxel_size
    if voxel_size is not None:
        voxel_size = voxel_size if isinstance(voxel_size, np.ndarray) else np.array(voxel_size, dtype=np.float32)
    # voxel_size = [0.025, 0.005, 0.005]  # [8.694*2, 8.694*2, 8.694*2])

    paths = io.load_file_paths(args.base_path, ext=ife)
    paths_2 = io.load_file_paths(args.second_base_path, ext=ife2)

    for path in tqdm(paths, total=len(paths)):
        path2 = util.find_label_file(path, paths_2)
        if path2 is None:
            print("Could not find label file for", path)
            continue
        export_file_name, rel_path = get_filename_and_inter_dirs(path, args.base_path)
        export_file_name = export_file_name.replace("mitotomo-net32-lr1e-4-bs8-ps32x256x256-s4_sd4_bt015_with_pred_ts_z32_y256_x256_halo_z8_y64_x64_", "").replace(
            "_s2_refined", ""
        )
        if voxel_size is None:
            voxel_size = h5_util.read_voxel_size(
                path,
                default=None  # h5_util.read_voxel_size(path2),
            )
        raw_shape = util.read_data(path, scale=scale)["raw"].shape
        if ife2 == ".h5":
            tmp = util.read_data(path2, scale=scale)
            tmp = tmp.pop(dataset_name, None)
        else:
            tmp = tifffile.imread(path2)
        # preprocess
        if args.preprocess:
            tmp = remove_small_holes(tmp, area_threshold=200)
            tmp = label(tmp)
            tmp = remove_small_blobs_3d_instances(tmp, min_voxels=100)
        data2 = {}
        if not np.array_equal(tmp.shape, raw_shape):
            data2[dataset_name] = resize(tmp, raw_shape, preserve_range=True, order=0, anti_aliasing=False).astype(np.uint8)
        else:
            data2[dataset_name] = tmp.astype(np.uint8)
        with open_file(path, "a") as f:
            ds = f.require_dataset(
                dataset_name,
                data=data2[dataset_name],
                compression="gzip",
                dtype=data2[dataset_name].dtype,
                shape=data2[dataset_name].shape,
            )
            ds[:] = data2[dataset_name]
            print("mitochondria added to file", path)


if __name__ == "__main__":
    main()
