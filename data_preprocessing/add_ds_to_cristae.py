import h5py
import os
from glob import glob
import numpy as np
from tqdm import tqdm
from skimage.transform import rescale, resize
import argparse
# from synapse.util import get_data_metadata
from elf.io import open_file
import elf.parallel as parallel
import tifffile
import synapse.util as util
import synapse.io.util as io
import synapse.h5_util as h5_util
from skimage.morphology import remove_small_holes
from skimage.measure import label
from typing import Tuple
import time
from skimage.morphology import ball, binary_erosion, remove_small_objects 
from scipy.ndimage import distance_transform_edt


def grow_labels_to_mask(seed_labels, target_mask):
    """Expand seed instance labels to cover target_mask without merging labels."""
    # distance_transform_edt returns indices of nearest zero; use it on background of seeds
    bg = seed_labels == 0
    _, inds = distance_transform_edt(bg, return_indices=True)
    nearest = seed_labels[tuple(inds)]  # nearest seed label for every voxel

    out = np.zeros_like(seed_labels, dtype=seed_labels.dtype)
    out[target_mask] = nearest[target_mask]  # only fill where original foreground is True
    return out


def remove_disconnected_islands_per_id(seg, connectivity=1, min_island_voxels=0, verbose=False):
    """
    For each nonzero instance id, keep only its largest connected component.
    Optionally also remove kept components smaller than min_island_voxels.

    Parameters
    ----------
    seg : np.ndarray (int)
        Instance segmentation (0=background).
    connectivity : int
        1 -> 6-neighborhood in 3D (recommended).
    min_island_voxels : int
        If >0, instances whose largest component is smaller than this are removed entirely.
    """
    out = seg.copy()
    ids = np.unique(out)
    ids = ids[ids != 0]

    for obj_id in ids:
        m = (out == obj_id)
        cc = label(m, connectivity=connectivity)
        if cc.max() <= 1:
            continue

        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        keep_cc = sizes.argmax()
        keep_size = sizes[keep_cc]

        # remove all components except the largest
        out[(cc != 0) & (cc != keep_cc)] = 0

        # optionally drop tiny instances entirely
        if min_island_voxels > 0 and keep_size < min_island_voxels:
            out[out == obj_id] = 0

        if verbose:
            removed = int(sizes.sum() - keep_size)
            if removed > 0:
                print(f"ID {obj_id}: removed {removed} island voxels (kept {keep_size})")

    return out


def apply_size_filter(
    segmentation: np.ndarray,
    min_size: int,
    verbose: bool = False,
    block_shape: Tuple[int, int, int] = (128, 256, 256),
) -> np.ndarray:
    """Apply size filter to the segmentation to remove small objects.

    Args:
        segmentation: The segmentation.
        min_size: The minimal object size in pixels.
        verbose: Whether to print runtimes.
        block_shape: Block shape for parallelizing the operations.

    Returns:
        The size filtered segmentation.
    """
    if min_size == 0:
        return segmentation
    t0 = time.time()
    if segmentation.ndim == 2 and len(block_shape) == 3:
        block_shape_ = block_shape[1:]
    else:
        block_shape_ = block_shape
    ids, sizes = parallel.unique(segmentation, return_counts=True, block_shape=block_shape_, verbose=verbose)
    filter_ids = ids[sizes < min_size]
    segmentation[np.isin(segmentation, filter_ids)] = 0
    if verbose:
        print("Size filter in", time.time() - t0, "s")
    return segmentation

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
    parser.add_argument("--dataset_name", "-dn", type=str, default="raw_mitos_combined", help="ds name to add")
    parser.add_argument("--preprocess", "-pre", action="store_true", help="Whether to preprocess the data")
    parser.add_argument("--with_channel", "-wc", type=int, default=None, help="To whcihc channel to add the ds")
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
        if len(paths) == len(paths_2) == 1:
            path2 = paths_2[0]
        else:
            path2 = util.find_label_file(path, paths_2)

        if path2 is None:
            print("Could not find label file for", path)
            continue

        if voxel_size is None:
            voxel_size = h5_util.read_voxel_size(
                path,
                default=None,  # h5_util.read_voxel_size(path2),
                h5_key="raw_mitos_combined"
            )
        if args.with_channel is not None:
            raw_shape = util.read_data(path, scale=scale)[dataset_name][args.with_channel].shape
        else:
            raw_shape = util.read_data(path, scale=scale)[dataset_name].shape
        if ife2 == ".h5":
            tmp = util.read_data(path2, scale=scale)[dataset_name]
        else:
            tmp = tifffile.imread(path2)
        # preprocess
        if args.preprocess:
            print("Preprocessing Files:\n", path, "\n", path2)
            min_size = 250

            mask = tmp.astype(bool)
            for z in tqdm(range(mask.shape[0]), desc="Removing holes in 2D"):
                mask[z] = remove_small_holes(mask[z], area_threshold=min_size, connectivity=1)
            # mask = remove_small_holes(mask, max_size=200)
            mask = remove_small_objects(mask, min_size=min_size)
            print("Finished removing holes")

            # Erode to break thin connections between large axon segments
            erode_radius = 3  # increase to 2 if connections are thicker (more risk of splitting)
            eroded = binary_erosion(mask, footprint=ball(erode_radius))
            print("Finished erosion")

            # tmp = label(eroded, connectivity=1)  # 6-connectivity to reduce merges
            tmp = parallel.label(
                data=eroded,
                block_shape=(128, 256, 256),
                verbose=True
            )
            print("Finished labeling (on eroded mask)")
            
            tmp = remove_disconnected_islands_per_id(tmp, connectivity=1, min_island_voxels=min_size, verbose=True)
            print("Finished removing disconnected islands with min_size of", min_size)

            tmp = apply_size_filter(tmp, min_size=min_size, verbose=True)
            print("Finished removing small instances with min_size of", min_size)

            # Grow labels back to the original (hole-filled) mask without merging
            tmp = grow_labels_to_mask(tmp, mask)
            print("Finished growing labels back")

        data2 = {}
        if not np.array_equal(tmp.shape, raw_shape):
            print("resiszing", tmp.shape, "to", raw_shape)
            data2[dataset_name] = resize(tmp, raw_shape, preserve_range=True, order=0, anti_aliasing=False).astype(np.uint8)
        else:
            data2[dataset_name] = tmp.astype(np.uint8)
        with open_file(path, "a") as f:
            if dataset_name in f:
                ds = f[dataset_name]
                if args.with_channel is not None:
                    ds[args.with_channel] = data2[dataset_name]
                else:
                    ds[:] = data2[dataset_name]
            else:
                raise ValueError(f"Dataset {dataset_name} not found in file {path}")
            
                ds = f.require_dataset(
                    dataset_name,
                    data=data2[dataset_name] if args.with_channel is None else None,
                    compression="gzip",
                    dtype=data2[dataset_name].dtype,
                    shape=raw_shape,
                )
                if args.with_channel is not None:
                    ds[args.with_channel] = data2[dataset_name]
                else:
                    ds[:] = data2[dataset_name]
            print(f"{args.dataset_name} added to file", path)


if __name__ == "__main__":
    main()
