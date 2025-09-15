import argparse
import os
from glob import glob
import h5py
import zarr
import torch_em
import torch_em.transform
from tqdm import tqdm
from elf.io import open_file
import numpy as np
import synapse.io.util as io
from synapse_net.inference.mitochondria import segment_mitochondria
# from synapse_net.ground_truth.matching import find_additional_objects
from elf.evaluation.matching import label_overlap, intersection_over_union
from skimage.segmentation import relabel_sequential
from skimage.measure import label
from skimage.transform import resize


def find_additional_objects(
    ground_truth: np.ndarray,
    segmentation: np.ndarray,
    matching_threshold: float = 0.5
) -> np.ndarray:
    """
    Identify additional objects in the segmentation that are not sufficiently covered
    by the ground truth based on a matching threshold.

    Args:
        ground_truth (np.ndarray): Ground truth labeled segmentation.
        segmentation (np.ndarray): Predicted labeled segmentation.
        matching_threshold (float): IoU threshold to identify matched objects. 
                                    Objects with IoU > threshold are considered covered.

    Returns:
        np.ndarray: A labeled segmentation containing only the additional objects.
    """

    # Relabel both ground truth and segmentation sequentially for consistent IDs
    ground_truth = relabel_sequential(ground_truth)[0]
    segmentation = relabel_sequential(segmentation)[0]

    # Compute overlap and IoU between segmentation and ground truth
    overlap, _ = label_overlap(segmentation, ground_truth)
    iou = intersection_over_union(overlap)

    # Get all segmentation IDs
    seg_ids = np.unique(segmentation)

    # Identify IDs of segmentation objects that overlap with ground truth objects above the threshold
    matched_ids = set()
    for seg_id in seg_ids:
        if seg_id == 0:  # Skip background
            continue
        max_overlap = iou[seg_id, :].max()
        if max_overlap > matching_threshold:
            matched_ids.add(seg_id)

    # Create a mask for additional objects (segmentation IDs not matched)
    additional_objects = segmentation.copy()
    for matched_id in matched_ids:
        additional_objects[additional_objects == matched_id] = 0

    # Relabel the additional objects to keep them contiguous
    additional_objects = relabel_sequential(additional_objects)[0]

    return additional_objects


def export_to_h5(data, export_path):
    with h5py.File(export_path, 'x') as h5f:
        for key in data.keys():
            h5f.create_dataset(key, data=data[key], compression="gzip")
    print("exported to", export_path)


def _read_h5(path, key, scale_factor, z_offset=None):
    with h5py.File(path, "r") as f:
        try:
            print(f"{key} data shape", f[key].shape)
            if key == "prediction" or "pred" in key:
                image = f[key][:, ::scale_factor, ::scale_factor, ::scale_factor]
                if z_offset:
                    image = image[z_offset[0]:z_offset[1], :, :]
            else:
                image = f[key][::scale_factor, ::scale_factor, ::scale_factor]
                if z_offset:
                    image = image[z_offset[0]:z_offset[1], :, :]
            print(f"{key} data shape after downsampling", image.shape)
            # if not key == "raw":
            #     print(np.unique(image))

        except KeyError:
            print(f"Error: {key} dataset not found in {path}")
            return None  # Indicate error

        return image


def get_all_keys_from_h5(file_path):
    keys = []
    with h5py.File(file_path, 'r') as h5file:
        def collect_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.append(name)  # Add each key (path) to the list
        h5file.visititems(collect_keys)  # Visit all groups and datasets
    return keys


def get_all_dataset_keys(file_path):
    """
    Returns a list of all dataset keys in a file (HDF5, Zarr, or N5).
    
    Parameters:
        file_path (str): Path to the file or directory.
        
    Returns:
        keys (list): List of dataset keys (paths).
    """
    keys = []

    if os.path.isfile(file_path) and file_path.endswith(('.h5', '.hdf5')):
        # HDF5
        with h5py.File(file_path, 'r') as h5file:
            def collect_keys(name, obj):
                if isinstance(obj, h5py.Dataset):
                    keys.append(name)
            h5file.visititems(collect_keys)

    else:
        # Assume Zarr or N5 directory
        store = zarr.N5Store(file_path) if 'attributes.json' in os.listdir(file_path) else zarr.DirectoryStore(file_path)
        root = zarr.open(store, mode='r')

        def collect_keys(name, obj):
            if isinstance(obj, zarr.core.Array):
                keys.append(name)
        root.visititems(collect_keys)

    return keys


def main(visualize=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/embl/cutout_2/images/ome-zarr/raw.ome.zarr", help="Path to the root data directory")
    parser.add_argument("--file_extension", "-fe",  type=str, default=".zarr", help="Path to the root data directory")
    parser.add_argument("--key", "-k",  type=str, default="0", help="Path to the root data directory")
    parser.add_argument("--label_path", "-lp",  type=str, default=None, help="Path to a specific label file")
    parser.add_argument("--label_key", "-lk",  type=str, default=None, help="Key to label data within the label file")
    parser.add_argument("--export_path", "-e",  type=str, default="/scratch-grete/usr/nimlufre/volume-em/mitochondria/test_segmentations", help="Path to the root data directory")
    parser.add_argument("--model_path", "-m", type=str, default="/scratch-grete/projects/nim00007/models/exports_for_cooper/mito_model_s2.pt")
    parser.add_argument("--add_missing_mitos", "-am", default=False, action='store_true', help="If to add missing mitos to segmentation and keep original labels")
    # parser.add_argument("--resize", "-r", default=False, action='store_true', help="Resize to some shape")
    parser.add_argument("--seed_distance", "-sd", type=int, default=6*2, help="Seed distance")
    parser.add_argument("--boundary_threshold", "-bt", type=float, default=0.15, help="Boundary threshold")
    parser.add_argument("--tile_shape", "-ts", type=int, nargs=3, default=(32, 512, 512), help="Tile shape")
    parser.add_argument("--all_keys", "-ak", default=False, action='store_true', help="If to add all keys from raw file to export file")
    parser.add_argument("--force_overwrite", "-fo", action="store_true", default=False, help="Force overwrite of existing files")
    
    args = parser.parse_args()
    add_missing_mitos = args.add_missing_mitos
    print(args.base_path)
    print("\nUsing model", args.model_path)
    # tile_shape
    z, y, x = args.tile_shape
    ts = {
        "z": z,
        "y": y,
        "x": x
        }
    halo = {
        "z": int(ts["z"] * 0.25),
        "y": int(ts["y"] * 0.25),
        "x": int(ts["x"] * 0.25)
        }
    # halo = {'z': 12, 'y': 128, 'x': 128}
    # ts = {'z': ts["z"]+2*halo["z"], 'y': ts["y"]+2*halo["y"], 'x': ts["x"]+2*halo["x"]}
    h5_paths = io.load_file_paths(args.base_path, args.file_extension)
    # if args.file_extension == ".h5":
    #     h5_paths = sorted(glob(os.path.join(args.base_path, "**", "*.h5"), recursive=True), reverse=True)
    # else:
    #     h5_paths = sorted(glob(os.path.join(args.base_path, "**", "*"+args.file_extension), recursive=True), reverse=True)

    print("len(h5_paths)", len(h5_paths))
    tiling = {"tile": ts, "halo": halo}  # prediction function automatically subtracts the 2*halo from tile
    print("tiling:", tiling)
    scale = None

    for path in tqdm(h5_paths):

        print("opening file", path)
        os.makedirs(args.export_path, exist_ok=True)
        output_path = os.path.join(args.export_path, os.path.basename(args.model_path).replace(".pt", "") +
                                   f"_sd{args.seed_distance}_bt{args.boundary_threshold}_with_pred_" +
                                   os.path.basename(path))
        output_path = output_path.replace(".zarr", ".h5")
        if os.path.exists(output_path) and not args.force_overwrite:
            print("Skipping... output path exists", output_path)
            continue
        elif os.path.exists(output_path) and args.force_overwrite:
            print("Overwriting... output path exists", output_path)
            os.remove(output_path)
        if path.endswith(".h5"):
            keys = get_all_keys_from_h5(path)
        else:
            keys = get_all_dataset_keys(path)
        data = {}
        scale_factor = 1
        with open_file(path, "r") as f:
            if args.key is not None and not args.all_keys:
                image = f[args.key][::scale_factor, ::scale_factor, ::scale_factor]
            else:
                # max_shape = (128, 1024, 1024)
                max_shape = (200, 1600, 1600)
                slices = None
                for idx, key in enumerate(keys):
                    arr = f[key][...]
                    if slices is None:
                        # Compute centered crop slices once
                        slices = []
                        for i, max_sz in enumerate(max_shape):
                            if arr.shape[i] > max_sz:
                                start = (arr.shape[i] - max_sz) // 2
                                slices.append(slice(start, start + max_sz))
                            else:
                                slices.append(slice(None))
                        slices = tuple(slices)
                    data[key] = arr[slices]
                    # data[key] = f[key][:128, :512*2, :512*2]
            orig_shape = None
            # if args.resize:
            #     orig_shape = data[args.key].shape
            #     raw = data[args.key]
            #     image = resize(
            #         raw,
            #         output_shape=(
            #             int(raw.shape[0] * 1.67),
            #             int(raw.shape[1] * 1.33),
            #             int(raw.shape[2] * 1.33),
            #         ),
            #         order=1,
            #         preserve_range=True,
            #         anti_aliasing=True
            #         )

            # image = torch_em.transform.raw.standardize(image)
            image = torch_em.transform.raw.normalize_percentile(data[args.key])
        uniq = np.unique(data[args.key])
        print("np unique raw", uniq)
        if np.all(uniq == 0):
            print("Skipping empty file", path)
            continue

        seg, pred = segment_mitochondria(
            image, args.model_path,
            scale=scale,
            tiling=tiling,
            return_predictions=True,
            min_size=5000,  # 50000*1,
            seed_distance=args.seed_distance,  # default 6
            ws_block_shape=(128, 256, 256),
            ws_halo=(48, 48, 48),
            boundary_threshold=args.boundary_threshold,
            area_threshold=500,
            )
        with open_file(output_path, "w", ".h5") as f1:
            print("output_path", output_path)

            # Save all relevant keys if requested, else save raw
            if args.key is not None and args.all_keys:
                print("keys", keys)
                for key in keys:
                    if "mito" in key and add_missing_mitos:
                        added = label(data[key] + find_additional_objects(data[key], seg, matching_threshold=0.1))
                        f1.create_dataset(key, data=added, compression="gzip")
                    else:
                        f1.create_dataset(key, data=data[key], compression="gzip")

            f1.create_dataset("seg", data=seg, compression="gzip", dtype=seg.dtype)

            f1.create_dataset("pred/foreground", data=pred[0], compression="gzip", dtype=pred.dtype)
            f1.create_dataset("pred/boundary", data=pred[1], compression="gzip", dtype=pred.dtype)

            # Optionally include additional label datasets
            if args.label_path is not None:
                with open_file(args.label_path, "r") as f2:
                    if args.label_key is not None:
                        f1.create_dataset(f"labels/{args.label_key}", data=f2[args.label_key], compression="gzip", dtype=f2[args.label_key].dtype)
                    else:
                        for key2 in f2.keys():
                            f1.create_dataset(f"labels/{key2}", data=f2[key2], compression="gzip", dtype=f2[key2].dtype)
                print("Saved original segmentation from", args.label_path, "to", output_path)
            print("Saved to", output_path)


if __name__ == "__main__":
    main()
