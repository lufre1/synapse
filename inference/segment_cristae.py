import argparse
import os
from glob import glob
import h5py
import torch_em
import torch_em.transform
from tqdm import tqdm
from elf.io import open_file
import numpy as np
from synapse_net.inference.cristae import segment_cristae
# from synapse_net.ground_truth.matching import find_additional_objects
from elf.evaluation.matching import label_overlap, intersection_over_union
from skimage.segmentation import relabel_sequential
from skimage.measure import label
import synapse.util as util


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


def main(visualize=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/wichmann/extracted/", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/cristae_test_segmentations", help="Path to the root data directory")
    parser.add_argument("--model_path", "-m", type=str, default="/scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/cristae-net32-bs2-ps48512-cooper-wichmann-new-transform")
    parser.add_argument("--add_missing", "-am", default=False, action='store_true', help="If to add missing objects to segmentation and keep original labels")
    parser.add_argument("--tile_shape", "-ts", type=int, nargs=3, default=(32, 512, 512), help="Tile shape")
    args = parser.parse_args()
    add_missing = args.add_missing
    print(args.base_path)
    os.makedirs(args.export_path, exist_ok=True)
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
    # h5_paths = ['/mnt/lustre-grete/usr/u12103/mitochondria/cooper/fidi_2025/raw_mitos_combined_s2/37371_O5_66K_TS_SP_34-01_rec_2Kb1dawbp_cropF_s2_combined.h5']
    h5_paths = util.get_file_paths(args.base_path, ".h5")
    test_file_paths = h5_paths
    print("len(h5_paths)", len(h5_paths))
    tiling = {"tile": ts, "halo": halo}  # prediction function automatically subtracts the 2*halo from tile
    print("tiling:", tiling)
    scale = None  # [1.0, 1.0, 1.0]
    global raw_transform
    raw_transform = util.standardize_channel

    for path in tqdm(test_file_paths):
        # skip = True
        # if "KO8_eb2_model" in path or "KO9_eb6_model" in path or "M7_eb2_model" in path:
        #     # breakpoint()
        #     skip = False
        # if skip:
        #     continue
        print("opening file", path)
        filename, inter_dirs = util.get_filename_and_inter_dirs(path, args.base_path)
        output_path = os.path.join(args.export_path, inter_dirs, filename + ".h5")
        util.create_directories_if_not_exists(args.export_path, inter_dirs)
        if os.path.exists(output_path):
            print("Skipping... output path exists", output_path)
            continue
        keys = get_all_keys_from_h5(path)
        data = {}
        scale_factor = None
        with open_file(path, "r") as f:
            for key in keys:
                data[key] = f[key][:]
                # if "raw" in key:
                #     data[key] = f[key][:]
            # image = util.standardize_channel(data["raw_mitos_combined"])
            # image = util.normalize_percentile_with_channel_cgpt(data["raw_mitos_combined"])
            image = data["raw_mitos_combined"]
            print("image shape", image.shape)
            # raw = torch_em.transform.raw.normalize_percentile(data["raw_mitos_combined"][0])
            # image = np.stack([raw, data["raw_mitos_combined"][1]], axis=0)
        kwargs = {
            "extra_segmentation": image[1],
            "channels_to_standardize": [0],
            "with_channels": True,
            }
        seg, pred = segment_cristae(
            image[0], args.model_path,
            scale=scale,
            tiling=tiling,
            return_predictions=True,
            **kwargs
            )
        if add_missing:
            with open_file(output_path, "w") as f1:
                print("output_path", output_path)
                print("keys", keys)
                for key in keys:
                    f1[key] = data[key]
                    if add_missing and "cristae" in key:
                        additional_objects = find_additional_objects(data[key], seg, matching_threshold=0.1)
                        f1[key] = label(data[key] + additional_objects)

                f1["pred"] = pred
                f1["labels/new_cristae_seg"] = seg
                print("Saved to", output_path)
                print("\n")
        else:
            data["pred/foreground"] = pred[0]
            data["pred/boundary"] = pred[1]
            data["seg"] = seg
            data["raw"] = data["raw_mitos_combined"][0]
            data["labels/mitochondria"] = data["raw_mitos_combined"][1]
            del data["raw_mitos_combined"]
            util.export_data(output_path, data)


if __name__ == "__main__":
    main()