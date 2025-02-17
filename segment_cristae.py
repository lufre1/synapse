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
    args = parser.parse_args()
    add_missing = args.add_missing
    print(args.base_path)
    # tile_shape
    ts = {
        "z": 48,
        "y": 512,
        "x": 512
        }
    halo = {
        "z": int(ts["z"] * 0.1),
        "y": int(ts["y"] * 0.25),
        "x": int(ts["x"] * 0.25)
        }
    # halo = {'z': 12, 'y': 128, 'x': 128}
    ts = {'z': ts["z"]+2*halo["z"], 'y': ts["y"]+2*halo["y"], 'x': ts["x"]+2*halo["x"]}
    h5_paths = sorted(glob(os.path.join(args.base_path, "**", "*.h5"), recursive=True), reverse=True)
    # test_file_paths = [
        # "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M2_eb10_model.h5",
        # '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/WT21_eb3_model2.h5',
        # '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M10_eb9_model.h5',
        # '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/KO9_eb4_model.h5',
        # '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M7_eb11_model.h5',
        # '/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2/36859_J1_66K_TS_CA3_PS_25_rec_2Kb1dawbp_crop_downscaled.h5'
    # ]
    ### fidi
    # test_file_paths = [
    #     '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M2_eb10_model.h5', '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/WT21_eb3_model2.h5', '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M10_eb9_model.h5', '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/KO9_eb4_model.h5', '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/refined_mitos/M7_eb11_model.h5', '/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2/36859_J1_66K_TS_CA3_PS_25_rec_2Kb1dawbp_crop_downscaled.h5'
        
    # ]
    test_file_paths = ['/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_and_cristae/Otof-WT_P10/WT20_eb7_AZ1_model2_combined.h5', '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_and_cristae/Otof-WT_P10/WT20_syn7_model2_combined.h5', '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_and_cristae/Otof-WT_P10/WT20_syn2_model2_combined.h5', '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_and_cristae/Otof-KO_P10/M1_eb1_model_combined.h5', '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_in_endbuld/Otof_AVCN03_429C_WT_M.Stim_G3_1_model_combined.h5', '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_and_cristae/Otof-WT_P10/WT13_syn6_model2_combined.h5', '/scratch-grete/projects/nim00007/data/mitochondria/wichmann/raw_mito_combined/mitos_and_cristae/Otof-KO_P22/M7_eb12_model_combined.h5']

    print("len(h5_paths)", len(h5_paths))
    tiling = {"tile": ts, "halo": halo}  # prediction function automatically subtracts the 2*halo from tile
    print("tiling:", tiling)
    scale = None  # [1.0, 1.0, 1.0]

    for path in tqdm(test_file_paths):
        # skip = True
        # if "KO8_eb2_model" in path or "KO9_eb6_model" in path or "M7_eb2_model" in path:
        #     # breakpoint()
        #     skip = False
        # if skip:
        #     continue
        print("opening file", path)
        output_path = os.path.join(args.export_path, os.path.basename(args.model_path) + "_new-transform_" + os.path.basename(path)).replace(".h5", ".n5")
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
            image = util.normalize_percentile_with_channel(data["raw_mitos_combined"])
            # raw = torch_em.transform.raw.normalize_percentile(data["raw_mitos_combined"][0])
            # image = np.stack([raw, data["raw_mitos_combined"][1]], axis=0)
        seg, pred = segment_cristae(
            image, args.model_path,
            scale=scale,
            tiling=tiling,
            return_predictions=True,
            )
        with open_file(output_path, "w", ".n5") as f1:
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


if __name__ == "__main__":
    main()