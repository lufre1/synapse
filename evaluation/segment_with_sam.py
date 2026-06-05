import argparse
import os
from glob import glob
from typing import Optional, Tuple
import h5py
import zarr
import torch
import elf.parallel as parallel
from elf.parallel.filters import apply_filter
import multiprocessing as mp
import torch_em
import torch_em.transform
from tqdm import tqdm
from elf.io import open_file
import numpy as np
import synapse.io.util as io
import synapse.label_utils as lutils
import synapse.util as util
import synapse.h5_util as h5_util
import synapse.sam_util as sutil
# from synapse_net.inference.mitochondria import segment_mitochondria
# from synapse_net.ground_truth.matching import find_additional_objects
from elf.evaluation.matching import label_overlap, intersection_over_union
from skimage.segmentation import relabel_sequential
from skimage.measure import label
from skimage.transform import resize
from scipy.ndimage import binary_closing

import micro_sam.evaluation.inference as inference
from micro_sam.multi_dimensional_segmentation import _filter_z_extent
import micro_sam.instance_segmentation as instance_segmentation
from micro_sam.instance_segmentation import (
    mask_data_to_segmentation, get_predictor_and_decoder,
    AutomaticMaskGenerator, InstanceSegmentationWithDecoder,
    TiledAutomaticMaskGenerator, TiledInstanceSegmentationWithDecoder,
)


def _read_zarr(path, key):
    print("debug", path, key)
    f = zarr.open(store=path, mode="r")
    try:
        print(f"{key} data shape", f[key].shape)
        image = f[key][:]
    except KeyError:
        print(f"Error: {key} dataset not found in {path}")
        return None  # Indicate error
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/", help="Path to the root data directory")
    parser.add_argument("--file_extension", "-fe",  type=str, default=".h5", help="Path to the root data directory")
    parser.add_argument("--key", "-k",  type=str, default="raw", help="Path to the root data directory")
    parser.add_argument("--label_directory", "-ld",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/embl/cutout_1/images/committed_objects_leonie_2025-08-07.tif")
    parser.add_argument("--label_key", "-lk",  type=str, default=None, help="Key for the labels within file")
    parser.add_argument("--export_path", "-e",  type=str, default="/scratch-grete/usr/nimlufre/cellmap/test_segmentations_microsam_resized", help="Path to the root data directory")
    parser.add_argument("--model_type", "-mt",  type=str, default="vit_b_em_organelles", help="Type of the model: vit-b")
    parser.add_argument("--model_path", "-m", type=str, default=None)
    parser.add_argument("--force_override", "-fo", action="store_true", default=False, help="Force overwrite of existing files")
    parser.add_argument("--block_shape", "-bs",  type=int, nargs=3, default=(1, 256, 256), help="Path to the root data directory")
    parser.add_argument("--halo", "-halo",  type=int, nargs=3, default=None, help="Path to the root data directory")
    parser.add_argument("--with_distances", "-wd",  action="store_true", default=True, help="Save distance map predictions (other options not implemented atm).")
    args = parser.parse_args()
    print("model path", args.model_path)
    os.makedirs(args.export_path, exist_ok=True)

    h5_paths = io.load_file_paths(args.base_path, args.file_extension)

    label_transform = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            instances=True,
            min_size=10,
        )

    if args.with_distances:
        if args.model_path is not None:
            checkpoint_path = os.path.join(args.model_path, "best.pt") if "best.pt" not in args.model_path else args.model_path
        else:
            checkpoint_path = None
        for sample_path in h5_paths:
            print("Loading", sample_path)
            pred_path = os.path.join(args.export_path, os.path.basename(sample_path))
            if os.path.exists(pred_path) and not args.force_override:
                print("Skipping, because it already exists", pred_path)
                continue
            elif os.path.exists(pred_path) and args.force_override:
                print("overriding", pred_path)
                os.remove(pred_path)
            if ".h5" in sample_path:
                raw = h5_util.read_h5(sample_path, args.key, 1)
            else:
                raw = _read_zarr(sample_path, args.key)
            res = sutil.run_decoder_prediction(
                data=sutil.raw_transform(raw),
                model_type=args.model_type,
                checkpoint=checkpoint_path,
                use_tiling=True if any(dim > 256 for dim in raw.shape) else False,
            )
            out = {
                "raw": raw,
                # "label": label_transform(h5_util.read_h5(sample_path, args.label_key, 1))
            }
            num_channels = label_transform(h5_util.read_h5(sample_path, args.label_key, 1)).shape[0]

            # Dynamically add all channels to the out dictionary
            for i in range(num_channels):
                out[f"label_{i}"] = label_transform(h5_util.read_h5(sample_path, args.label_key, 1))[i, :, :, :]
            out.update(res)
            seg = sutil.volumetric_segmentation(
                foreground=out["prediction/foreground"],
                center_dists=out["prediction/center_dists"],
                boundary_dists=out["prediction/boundary_dists"],
            )
            # seg2 = sutil.alt_segmentation(
            #     foreground=out["prediction/foreground"],
            #     boundary_distances=out["prediction/boundary_dists"],
            # )
            out["segmentation"] = seg
            # out["segmentation2"] = seg2

            util.export_to_h5(out, pred_path, compression="lzf", mode="x")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
