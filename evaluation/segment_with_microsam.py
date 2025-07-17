import argparse
import os
from glob import glob
import h5py
import zarr
import torch
import torch_em
import torch_em.transform
from tqdm import tqdm
from elf.io import open_file
import numpy as np
import synapse.io.util as io
# from synapse_net.inference.mitochondria import segment_mitochondria
# from synapse_net.ground_truth.matching import find_additional_objects
from elf.evaluation.matching import label_overlap, intersection_over_union
from skimage.segmentation import relabel_sequential
from skimage.measure import label
from skimage.transform import resize

import micro_sam.evaluation.inference as inference
import micro_sam.instance_segmentation as instance_segmentation
from micro_sam.instance_segmentation import (
    mask_data_to_segmentation, get_predictor_and_decoder,
    AutomaticMaskGenerator, InstanceSegmentationWithDecoder,
    TiledAutomaticMaskGenerator, TiledInstanceSegmentationWithDecoder,
)


def run_prediction(data, checkpoint, use_tiling=True):
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter
    from synapse.sam_util import get_decoder_outputs

    if use_tiling:
        tile_shape, halo = (384, 384), (64, 64)
    else:
        tile_shape, halo = None, None

    predictor, segmenter = get_predictor_and_segmenter("vit_b", checkpoint=checkpoint, is_tiled=use_tiling)

    foreground, center_dists, boundary_dists = get_decoder_outputs(
        predictor, segmenter, data, tile_shape=tile_shape, halo=halo, batch_size=4, verbose=True,
    )

    res = {
        "prediction/foreground": foreground,
        "prediction/center_dists": center_dists,
        "prediction/boundary_dists": boundary_dists
    }

    return res
    # with open_file(save_path, mode="a") as f:
    #     g = f.require_group(name)
    #     ds = g.require_dataset("prediction/foreground", shape=shape, compression="lzf", dtype="float32")
    #     ds[:] = foreground
    #     ds = g.require_dataset("prediction/center_dists", shape=shape, compression="lzf", dtype="float32")
    #     ds[:] = center_dists
    #     ds = g.require_dataset("prediction/boundary_dists", shape=shape, compression="lzf", dtype="float32")
    #     ds[:] = boundary_dists


def export_to_h5(data, export_path):
    with h5py.File(export_path, 'x') as h5f:
        for key in data.keys():
            h5f.create_dataset(key, data=data[key], compression="lzf")
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
            if isinstance(obj, h5py.Dataset) and ("raw" in name or "all" in name):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/scratch-grete/projects/nim00007/data/cellmap/resized_crops/", help="Path to the root data directory")
    parser.add_argument("--file_extension", "-fe",  type=str, default=".h5", help="Path to the root data directory")
    parser.add_argument("--key", "-k",  type=str, default="raw", help="Path to the root data directory")
    parser.add_argument("--label_key", "-lk",  type=str, default="label_crop/all", help="Key for the labels within file")
    parser.add_argument("--export_path", "-e",  type=str, default="/scratch-grete/usr/nimlufre/cellmap/test_segmentations", help="Path to the root data directory")
    parser.add_argument("--model_type", "-mt",  type=str, default="vit-b", help="Type of the model: vit-b")
    parser.add_argument("--model_path", "-m", type=str, default="/scratch-grete/usr/nimlufre/cellmap/checkpoints/micosam-cellmaps-bs1-ps256-resized-mitos")
    parser.add_argument("--force_override", "-fo", action="store_true", help="Force overwrite of existing files")
    parser.add_argument("--block_shape", "-bs",  type=int, nargs=3, default=(1, 128, 128), help="Path to the root data directory")
    parser.add_argument("--halo", "-halo",  type=int, nargs=3, default=None, help="Path to the root data directory")
    parser.add_argument("--with_distances", "-wd",  action="store_true", default=False, help="Save distances as well.")
    args = parser.parse_args()
    print(args.base_path)
    os.makedirs(args.export_path, exist_ok=True)

    # h5_paths = io.load_file_paths(args.base_path, args.file_extension)
    # test file paths for model:
    # /scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/net32-bs8-128-lr1e-4-cellmap-medium-organelles-gldp
    h5_paths = ['/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_313.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_199.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_43.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_39.h5']
    val_paths = ['/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_291.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_240.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_276.h5']
    print("len(h5_paths)", len(h5_paths))
    
    # predictor, decoder = get_predictor_and_decoder(
    #     model_type=args.model_type, checkpoint_path=args.model_path, peft_kwargs=None,
    # )
    # grid_search_values = instance_segmentation.default_grid_search_values_instance_segmentation_with_decoder

    if args.with_distances:
        breakpoint()
        checkpoint_path = os.path.join(args.model_path, "best.pt")
        sample_path = h5_paths[0]
        for sample_path in h5_paths:
            pred_path = os.path.join(args.export_path, os.path.basename(sample_path))
            if os.path.exists(pred_path) and not args.force_override:
                print("Skipping, because it alöready exists", pred_path)
                continue
            raw = _read_h5(sample_path, args.key, 1)
            res = run_prediction(
                data=raw,
                checkpoint=checkpoint_path,
                use_tiling=True if any(dim > 256 for dim in raw.shape) else False,
            )
            out = {
                "raw": raw,
            }
            out.update(res)
            
            export_to_h5(out, pred_path)
    else:
        pred_path = inference.run_instance_segmentation_with_decoder(
            checkpoint=args.model_path,
            model_type=args.model_type,
            experiment_folder=args.export_path,
            test_image_paths=h5_paths,
            val_image_paths=val_paths,
            val_gt_paths=val_paths,
            gt_key=args.label_key,
            image_key=args.key,
        )


if __name__ == "__main__":
    main()