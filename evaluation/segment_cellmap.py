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


def main(visualize=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/scratch-grete/projects/nim00007/data/cellmap/resized_crops/", help="Path to the root data directory")
    parser.add_argument("--file_extension", "-fe",  type=str, default=".h5", help="Path to the root data directory")
    parser.add_argument("--key", "-k",  type=str, default="raw", help="Path to the root data directory")
    parser.add_argument("--export_path", "-e",  type=str, default="/scratch-grete/projects/nim00007/data/cellmap/test_segmentations", help="Path to the root data directory")
    parser.add_argument("--model_path", "-m", type=str, default="/scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/net32-bs8-128-lr1e-4-cellmap-medium-organelles-gldp")
    parser.add_argument("--force_override", "-fo", action="store_true", help="Force overwrite of existing files")
    parser.add_argument("--block_shape", "-bs",  type=int, nargs=3, default=(128, 128, 128), help="Path to the root data directory")
    parser.add_argument("--halo", "-halo",  type=int, nargs=3, default=None, help="Path to the root data directory")
    args = parser.parse_args()
    print(args.base_path)

    # h5_paths = io.load_file_paths(args.base_path, args.file_extension)
    # test file paths for model:
    # /scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/net32-bs8-128-lr1e-4-cellmap-medium-organelles-gldp
    h5_paths = ['/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_143.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_39.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_190.h5']
    # test file paths for model:
    # /scratch-grete/usr/nimlufre/synapse/mito_segmentation/checkpoints/mitonet32-bs4-128-lr1e-4-cellmap-medium-organelles-weighted75
    h5_paths = ['/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_129.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_248.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_275.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_188.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_38.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_8.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_259.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_156.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_133.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_40.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_231.h5']
    # er test data
    h5_paths = ['/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_219.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_175.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_178.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_8.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_121.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_235.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_148.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_239.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_37.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_269.h5']

    print("len(h5_paths)", len(h5_paths))
    # tiling = {"tile": ts, "halo": halo}  # prediction function automatically subtracts the 2*halo from tile
    # print("tiling:", tiling)
    # scale = None

    for path in tqdm(h5_paths):

        print("opening file", path)
        os.makedirs(args.export_path, exist_ok=True)
        output_path = os.path.join(args.export_path, os.path.basename(args.model_path).replace(".pt", "_") + "_" + os.path.basename(path))
        output_path = output_path.replace(".h5", "_with_pred.h5")
        print("args.model path basename", os.path.basename(args.model_path))
        if os.path.exists(output_path) and not args.force_override:
            print("Skipping... output path exists", output_path)
            continue
        if path.endswith(".h5"):
            keys = get_all_keys_from_h5(path)
        else:
            keys = get_all_dataset_keys(path)
        data = {}
        scale_factor = 1
        if args.halo is None:
            halo = [int(x * 0.1) for x in args.block_shape]
            block_shape = [int(x - 2*h) for x, h in zip(args.block_shape, halo)]
        else:
            halo = args.halo
            block_shape = args.block_shape
        print("blocksize and halo", block_shape, halo)
        with open_file(path, "r") as f:
            for key in keys:
                data[key] = f[key][::scale_factor, ::scale_factor, ::scale_factor]

            # image = torch_em.transform.raw.standardize(image)
            image = torch_em.transform.raw.normalize_percentile(data["raw"])

        out = torch_em.util.prediction.predict_with_halo(
            image,
            torch_em.util.util.load_model(args.model_path),
            gpu_ids=[0],
            block_shape=block_shape,
            halo=halo,
            )
        with open_file(output_path, "w", ".h5") as f1:
            print("output_path", output_path)
            print("keys", keys)
            for key in keys:
                f1.create_dataset(key, data=data[key], compression="gzip")
            for i in range(0, out.shape[0]):
                f1.create_dataset(f"pred_{i}", data=out[i], compression="gzip")
            print("Saved to", output_path)


if __name__ == "__main__":
    main()