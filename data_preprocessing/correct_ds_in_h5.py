import synapse.util as util
from config import *
import h5py
import argparse
import os
from glob import glob
import numpy as np
import torch_em
import napari
import elf.parallel as parallel
from scipy.ndimage import binary_erosion, binary_fill_holes, binary_closing
from synapse_net.inference.util import apply_size_filter, get_prediction, _Scaler, _postprocess_seg_3d


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


def get_file_paths(path):
    if os.path.isfile(path):
        return [path]
    else:
        paths = sorted(glob(os.path.join(path, "**", "*.h5"), recursive=True))
        print(f"Found {len(paths)} files:")
        return paths


def visualize_data(data):
    viewer = napari.Viewer()
    for key, value in data.items():
        if key == "raw" or "raw" in key:
            value = torch_em.transform.raw.standardize(value)
            viewer.add_image(value, name=key)
        elif key == "prediction" or "pred" in key:
            viewer.add_image(value, name=key, blending="additive")
        else:
            viewer.add_labels(value, name=key)

    napari.run()


def visualize():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--path", "-p", type=str, required=True, help="Path to the data directory or single file")
    parser.add_argument("--scale_factor", "-s", type=int, default=1, help="Scale factor for the data")
    parser.add_argument("--no_visualize", "-nv", action="store_true", default=False, help="Don't visualize data with napari")
    parser.add_argument("--z_offset", "-z", type=int, nargs=2, default=None, help="Z offset for the data e.g. 5 -5")
    args = parser.parse_args()

    paths = get_file_paths(args.path)
    
    for path in paths:
        print(path)
        # if "M7_eb2" not in path:
        #     continue
        # if i < 2:
        #     i += 1
        #     continue
        keys = get_all_keys_from_h5(path)
        keys.sort(reverse=True)
        print("\ndata keys", keys)
        print("in path", path)
        data = {}
        for key in keys:
            data[key] = _read_h5(path, key, args.scale_factor, z_offset=(args.z_offset))
        
        visualize_data(data)