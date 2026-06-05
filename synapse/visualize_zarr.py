# import time
from typing import Any, Dict
import synapse.util as util
import synapse.io.util as io
# from config import *
import h5py
import zarr
import argparse
import os
from glob import glob
import numpy as np
import torch_em
import napari
# import elf.parallel as parallel
from elf.io import open_file
from tqdm import tqdm
import z5py
from tifffile import imread
from skimage.transform import resize

# Shared implementations (consolidated): see synapse.io.util / synapse.h5_util.
from synapse.io.util import get_file_paths, extract_data, upsample_data  # noqa: F401


def visualize_data(data):
    viewer = napari.Viewer()
    for key, value in data.items():
        if key == "raw" or "raw" in key or "0_0" in key:
            # if data[key].ndim == 4:
            #     data[key] = util.normalize_percentile_with_channel(data[key], lower=1, upper=99, channel=0)
            # else:
            #     value = torch_em.transform.raw.normalize_percentile(value, lower=1, upper=99)
            viewer.add_image(value, name=key)
        elif key == "prediction" or "pred" in key:
            viewer.add_image(value, name=key, blending="additive")
        else:
            viewer.add_labels(value, name=key)
    # Get the "raw" layer
    raw_layer = next((layer for layer in viewer.layers if "raw" in layer.name), None)
    if raw_layer:
        # Remove the "raw" layer from its current position
        viewer.layers.remove(raw_layer)
        # Add the "raw" layer to the beginning of the layer list
        viewer.layers.insert(0, raw_layer)

    napari.run()


def main(root_path: str, ext: str = None, scale: int = 1, upsample: bool = False, root_label_path: str = None, args=None):
    if ext is None:
        print("Loading h5, n5 and zarr files")
        paths = get_file_paths(root_path, ".h5")
        paths.extend(get_file_paths(root_path, ".n5"))
        paths.extend(get_file_paths(root_path, ".zarr"))
        paths.extend(get_file_paths(root_path, ".mrc"))
        paths.extend(get_file_paths(root_path, ".rec"))
    else:
        paths = get_file_paths(root_path, ext)
    paths[:] = [path for path in paths if "embedding" not in path]
    paths[:] = [path for path in paths if "mitos.ome" not in path]
    if root_label_path is not None:
        label_paths = get_file_paths(root_label_path, ".tif")
    else:
        label_paths = None
    print("Found files:", len(paths))
    if args is not None:
        if args.all:
            all_data = {}
            for i, path in enumerate(tqdm(paths)):
                data = io.load_data_from_file(path, scale=scale, upsample=upsample, label_paths=label_paths)
                for key, value in data.items():
                    all_data[f"{key}_{i}"] = value
            for k in all_data.keys():
                print(k, all_data[k].shape)
            visualize_data(all_data)
            return
    for path in tqdm(paths):
        print("\n", path)
        if label_paths is not None:
            label_path = util.find_label_file(path, label_paths)
        else:
            label_path = None
        with open_file(path, mode="r") as f:
            data = {}
            if label_path is not None:
                print("Loading label data from", label_path)
                if "data" in f.keys():
                    ndim = f["data"].ndim
                elif "raw" in f.keys():
                    ndim = f["raw"].ndim
                else:
                    print("Warning! Assuming NDIM = 3")
                    ndim = 3
                slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                data["label"] = imread(label_path)[slicing] if scale > 1 else imread(label_path)
            else:
                print("No specific label path loaded.")
            if ".mrc" in path or ".rec" in path:
                ndim = f["data"].ndim
                slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                data["raw"] = f["data"][slicing] if scale > 1 else f["data"][:]
            else:
                print(f.keys())
                for key in f.keys():
                    if isinstance(f[key], (zarr.Group, h5py.Group, z5py.Group)):
                        print(f"Loading group: {key}")
                        extract_data(f[key], data, scale=scale)
                        continue
                    ndim = f[key].ndim

                    # Generate a slicing tuple based on the number of dimensions
                    slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                    # Apply downsampling while preserving batch/channel dimensions
                    data[key] = f[key][slicing] if scale > 1 else f[key][:]

        if upsample:
            del data["pred"]
            del data["raw"]

            for key in data.keys():
                data[key] = upsample_data(data[key], upsample)

        raw_shape = None
        for k in data.keys():
            if "raw" in k:
                raw_shape = data[k].shape
        if raw_shape:
            for k in data.keys():
                if "raw" not in k:
                    if raw_shape != data[k].shape:
                        print(f"Resizing {k} from {data[k].shape} to {raw_shape}")
                        data[k] = util.downsample_to_shape(data[k], raw_shape)
        visualize_data(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str)
    parser.add_argument("--all", "-a", action="store_true", default=False, help="Load all files in path")
    parser.add_argument("--ext", "-e", type=str, default=None)
    parser.add_argument("--scale", "-s", type=int, default=1)
    parser.add_argument("--upsample", "-u", type=int, default=None)
    parser.add_argument("--label_path", "-lp", type=str, default=None)
    args = parser.parse_args()
    path = args.path
    ext = args.ext
    scale = args.scale
    upsample = args.upsample
    label_path = args.label_path
    main(path, ext, scale, upsample, label_path, args)
