import util
import data_classes
from config import *
import h5py
import argparse
import os
from glob import glob
import numpy as np
import torch_em
import napari


def _read_h5(path, key, scale_factor):
    with h5py.File(path, "r") as f:
        try:
            print(f"{key} data shape", f[key].shape)
            image = f[key][:, ::scale_factor, ::scale_factor]
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
        if key == "raw":
            viewer.add_image(value, name="Raw")
        else:
            viewer.add_labels(value, name=key)

    napari.run()


def visualize():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--path", "-p", type=str, required=True, help="Path to the data directory or single file")
    parser.add_argument("--scale_factor", "-s", type=int, default=1, help="Scale factor for the data")
    parser.add_argument("--no_visualize", "-nv", action="store_true", default=False, help="Don't visualize data with napari")
    args = parser.parse_args()

    paths = get_file_paths(args.path)

    shapes = []
    for path in paths:
        print(path)
        if path == "/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi/36859_J1_66K_TS_CA3_MF_18_rec_2Kb1dawbp_crop.h5":
            continue
        keys = get_all_keys_from_h5(path)
        keys.sort(reverse=True)
        print("data keys", keys)
        print("in path", path)
        data = {}
        for key in keys:
            data[key] = _read_h5(path, key, args.scale_factor)

        if data and not args.no_visualize:
            visualize_data(data)
        if data:
            shapes.append(data["raw"].shape)
            print("min", np.min(data["raw"]))
            print("max", np.max(data["raw"]))
            print("mean", np.mean(data["raw"]))
            print("std", np.std(data["raw"]))
            print("percentile", np.percentile(data["raw"], [0, 25, 50, 75, 100]))
            # print("data.keys", data.keys())
            # shapes = []
            # for key, value in data.items():
            #     print(key, value.shape)
            #     shapes.append(value.shape)
            # print("shapes", shapes)
            # avg0 = np.mean(shapes, axis=0)    
            # avg1 = np.mean(data["raw"].shape, axis=1)    
            # avg2 = np.mean(data["raw"].shape, axis=2)    
            # print(avg0)#, avg1.shape, avg2.shape)
    for shape in shapes:
        print(shape)
    print("average shapes", np.mean(shapes, axis=0))


if __name__ == "__main__":
    visualize()