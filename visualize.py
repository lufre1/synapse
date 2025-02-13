import time
from typing import Any, Dict
import synapse.util as util
from config import *
import h5py
import zarr
import argparse
import os
from glob import glob
import numpy as np
import torch_em
import napari
import elf.parallel as parallel
from elf.io import open_file
from tqdm import tqdm
import z5py

from synapse_net.inference.cristae import _run_segmentation

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import remove_small_objects, binary_closing, ball


def get_file_paths(path, ext=".h5", reverse=False):
    if ext in path:
        return [path]
    else:
        paths = sorted(glob(os.path.join(path, "**", f"*{ext}"), recursive=True), reverse=reverse)
        return paths


def visualize_data(data):
    viewer = napari.Viewer()
    for key, value in data.items():
        if key == "raw" or "raw" in key:
            value = torch_em.transform.raw.normalize_percentile(value, lower=5, upper=95)
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


def extract_data(group: Any, data: Dict[str, Any], prefix: str = ""):
    """
    Recursively extract datasets from a group and store them in a dictionary.
    """
    for key, item in group.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(item, (zarr.Group, h5py.Group, z5py.Group)):
            # Recursively extract data from subgroups
            extract_data(item, data, prefix=full_key)
        else:
            # Store the dataset in the dictionary
            data[full_key] = item[:]


def main(path: str, ext: str = None):
    if ext is None:
        print("Loading h5, n5 and zarr files")
        paths = get_file_paths(path, ".h5")
        paths.extend(get_file_paths(path, ".n5"))
        paths.extend(get_file_paths(path, ".zarr"))
    else:
        paths = get_file_paths(path, ext)
    for path in paths:
        print(path)
        with open_file(path) as f:
            print(f.keys())
            data = {}
            for key in f.keys():
                if isinstance(f[key], (zarr.Group, h5py.Group, z5py.Group)):
                    print(f"Loading group: {key}")
                    extract_data(f[key], data)
                    continue
                data[key] = f[key][:]
        # new_seg = _run_segmentation(data["pred"][0], min_size=1000, verbose=True)
        # data["new_segmentation"] = new_seg
        visualize_data(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str)
    parser.add_argument("--ext", "-e", type=str, default=".h5")
    args = parser.parse_args()
    path = args.path
    ext = args.ext
    main(path, ext)
