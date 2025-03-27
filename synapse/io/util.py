import h5py
from typing import Any, Dict
import zarr
import os
from glob import glob
import numpy as np
import torch_em
import napari
import z5py
from skimage.transform import resize
import synapse.util as util
from tqdm import tqdm
from elf.io import open_file
from tifffile import imread


def load_file_paths(root_path: str, ext: str = None,
                    root_label_path: str = None) -> Dict[str, Any]:
    if ext is None:
        print("Loading h5, n5, zarr, mrc and rec files")
        paths = get_file_paths(root_path, ".h5")
        paths.extend(get_file_paths(root_path, ".n5"))
        paths.extend(get_file_paths(root_path, ".zarr"))
        paths.extend(get_file_paths(root_path, ".mrc"))
        paths.extend(get_file_paths(root_path, ".rec"))
    else:
        paths = get_file_paths(root_path, ext)
    if root_label_path is not None:
        label_paths = get_file_paths(root_label_path, ".tif")
        return paths, label_paths
    else:
        label_paths = None
    print("Found files:", len(paths))
    return paths


def load_data_from_file(path: str, scale: int = 1, upsample: int = 0,
                        label_paths: str = None
                        ) -> Dict[str, Any]:
    if label_paths is not None:
        label_path = util.find_label_file(path, label_paths)
    else:
        label_path = None
    with open_file(path, mode="r") as f:
        data = {}
        if label_path is not None:
            print("Loading label data from", label_path)
            ndim = f["data"].ndim
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

    # new_seg = _run_segmentation(data["pred"][0], min_size=1000, verbose=True)
    # data["new_segmentation"] = new_seg
    if upsample:
        del data["pred"]
        del data["raw"]

        for key in data.keys():
            data[key] = upsample_data(data[key], upsample)

    return data


def load_data(root_path: str, ext: str = None, scale: int = 1,
              upsample: bool = False, root_label_path: str = None) -> Dict[str, Any]:
    if ext is None:
        print("Loading h5, n5, zarr, mrc and rec files")
        paths = get_file_paths(root_path, ".h5")
        paths.extend(get_file_paths(root_path, ".n5"))
        paths.extend(get_file_paths(root_path, ".zarr"))
        paths.extend(get_file_paths(root_path, ".mrc"))
        paths.extend(get_file_paths(root_path, ".rec"))
    else:
        paths = get_file_paths(root_path, ext)
    if root_label_path is not None:
        label_paths = get_file_paths(root_label_path, ".tif")
    else:
        label_paths = None
    print("Found files:", len(paths))
    files = {}
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
                ndim = f["data"].ndim
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

        # new_seg = _run_segmentation(data["pred"][0], min_size=1000, verbose=True)
        # data["new_segmentation"] = new_seg
        if upsample:
            del data["pred"]
            del data["raw"]

            for key in data.keys():
                data[key] = upsample_data(data[key], upsample)

        files[path] = data


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
            if data[key].ndim == 4:
                data[key] = util.normalize_percentile_with_channel(data[key], lower=1, upper=99, channel=0)
            else:
                value = torch_em.transform.raw.normalize_percentile(value, lower=1, upper=99)
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


def extract_data(group: Any, data: Dict[str, Any], prefix: str = "", scale: int = 1):
    """
    Recursively extract datasets from a group and store them in a dictionary.
    """
    for key, item in group.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(item, (zarr.Group, h5py.Group, z5py.Group)):
            # Recursively extract data from subgroups
            extract_data(item, data, prefix=full_key, scale=scale)
        else:
            ndim = item.ndim
            # Generate a slicing tuple based on the number of dimensions
            slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
            
            # Apply downsampling while preserving batch/channel dimensions
            data[full_key] = item[slicing] if scale > 1 else item[:]
            # # Store the dataset in the dictionary
            # data[full_key] = item[:]


def upsample_data(data, factor):
    """Upsample a 3D dataset in chunks to avoid memory overload."""
    upsampled_data = np.zeros(tuple(dim * factor for dim in data.shape), dtype=data.dtype)
    for z in range(data.shape[0]):
        upsampled_data[z * factor : (z + 1) * factor] = resize(
            data[z], 
            (factor * data.shape[1], factor * data.shape[2]),
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(data.dtype)
    return upsampled_data