# import time
from typing import Any, Dict
import synapse.util as util
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
from elf.parallel import label
from tqdm import tqdm
import z5py
from tifffile import imread
from skimage.transform import resize
import dask.array as da


def _segment(foreground, boundary, block_shape=(128, 128, 128), threshold=0.5):
    foreground_mask = np.where(foreground > threshold, 1, 0)
    boundary_mask = np.where(boundary > threshold, 1, 0)
    mask = np.logical_or(foreground_mask, np.logical_and(foreground_mask, boundary_mask))
    seg = label(mask, block_shape=block_shape)
    return seg


def get_file_paths(path, ext=".h5", reverse=False):
    if ext in path:
        return [path]
    else:
        paths = sorted(glob(os.path.join(path, "**", f"*{ext}"), recursive=True), reverse=reverse)
        return paths


def visualize_data(data, name=None, offset_z=None):
    viewer = napari.Viewer()
    if name is not None:
        viewer.title = name
    print("vis data keys", data.keys())
    for key, value in data.items():
        if "raw" in key or key == "0":
            # if data[key].ndim == 4:
            #     data[key] = util.normalize_percentile_with_channel(data[key], lower=1, upper=99, channel=0)
            # else:
            #     value = torch_em.transform.raw.normalize_percentile(value, lower=1, upper=99)
            viewer.add_image(value, name=key)
        elif key == "prediction" or "pred" in key or "dist" in key or "fore" in key or "bound" in key:
            viewer.add_image(value, name=key, blending="additive")
        else:
            if offset_z is not None:
                viewer.add_labels(value.astype(np.uint8), name=key, translate=(0, 0, offset_z))
            viewer.add_labels(value.astype(np.uint8), name=key)
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


def _as_lazy(item, scale: int = 1) -> da.Array:
    """
    Convert a NumPy/Zarr/H5/Z5 dataset to a *lazy* Dask array.
    `scale` is applied only to the last three (spatial) axes.
    """
    # 1️⃣  Already a NumPy array → just wrap it (no copy)
    if isinstance(item, np.ndarray):
        return da.from_array(item, chunks=item.shape)

    # 2️⃣  Zarr / H5 / Z5 objects – they expose a ``chunks`` attribute
    if hasattr(item, "chunks"):
        ndim = item.ndim
        # Down‑sample only the spatial axes (the last three dimensions)
        slicing = tuple(
            slice(None, None, scale) if i >= (ndim - 3) else slice(None)
            for i in range(ndim)
        )
        # Create a lazy Dask array and apply the slicing
        return da.from_array(item, chunks=item.chunks)[slicing]

    # 3️⃣  Fallback – treat anything else as a NumPy‑like object
    return da.from_array(item, chunks=item.shape)


def extract_data_lazy(group, data, prefix="", scale=1):
    for key, item in group.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(item, (zarr.Group, h5py.Group, z5py.Group)):
            extract_data_lazy(item, data, prefix=full_key, scale=scale)
        else:
            data[full_key] = _as_lazy(item, scale)


def upsample_data(data, factor):
    """Upsample a 3D dataset in chunks to avoid memory overload."""
    upsampled_data = np.zeros(tuple(dim * factor for dim in data.shape), dtype=data.dtype)
    for z in range(data.shape[0]):
        upsampled_data[z * factor: (z + 1) * factor] = resize(
            data[z],
            (factor * data.shape[1], factor * data.shape[2]),
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(data.dtype)
    return upsampled_data


def main(root_path: str, ext: str = None, scale: int = 1, upsample: bool = False,
         root_label_path: str = None, segment: bool = False,
         offset_z: int = None):
    if ext is None:
        if os.path.isfile(root_path):
            ext = os.path.splitext(root_path)[1]
            paths = get_file_paths(root_path, ext)
        else:
            print("Loading h5, n5 and zarr files")
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
    for path in tqdm(paths):
        print("\n", path)
        if label_paths is not None and len(label_paths) > 1:
            label_path = util.find_label_file(path, label_paths)
        elif label_paths and len(label_paths) == 1:
            label_path = label_paths[0]
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
                # import skimage as ski
                # data["label"] = ski.morphology.remove_small_objects(data["label"], min_size=1000)
            else:
                print("No specific label path loaded.")
            if ".mrc" in path or ".rec" in path:
                ndim = f["data"].ndim
                slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                data["raw"] = f["data"][slicing] if scale > 1 else f["data"][:]
            elif ".tif" in path:
                ndim = f[""][:].ndim
                slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                data["label"] = f[""][slicing] if scale > 1 else f[""][:]  # tif has no keys
            else:
                print(f.keys())
                for key in f.keys():
                    if isinstance(f[key], (zarr.Group, h5py.Group, z5py.Group)):
                        print(f"Loading group: {key}")
                        extract_data(f[key], data, scale=scale, prefix=key)
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

        if segment:
            # get foreground and boundary
            new_seg = _segment(data["pred/foreground"], data["pred/boundary"])
            data["new_seg"] = new_seg
        visualize_data(data, name=os.path.basename(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str)
    parser.add_argument("--ext", "-e", type=str, default=None)
    parser.add_argument("--scale", "-s", type=int, default=1)
    parser.add_argument("--upsample", "-u", type=int, default=None)
    parser.add_argument("--label_path", "-lp", type=str, default=None)
    parser.add_argument("--segment", "-seg", default=False, action="store_true")
    parser.add_argument("--offset_z", "-o", type=int, default=None, help="Offset in z direction")
    
    args = parser.parse_args()
    path = args.path
    ext = args.ext
    scale = args.scale
    upsample = args.upsample
    label_path = args.label_path
    main(path, ext, scale, upsample, label_path, segment=args.segment, offset_z=args.offset_z)
