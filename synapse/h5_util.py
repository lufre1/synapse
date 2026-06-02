from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import h5py
import numpy as np
from synapse_net.file_utils import read_ome_zarr
from elf.io import open_file
import tifffile
import zarr
import z5py


def read_data(path, scale=1):
    data = {}
    if ".tif" in path:
        img = tifffile.imread(path)
        ndim = img.ndim
        slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
        data["label"] = img[slicing] if scale > 1 else img
    elif (".mrc" in path or ".rec" in path):
        with open_file(path, "r") as f:
            ndim = f["data"].ndim
            slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
            data["raw"] = f["data"][slicing] if scale > 1 else f["data"][:]
    elif (".h5" in path or ".n5" in path):
        with open_file(path, "r") as f:
            # all_ds = get_all_datasets(path)
            for key in f.keys():
                if isinstance(f[key], (zarr.Group, h5py.Group, z5py.Group)):
                    extract_data(f[key], data, scale=scale, prefix=key)
                else:
                    ndim = f[key].ndim
                    slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
                    data[key] = f[key][slicing] if scale > 1 else f[key][:]

    elif (".zarr" in path):
        img, voxel_size = read_ome_zarr(path)
        ndim = img.ndim
        slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
        data["raw"] = img[slicing] if scale > 1 else img

    return data


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


def _parse_axes_order(axes_attr) -> str:
    """Return a comma-separated lowercase axes string, e.g. 'z,y,x'."""
    if axes_attr is None:
        return None
    if isinstance(axes_attr, np.ndarray):
        return ",".join(str(a).lower() for a in axes_attr)
    if isinstance(axes_attr, bytes):
        axes_attr = axes_attr.decode("utf-8")
    return str(axes_attr).replace(" ", "").lower()


def read_voxel_size(
    h5_path: Union[str, Path],
    h5_key: str = "raw",
    *,
    default: Optional[Tuple[float, float, float]] = None,
) -> Optional[Tuple[float, float, float]]:
    """
    Read the ``voxel_size`` attribute and return it as (z, y, x).

    Checks attributes in this priority order:
      1. The dataset/group at ``h5_key``
      2. The root of the HDF5 file (common in files written by tools like IMOD)

    The ``axes`` attribute (array of axis names, e.g. ``['z','y','x']``) or
    ``voxel_size_order`` string is used to determine the storage order so the
    result is always returned as **(z, y, x)**.

    Parameters
    ----------
    h5_path : str or pathlib.Path
        Path to the HDF5 file.
    h5_key : str
        Internal HDF5 path to the object that holds the attribute.
    default : tuple of three floats, optional
        Value to return if the attribute is missing everywhere. If ``None``
        (default), returns None instead of raising an error.

    Returns
    -------
    voxel_size : tuple (z, y, x) or None
    """
    h5_path = Path(h5_path).expanduser().resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        # ------------------------------------------------------------------
        # 1. Find the object that actually carries voxel_size
        # ------------------------------------------------------------------
        raw = None
        axes_attr = None

        # Check the requested dataset/group first
        if h5_key in f and "voxel_size" in f[h5_key].attrs:
            obj = f[h5_key]
            raw = obj.attrs["voxel_size"]
            axes_attr = obj.attrs.get("axes", obj.attrs.get("voxel_size_order", None))

        # Fall back to root-level attributes (e.g. files written by IMOD/tomo tools)
        elif "voxel_size" in f.attrs:
            raw = f.attrs["voxel_size"]
            axes_attr = f.attrs.get("axes", f.attrs.get("voxel_size_order", None))

        if raw is None:
            return tuple(default) if default is not None else None

        # ------------------------------------------------------------------
        # 2. Parse the voxel_size value
        # ------------------------------------------------------------------
        if isinstance(raw, np.ndarray) and raw.dtype.names:
            # Structured array with named fields z/y/x
            return (float(raw["z"]), float(raw["y"]), float(raw["x"]))

        try:
            seq = tuple(float(v) for v in (raw.flatten() if isinstance(raw, np.ndarray) else raw))
            if len(seq) != 3:
                raise ValueError
        except Exception as exc:
            raise TypeError(
                f"Unable to interpret 'voxel_size' as a length-3 float sequence "
                f"in '{h5_path}' (key: '{h5_key}')."
            ) from exc

        # ------------------------------------------------------------------
        # 3. Re-order to (z, y, x)
        # ------------------------------------------------------------------
        order = _parse_axes_order(axes_attr)
        if order == "z,y,x":
            return (seq[0], seq[1], seq[2])
        elif order == "x,y,z":
            return (seq[2], seq[1], seq[0])
        else:
            # Legacy fallback: check old voxel_size_order string
            if axes_attr is not None:
                order_str = _parse_axes_order(axes_attr)
                if order_str and order_str.startswith("z"):
                    return (seq[0], seq[1], seq[2])
            # Default assumption: XYZ → reverse to ZYX
            return (seq[2], seq[1], seq[0])


def read_h5(path, key, scale_factor=1, z_offset=None):
    with h5py.File(path, "r") as f:
        try:
            print(f"{key} data shape", f[key].shape)
            if key == ("prediction" or "pred" in key) and not ("foreground" in key or "boundar" in key) :
                image = f[key][:, ::scale_factor, ::scale_factor, ::scale_factor]
                if z_offset:
                    image = image[z_offset[0]:z_offset[1], :, :]
            else:
                image = f[key][::scale_factor, ::scale_factor, ::scale_factor]
                if z_offset:
                    image = image[z_offset[0]:z_offset[1], :, :]
            if scale_factor != 1:
                print(f"{key} data shape after downsampling during read operation", image.shape)
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
