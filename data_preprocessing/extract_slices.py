import argparse
import csv
import os
from glob import glob
from typing import Any, Dict
from tqdm import tqdm
import zarr
import z5py
import synapse.io.util as io
from tifffile import imread
import numpy as np
from collections import Counter, defaultdict
import h5py
from elf.io import open_file
import synapse.cellmap_util as cutil
import synapse.io.util as ioutil


def _extract_slices(data: Dict[str, Any], perc_slices: float = 0.01):
    """
    Extracts slices from a dataset along the z-axis.

    Parameters
    ----------
    data : Dict[str, Any]
        The input data, where each value is a 3D array.
    perc_slices : float, optional
        The percentage of slices to extract. Defaults to 0.01.

    Returns
    -------
    Dict[str, Any]
        A new dictionary where the values are the extracted slices.
    """
    slices = {}
    for key, item in data.items():
        # Calculate how many slices to take
        zdim = item.shape[0]
        num_slices = int(zdim * perc_slices)
        # Ensure at least 1 slice
        num_slices = max(num_slices, 1)

        # Center index
        center = zdim // 2
        # Compute a start and end such that we take 'num_slices' slices around the center
        start = max(center - num_slices // 2, 0)
        end = min(start + num_slices, zdim)

        slices[key] = item[start:end]
    return slices


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


def main(visualize=False):
    parser = argparse.ArgumentParser()
    # /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer
    parser.add_argument("--path", "-p",  type=str, default="/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/", help="Path to the root data directory")
    parser.add_argument("--ext", "-e", type=str, default=".h5")
    parser.add_argument("--path2", "-p2",  type=str, default="/mnt/lustre-grete/usr/u12103/cellmap/resized_crops_1percent/", help="Path to the root data directory")
    parser.add_argument("--ext2", "-e2", type=str, default=".h5")
    #parser.add_argument("--save_dir", type=str, default="", help="Path to save the data to")
    args = parser.parse_args()
    # print(args.base_path, "\n", args.base_path2)
    # /scratch-grete/projects/nim00007/data/mitochondria/cooper/20250308_Mito_Seg_Done/refined

    b1_paths = sorted(glob(os.path.join(args.path, "**", f"*{args.ext}"), recursive=True))#, reverse=True)
    #b2_paths = sorted(glob(os.path.join(args.path2, "**", f"*{args.ext2}"), recursive=True))#, reverse=True)
    os.makedirs(args.path2, exist_ok=True)

    for path in tqdm(b1_paths):
        filename = os.path.basename(path)
        out_path = os.path.join(args.path2, filename)
        if os.path.isfile(out_path):
            print(f"File already exists: {out_path}")
            continue
        # load data
        with open_file(path, mode="r") as f:
            data = {}
            for key in f.keys():
                if isinstance(f[key], (zarr.Group, h5py.Group, z5py.Group)):
                    extract_data(f[key], data)
                    continue
                # extract slices while preserving dimensions
                data[key] = f[key][:]
        print("all keys:", data.keys())
        # extract slices
        extracted_data = _extract_slices(data)

        print("Saving data to", out_path)
        ioutil.export_data(out_path, extracted_data)


if __name__ == "__main__":
    main()