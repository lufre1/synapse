import argparse
import os
import s3fs
import zarr
from tqdm import tqdm
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from synapse_net.file_utils import read_ome_zarr
import numpy as np
from typing import Dict


def write_ome_zarr(
    output_file,
    data,
    voxel_size,
    axes="zyx",
    compressor=None,
    target_dtype=None,
    chunk_size=(64, 64, 64),
    normalize=False
):
    """
    Write data to OME-Zarr with compression and optional dtype conversion.

    Args:
        output_file (str): Path to the output .zarr store.
        data (ndarray): The data to store.
        voxel_size (dict): Dict with keys ("z", "y", "x") → voxel sizes.
        axes (str): Axes string (e.g. "zyx", "czyx").
        compressor: Zarr compressor, e.g. GZip(level=4) or Blosc().
        target_dtype: Target NumPy dtype (e.g. np.uint8, np.int16).
        chunk_size: Chunk shape, matched to data dimensions.
        normalize (bool): If True, normalize float data to [0, 255] and convert to uint8.
    """
    if target_dtype is not None:
        if normalize and np.issubdtype(data.dtype, np.floating):
            data = np.clip(data, a_min=0, a_max=1)  # or use data.min(), data.max() for dynamic range
            data = (data * 255).astype(target_dtype)
        else:
            data = data.astype(target_dtype)

    # Ensure chunk size has correct number of dims
    chunk_size = tuple(min(c, s) for c, s in zip(chunk_size, data.shape[-len(chunk_size):]))

    store = parse_url(output_file, mode="w").store
    root = zarr.group(store=store)

    scale = [voxel_size[k] for k in axes[-3:]]  # Handle optional leading dims (e.g. 'c')
    trafo = [[{"scale": scale, "type": "scale"}]]

    write_image(
        data,
        root,
        axes=axes,
        coordinate_transformations=trafo,
        scaler=None,
        storage_options={"chunks": chunk_size, "compressor": compressor},
    )
    print("Wrote", output_file)

# # Luca's approach 
# def write_ome_zarr(output_file, data, voxel_size):
#     store = parse_url(output_file, mode="w").store
#     root = zarr.group(store=store)

#     scale = list(voxel_size.values())
#     trafo = [
#         [{"scale": scale, "type": "scale"}]
#     ]
#     write_image(data, root, axes="zyx", coordinate_transformations=trafo, scaler=None)
#     print("Wrote", output_file)


def list_s3_bucket(base_path="i2k-2020/", max_depth=2, include_dirs=True, fs=None):
    """
    Explore the EMBL public S3 bucket and return a list of paths.

    Args:
        base_path: Path inside the bucket to explore.
        max_depth: How many directory levels to recurse.
        include_dirs: Whether to include directory paths in the output list.

    Returns:
        A list of S3 paths (files and optionally directories).
    """
    if fs is None:
        fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": "https://s3.embl.de"})
    paths = []

    def list_recursive(path, depth=0):
        if depth > max_depth:
            return
        try:
            items = fs.ls(path, detail=True)
            for item in items:
                name = item["Key"] if "Key" in item else item["name"]
                # if item["type"] == "file" or include_dirs:
                if name.endswith(".zarr"):
                    paths.append(name)
                if item["type"] == "directory":
                    list_recursive(name, depth + 1)
        except Exception as e:
            print("Error listing:", path, e)

    list_recursive(base_path)
    return paths


def main():
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--path", "-p", type=str, default="i2k-2020/experimental/mitos")
    argsparse.add_argument("--resolution", "-r", type=int, default=0)
    argsparse.add_argument("--output_path", "-o", type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/embl")
    argsparse.add_argument("--download", "-d", action="store_true", default=False)
    args = argsparse.parse_args()
    
    s3_prefix = "https://s3.embl.de"
    local_output_dir = args.output_path

    os.makedirs(local_output_dir, exist_ok=True)

    print(f"Exploring OME-Zarr stores in s3 bucket: {s3_prefix}")
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": s3_prefix})
    paths = list_s3_bucket(base_path=args.path, max_depth=5)
    # print(paths)
    # filter for .zarr
    zarr_paths = [path for path in paths if ".zarr" in path and path.endswith(".zarr")]
    for path in zarr_paths:
        print("\nReading", path)
        print(fs.ls(path))
        
        if args.download:
            out_path = os.path.join(local_output_dir, path.replace("i2k-2020/experimental/mitos", "").lstrip("/"))
            if not os.path.exists(os.path.dirname(out_path)):
                print("Full path", out_path)
                dir_name = os.path.dirname(out_path)
                print("Creating directory", dir_name)
                os.makedirs(dir_name, exist_ok=True)
            if os.path.exists(out_path):
                print("Skipping", out_path)
                continue
            data, voxel_size = read_ome_zarr(path, args.resolution, fs=fs)
            
            print("Writing", out_path)
            print("Voxel size", voxel_size)
            write_ome_zarr(out_path, data, voxel_size)
            print("Finished writing", out_path)


if __name__ == "__main__":
    main()