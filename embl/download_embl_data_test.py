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

# def is_ome_zarr_store(s3, path: str) -> bool:
#     """Heuristically checks if a given S3 path is likely an OME-Zarr store."""
#     # Check for the presence of a .zgroup file at the root or multiscales directory
#     if s3.exists(f"{path}/.zgroup") or s3.exists(f"{path}/0/.zarray"):
#         return True
#     # Check for the presence of the multiscales directory and some level within it
#     elif s3.exists(f"{path}/multiscales/0/.zarray"):
#         return True
#     elif s3.exists(f"{path}/images/0/.zarray"): # Common for some OME-Zarr structures
#         return True
#     return False

# def download_ome_zarr_store(s3: s3fs.S3FileSystem, s3_path: str, local_dir: str):
#     """Downloads an entire OME-Zarr store from S3 to a local directory."""
#     local_path = os.path.join(local_dir, s3_path.replace("s3://i2k-2020/", "").replace("/", "_"))
#     os.makedirs(local_path, exist_ok=True)
#     print(f"Downloading {s3_path} to {local_path}...")
#     for s3_file in tqdm(s3.rglob(s3_path), desc=f"Downloading {os.path.basename(s3_path)}"):
#         if not s3_file.endswith("/"):  # Skip directories
#             relative_path = os.path.relpath(s3_file, s3_path)
#             local_file_path = os.path.join(local_path, relative_path)
#             os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
#             try:
#                 s3.get(s3_file, local_file_path)
#             except Exception as e:
#                 print(f"Error downloading {s3_file}: {e}")
#     print(f"Finished downloading {s3_path} to {local_path}")


def write_ome_zarr(output_file, data, voxel_size):
    store = parse_url(output_file, mode="w").store
    root = zarr.group(store=store)

    scale = list(voxel_size.values())
    trafo = [
        [{"scale": scale, "type": "scale"}]
    ]
    write_image(data, root, axes="zyx", coordinate_transformations=trafo, scaler=None)
    print("Wrote", output_file)


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
    argsparse.add_argument("--resolution", "-r", type=int, default=2)
    argsparse.add_argument("--output_path", "-o", type=str, default="/home/freckmann15/data/embl")
    argsparse.add_argument("--download", "-d", action="store_true", default=False)
    args = argsparse.parse_args()
    
    s3_prefix = "https://s3.embl.de"
    local_output_dir = args.output_path

    os.makedirs(local_output_dir, exist_ok=True)

    # s3 = s3fs.S3FileSystem(anon=True)

    print(f"Exploring OME-Zarr stores in s3 bucket: {s3_prefix}")
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": "https://s3.embl.de"})
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
            data, voxel_size = read_ome_zarr(path, args.resolution, fs=fs, client_kwargs={"endpoint_url": s3_prefix})
            
            print("Writing", out_path)
            write_ome_zarr(out_path, data, voxel_size)
            print("Finished writing", out_path)

            
        
    
    
    # potential_zarr_stores = [f"s3://{path}" for path in s3.ls(s3_prefix) if "." not in os.path.basename(path) and path.startswith(s3_prefix)]

    # identified_zarr_stores = []
    # for store_prefix in tqdm(potential_zarr_stores, desc="Identifying OME-Zarr stores"):
    #     if is_ome_zarr_store(s3, store_prefix):
    #         identified_zarr_stores.append(store_prefix)
    #     else:
    #         # Check one level deeper in case the OME-Zarr is within a subdirectory
    #         for deeper_path in s3.ls(store_prefix):
    #             full_deeper_path = f"s3://{deeper_path}"
    #             if is_ome_zarr_store(s3, full_deeper_path):
    #                 identified_zarr_stores.append(full_deeper_path)
    #                 break # Assuming one OME-Zarr per subdirectory

    # if identified_zarr_stores:
    #     print("\nIdentified OME-Zarr stores:")
    #     for store in identified_zarr_stores:
    #         print(f"- {store}")

    #     print("\nStarting download of identified OME-Zarr stores...")
    #     for store_path in identified_zarr_stores:
    #         download_ome_zarr_store(s3, store_path, local_output_dir)
    #     print("\nFinished processing.")
    # else:
    #     print("No OME-Zarr stores found in the specified S3 prefix.")


if __name__ == "__main__":
    main()