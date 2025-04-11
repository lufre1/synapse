import os
import s3fs
import zarr
from tqdm import tqdm
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
import numpy as np
from typing import Dict

def is_ome_zarr_store(s3, path: str) -> bool:
    """Heuristically checks if a given S3 path is likely an OME-Zarr store."""
    # Check for the presence of a .zgroup file at the root or multiscales directory
    if s3.exists(f"{path}/.zgroup") or s3.exists(f"{path}/0/.zarray"):
        return True
    # Check for the presence of the multiscales directory and some level within it
    elif s3.exists(f"{path}/multiscales/0/.zarray"):
        return True
    elif s3.exists(f"{path}/images/0/.zarray"): # Common for some OME-Zarr structures
        return True
    return False

def download_ome_zarr_store(s3: s3fs.S3FileSystem, s3_path: str, local_dir: str):
    """Downloads an entire OME-Zarr store from S3 to a local directory."""
    local_path = os.path.join(local_dir, s3_path.replace("s3://i2k-2020/", "").replace("/", "_"))
    os.makedirs(local_path, exist_ok=True)
    print(f"Downloading {s3_path} to {local_path}...")
    for s3_file in tqdm(s3.rglob(s3_path), desc=f"Downloading {os.path.basename(s3_path)}"):
        if not s3_file.endswith("/"):  # Skip directories
            relative_path = os.path.relpath(s3_file, s3_path)
            local_file_path = os.path.join(local_path, relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            try:
                s3.get(s3_file, local_file_path)
            except Exception as e:
                print(f"Error downloading {s3_file}: {e}")
    print(f"Finished downloading {s3_path} to {local_path}")

def main():
    s3_prefix = "i2k-2020/experimental/mitos/"
    local_output_dir = "/home/freckmann15/data/embl/mitos_downloaded"  # Adjust this to your desired local directory

    os.makedirs(local_output_dir, exist_ok=True)

    s3 = s3fs.S3FileSystem(anon=True)

    print(f"Exploring OME-Zarr stores in s3://{s3_prefix}")
    potential_zarr_stores = [f"s3://{path}" for path in s3.ls(s3_prefix) if "." not in os.path.basename(path) and path.startswith(s3_prefix)]

    identified_zarr_stores = []
    for store_prefix in tqdm(potential_zarr_stores, desc="Identifying OME-Zarr stores"):
        if is_ome_zarr_store(s3, store_prefix):
            identified_zarr_stores.append(store_prefix)
        else:
            # Check one level deeper in case the OME-Zarr is within a subdirectory
            for deeper_path in s3.ls(store_prefix):
                full_deeper_path = f"s3://{deeper_path}"
                if is_ome_zarr_store(s3, full_deeper_path):
                    identified_zarr_stores.append(full_deeper_path)
                    break # Assuming one OME-Zarr per subdirectory

    if identified_zarr_stores:
        print("\nIdentified OME-Zarr stores:")
        for store in identified_zarr_stores:
            print(f"- {store}")

        print("\nStarting download of identified OME-Zarr stores...")
        for store_path in identified_zarr_stores:
            download_ome_zarr_store(s3, store_path, local_output_dir)
        print("\nFinished processing.")
    else:
        print("No OME-Zarr stores found in the specified S3 prefix.")

if __name__ == "__main__":
    main()