import argparse
from glob import glob
import os
# import synapse.label_utils as lutils
# from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cryoet_data_portal as cdp
except ImportError:
    cdp = None

try:
    import zarr
except ImportError:
    zarr = None

try:
    import s3fs
except ImportError:
    s3fs = None

from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from synapse_net.file_utils import read_data_from_cryo_et_portal_run, read_ome_zarr
from tqdm import tqdm


def check_result(uri, params, download=True, output_folder=None):
    import napari

    data = {}
    segmentations = {}
    if output_folder is None:
        print("no ouptut folder given")
        return
    elif download:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        print("Downloading from URI:", uri)
        s3_uri = uri.replace("s3://", "")
        data, voxel_size = read_ome_zarr(s3_uri,
                                        #  client_kwargs={'endpoint_url': 'https://s3.embl.de'},
                                         )
        # Save the data to the output folder
        output_file = os.path.join(output_folder, os.path.basename(s3_uri)) + ".zarr"
        write_ome_zarr(output_file, data, voxel_size)
        print("Saved to", output_file)
    else:
        print("no in-memory feature implemented yet")
        return
    v = napari.Viewer()
    if data is not None:
        v.add_image(data)
    if segmentations is not None:
        for key, val in segmentations.items():
            if np.issubdtype(val.dtype, np.floating):  # Check if array contains floats
                v.add_image(val, name=key)
            else:
                v.add_labels(val, name=key)
    napari.run()


def write_ome_zarr(output_file, data, voxel_size):
    store = parse_url(output_file, mode="w").store
    root = zarr.group(store=store)

    scale = list(voxel_size.values())
    trafo = [
        [{"scale": scale, "type": "scale"}]
    ]
    write_image(data, root, axes="zyx", coordinate_transformations=trafo, scaler=None)
    print("Wrote", output_file)


def download_ome_zarr_data(uri: str, out_dir: str, scale_level: int = 0):
    """Download data from an OME-Zarr store and save locally.

    Args:
        uri: URL or path to OME-Zarr store.
        out_dir: Output directory to save data and metadata.
        scale_level: Multiscale level to download.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Read data and voxel size using your function
    data, voxel_size = read_ome_zarr(uri, scale_level=scale_level)

    # Save data
    out_file = os.path.join(out_dir, ".zarr")
    write_ome_zarr(out_file, data, voxel_size=voxel_size)


def main():
    parser = argparse.ArgumentParser()
    # Whether to check the result with napari instead of running the prediction.
    parser.add_argument("-c", "--check", action="store_true")
    parser.add_argument("-o", "--output_folder", default="/home/freckmann15/data/embl")
    parser.add_argument("-d", "--download", action="store_true", default=False)
    args = parser.parse_args()

    uris = [
        # "s3.embl.de/i2k-2020/experimental/mitos",
        "https://s3.embl.de/i2k-2020/experimental/mitos/4007/images/ome-zarr/mitos.ome.zarr"

        # "s3://i2k-2020/experimental/mitos/4007/images/ome-zarr/mitos.ome.zarr",
        # "https://s3.embl.de/i2k-2020/experimental/mitos/4007/images/ome-zarr"
    ]
    params = {}

    # Process each tomogram.
    for uri in tqdm(uris, desc="Downloading Data"):
        # Read tomogram data on the fly.
        try:
            data, voxel_size = read_ome_zarr(uri, scale_level="s0")
            print(f"Successfully read data from {uri} without downloading.")
            print("Data shape:", data.shape)
            print("Voxel size:", voxel_size)
            # Add your further processing of the 'data' here
        except ValueError as e:
            print(f"Error reading data from {uri}: {e}")
        # if args.check:
        #     check_result(uri,
        #                  params,
        #                  download=args.download,
        #                  output_folder=args.output_folder
        #                  )
        # else:
        #     download_ome_zarr_data(uri, args.output_folder, scale_level="s0")


if __name__ == "__main__":
    main()
