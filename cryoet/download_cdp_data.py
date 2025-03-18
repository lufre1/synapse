import argparse
from glob import glob
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import zarr
import pooch

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


def get_tomograms(deposition_id, processing_type=None):
    client = cdp.Client()
    if processing_type is not None:
        tomograms = cdp.Tomogram.find(
            client, [cdp.Tomogram.deposition_id == deposition_id, cdp.Tomogram.processing == processing_type]
        )
    else:
        tomograms = cdp.Tomogram.find(
            client, [cdp.Tomogram.deposition_id == deposition_id]
        )

    return tomograms


def get_annotations(id, id_field, search_string=None):
    assert id_field in ("id", "run_id", "deposition_id")
    client = cdp.Client()
    if search_string is None:
        annotations = cdp.Annotation.find(client, [getattr(cdp.Annotation, id_field) == id])
    else:
        annotations = cdp.Annotation.find(client, [getattr(cdp.Annotation, id_field) == id,
                                                   cdp.Annotation.object_name.ilike(f"%{search_string}%")])
    return annotations


def check_result(tomogram, deposition_id, processing_type, download=False):
    import napari
    annotations = get_annotations(tomogram.run_id, id_field="run_id", search_string="mito")
    # check for run_id in annotation and download

    # Read the output file if it exists.
    output_folder = os.path.join(
        "/home/freckmann15/data/cryo-et",
        f"upload_CZCDP-{deposition_id}",
        str(tomogram.run_id)
        )
    output_file = os.path.join(output_folder, f"{tomogram.run.name}.zarr")
    if os.path.exists(output_folder):
        print("Try reading segmenations from:", output_folder)
        present = True
    else:
        present = False
    if download or not present:
        for annotation in tqdm(annotations, desc="Downloading annotations"):
            annotation.download(dest_path=output_folder, format="zarr")

    # get all segmentations for corresponding tomogram (run_id)
    segmentation_paths = glob(os.path.join(output_folder, "**", "*.zarr"), recursive=True)
    # filter for only segmentations via mask
    segmentation_paths[:] = [path for path in segmentation_paths if "mask" in path.lower()]
    segmentations = {}
    for segmentation_path in segmentation_paths:
        labels, _voxel_size = read_ome_zarr(segmentation_path)
        segmentations[segmentation_path] = np.flip(labels, axis=1) if labels.ndim == 3 else np.flip(labels, axis=0)
    
    if not os.path.exists(output_file) and download:
        print("Downloading tomogram data and saving to", output_file)
        # Read tomogram data on the fly.
        data, voxel_size = read_data_from_cryo_et_portal_run(
            tomogram.run_id, id_field="run_id", processing_type=processing_type
        )
        write_ome_zarr(output_file, data, voxel_size)
    elif os.path.exists(output_file):
        print("Reading tomogram data from", output_file)
        data, voxel_size = read_ome_zarr(output_file)
    else:
        print("Streaming tomogram data")
        # Read tomogram data on the fly.
        data, voxel_size = read_data_from_cryo_et_portal_run(
            tomogram.run_id, id_field="run_id", processing_type=processing_type
        )

    v = napari.Viewer()
    # print("data", data)
    # print("voxel_size", voxel_size)
    # print("segmentation", segmentation)
    v.add_image(data)
    if segmentations is not None:
        for key, segmentation in segmentations.items():
            v.add_labels(segmentation, name=key)
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


def main():
    parser = argparse.ArgumentParser()
    # Whether to check the result with napari instead of running the prediction.
    parser.add_argument("-c", "--check", action="store_true")
    parser.add_argument("-o", "--output_folder", default="out")
    args = parser.parse_args()

    # deposition with mitos annotated 10010
    deposition_id = 10010  # 10313
    processing_type = None  # "denoised" 

    # Get all the (processed) tomogram ids in the deposition.
    tomograms = get_tomograms(deposition_id, processing_type)
    #annotations = get_annotations(deposition_id, search_string="mito")

    # Process each tomogram.
    for tomogram in tqdm(tomograms, desc="Downloading tomograms"):
        # Read tomogram data on the fly.
        if args.check:
            check_result(tomogram, deposition_id, processing_type)
            # data, voxel_size = read_data_from_cryo_et_portal_run(
            #     tomogram.run_id, processing_type=processing_type
            # )
        else:
            data, voxel_size = read_data_from_cryo_et_portal_run(
                tomogram.run_id, processing_type=processing_type
            )
            # Write the data to a zarr file.
            output_path = os.path.join(args.output_folder, f"upload_CZCDP-{deposition_id}", str(tomogram.run.dataset_id))
            write_ome_zarr(os.path.join(output_path, f"{tomogram.run.name}.zarr"), data, voxel_size)


if __name__ == "__main__":
    main()