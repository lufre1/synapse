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


def filter_annotations_shape(annotations, field="shape_type", shape_type="SegmentationMask"):
    assert shape_type in ("SegmentationMask", "OrientedPoint", "Point", "InstanceSegmentation", "Mesh")
    client = cdp.Client()
    new_annotations = cdp.AnnotationShape.find(client, [getattr(cdp.AnnotationShape, field) == shape_type])
    
    return annotations


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
        print("uri", uri)
        data, voxel_size = read_ome_zarr(uri,
                                         client_kwargs={'endpoint_url': 'https://s3.embl.de'}
                                         )
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


def main():
    parser = argparse.ArgumentParser()
    # Whether to check the result with napari instead of running the prediction.
    parser.add_argument("-c", "--check", action="store_true")
    parser.add_argument("-o", "--output_folder", default="/home/freckmann15/data/embl")
    parser.add_argument("-d", "--download", action="store_true", default=False)
    args = parser.parse_args()

    uris = [
        # "s3.embl.de/i2k-2020/experimental/mitos",
        "s3://i2k-2020/experimental/mitos/4007/images/ome-zarr/mitos.ome.zarr",
        # "https://s3.embl.de/i2k-2020/experimental/mitos/4007/images/ome-zarr"
    ]
    params = {}

    # Process each tomogram.
    for uri in tqdm(uris, desc="Downloading Data"):
        # Read tomogram data on the fly.
        if args.check:
            check_result(uri,
                         params,
                         download=args.download,
                         output_folder=args.output_folder
                         )


if __name__ == "__main__":
    main()
