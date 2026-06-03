import os

import numpy as np
from elf.io import open_file
from skimage.measure import label as skimage_label
from tqdm import tqdm

from synapse_net.inference.cristae import segment_cristae as _segment_cristae

import synapse.util as util
from synapse.io.util import get_all_keys_from_h5
from synapse.cristae.label_utils import binarize_and_erode_xy, find_additional_objects


def run_cristae_segmentation(
    h5_paths,
    model_path,
    export_path,
    tile_shape=(32, 512, 512),
    erode_mitos=False,
    add_missing=False,
    base_path=None,
):
    """Run cristae segmentation on a list of multi-channel H5 files.

    Each file must contain a `raw_mitos_combined` dataset with shape [2, Z, Y, X]:
    channel 0 = raw EM, channel 1 = mitochondria state mask.

    Args:
        h5_paths: List of H5 file paths to process.
        model_path: Path to the cristae model checkpoint.
        export_path: Root directory for output files.
        tile_shape: (z, y, x) tile size for tiled prediction.
        erode_mitos: Erode the mito mask in XY before using as extra_segmentation.
        add_missing: If True, merge model-found objects back into existing GT labels.
        base_path: If given, preserve relative directory structure in the output.
    """
    os.makedirs(export_path, exist_ok=True)
    z, y, x = tile_shape
    ts = {"z": z, "y": y, "x": x}
    halo = {"z": int(z * 0.25), "y": int(y * 0.25), "x": int(x * 0.25)}
    tiling = {"tile": ts, "halo": halo}

    for path in tqdm(h5_paths):
        print("opening file", path)

        if base_path is not None:
            filename, inter_dirs = util.get_filename_and_inter_dirs(path, base_path)
            output_path = os.path.join(export_path, inter_dirs, filename + ".h5")
            util.create_directories_if_not_exists(export_path, inter_dirs)
        else:
            stem = os.path.splitext(os.path.basename(path))[0]
            output_path = os.path.join(export_path, stem + ".h5")

        if os.path.exists(output_path):
            print("Skipping... output path exists", output_path)
            continue

        keys = get_all_keys_from_h5(path)
        with open_file(path, "r") as f:
            data = {key: f[key][:] for key in keys}

        image = data["raw_mitos_combined"]  # [2, Z, Y, X]
        mito = image[1]
        mito_proc = binarize_and_erode_xy(mito, radius_xy=5) if erode_mitos else (mito > 0)

        seg, pred = _segment_cristae(
            image[0], model_path,
            scale=None,
            tiling=tiling,
            return_predictions=True,
            extra_segmentation=mito_proc,
            channels_to_standardize=[0],
            with_channels=True,
        )

        if add_missing:
            with open_file(output_path, "w") as f:
                for key in keys:
                    f[key] = data[key]
                    if "cristae" in key:
                        additional = find_additional_objects(data[key], seg, matching_threshold=0.1)
                        f[key] = skimage_label(data[key] + additional)
                f["pred"] = pred
                f["labels/new_cristae_seg"] = seg
        else:
            out = {k: v for k, v in data.items() if k != "raw_mitos_combined"}
            out["raw"] = image[0]
            out["labels/mitochondria"] = image[1]
            out["pred/foreground"] = pred[0]
            out["pred/boundary"] = pred[1]
            out["seg"] = seg
            util.export_data(output_path, out)

        print("Saved to", output_path)
