import argparse
from glob import glob
import os
from tqdm import tqdm
import numpy as np

import synapse.util as util


def parse_shape(s):
    # expects e.g. "64,256,256" -> (64, 256, 256)
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be 'z,y,x'")
    return tuple(int(p) for p in parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_path", "-b",
        type=str,
        default="/home/freckmann15/data/mitochondria/cooper/fidi_2025/exported_to_hdf5/m13dko/37371_O4_66K_TS_SC_50_rec_2Kb1dawbp_crop_F.h5",
        help="Path to the root data directory",
    )
    parser.add_argument(
        "--export_path", "-e",
        type=str,
        default="/home/freckmann15/data/mitochondria/cooper/fidi_2025/exported_to_hdf5/m13dko/",
        help="Path to the export directory",
    )
    parser.add_argument(
        "--crop_shape", "-c",
        type=parse_shape,
        required=True,
        help="Crop shape as 'z,y,x'",
    )
    parser.add_argument(
        "--stride", "-t",
        type=parse_shape,
        help="Stride as 'z,y,x' (default: same as crop_shape)",
    )
    args = parser.parse_args()

    base_path = args.base_path
    export_path = args.export_path
    crop_shape = args.crop_shape
    stride = args.stride or crop_shape  # non-overlapping by default

    if os.path.isdir(base_path):
        paths = sorted(glob(os.path.join(base_path, "**", "*.h5"), recursive=True))
    elif os.path.isfile(base_path):
        paths = [base_path]
    else:
        raise ValueError(f"Invalid base path: {base_path}")

    os.makedirs(export_path, exist_ok=True)

    for path in tqdm(paths, desc="Processing files..."):
        data = util.read_data(path)  # assume dict of numpy arrays / h5py-like datasets [web:8][web:14]
        voxel_size = util.read_voxel_size_h5(path)

        # assume all arrays share the same spatial shape
        first_key = next(iter(data.keys()))
        vol_shape = data[first_key].shape
        if len(vol_shape) != 3:
            raise ValueError(f"Expected 3D data, got shape {vol_shape} for key {first_key}")

        z_max, y_max, x_max = vol_shape
        cz, cy, cx = crop_shape
        sz, sy, sx = stride

        basename = os.path.splitext(os.path.basename(path))[0]

        crop_idx = 0
        for z in range(0, z_max - cz + 1, sz):
            for y in range(0, y_max - cy + 1, sy):
                for x in range(0, x_max - cx + 1, sx):
                    crop = {}
                    z1, y1, x1 = z + cz, y + cy, x + cx
                    for key, arr in data.items():
                        crop[key] = arr[z:z1, y:y1, x:x1]

                    out_name = f"{basename}_z{z}-{z1}_y{y}-{y1}_x{x}-{x1}_{crop_idx}.h5"
                    output_path = os.path.join(export_path, out_name)
                    util.export_data(output_path, crop, voxel_size=voxel_size)
                    crop_idx += 1


if __name__ == "__main__":
    main()
