import argparse
import os
from glob import glob
from tqdm import tqdm
import tifffile
import numpy as np
import h5py
import synapse.util as util


def process_mitos_and_cristae(raw, mitos, cristae):
    mito_ids = np.unique(mitos)
    mito_ids = mito_ids[mito_ids != 0]

    mitos_with_cristae = np.zeros_like(mitos, dtype=np.uint8)
    for mito_id in mito_ids:
        mito_mask = (mitos == mito_id)
        if np.any(cristae[mito_mask]):
            mitos_with_cristae[mito_mask] = 1
        else:
            mitos_with_cristae[mito_mask] = 2
            print("DEBUG - Found mito without cristae")

    combined = np.stack([raw, mitos_with_cristae], axis=0)
    return {"raw_mitos_combined": combined, "labels/cristae": cristae}


def find_sample_groups(base_path):
    raw_files = sorted(glob(os.path.join(base_path, "*_raw.tif")))
    groups = []
    for raw_path in raw_files:
        base = raw_path[:-len("_raw.tif")]
        mito_path = base + "_mitochondria.tif"
        cristae_path = base + "_cristae.tif"
        if os.path.exists(mito_path) and os.path.exists(cristae_path):
            groups.append((base, raw_path, mito_path, cristae_path))
        else:
            missing = []
            if not os.path.exists(mito_path):
                missing.append("mitochondria")
            if not os.path.exists(cristae_path):
                missing.append("cristae")
            print(f"Skipping {os.path.basename(base)}: missing {', '.join(missing)}")
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b", type=str,
                        default="/home/freckmann15/data/cristae/wichmann/2026-05-26_corrected",
                        help="Directory containing the tif files")
    parser.add_argument("--export_path", "-e", type=str,
                        default="/home/freckmann15/data/cristae/wichmann/2026-05-26_corrected_combined",
                        help="Directory for output h5 files")
    parser.add_argument("--voxel_size", "-v", type=float, nargs=3, default=None,
                        metavar=("Z", "Y", "X"),
                        help="Voxel size in nm (z y x). Optional.")
    args = parser.parse_args()

    util.create_directories_if_not_exists(args.export_path, "")
    voxel_size = np.array(args.voxel_size, dtype=np.float32) if args.voxel_size else None

    groups = find_sample_groups(args.base_path)
    print(f"Found {len(groups)} complete sample groups.")

    for base, raw_path, mito_path, cristae_path in tqdm(groups, desc="Processing files..."):
        sample_name = os.path.basename(base)
        output_path = os.path.join(args.export_path, sample_name + "_combined.h5")

        if os.path.exists(output_path):
            print(f"Output already exists, skipping: {output_path}")
            continue

        try:
            raw = tifffile.imread(raw_path).astype(np.float32)
            mitos = tifffile.imread(mito_path).astype(np.int32)
            cristae = tifffile.imread(cristae_path).astype(np.uint8)
        except Exception as e:
            print(f"Error reading {sample_name}: {e}")
            continue

        try:
            export_data = process_mitos_and_cristae(raw, mitos, cristae)
        except Exception as e:
            print(f"Error processing {sample_name}: {e}")
            continue

        util.export_data(output_path, export_data, voxel_size=voxel_size)


if __name__ == "__main__":
    main()
