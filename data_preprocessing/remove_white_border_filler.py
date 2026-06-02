"""Zero out large white (255) regions in HDF5 volume EM files.

Finds 3D connected components of voxels == 255, then sets any component
with >= min_size voxels to 0.  All other voxels are left untouched.

Usage
-----
python remove_white_border_filler.py -i /path/to/file_or_dir [-o /output/dir]
"""

import argparse
import os
import sys
from glob import glob

import h5py
import numpy as np
from scipy.ndimage import label as nd_label
from tqdm import tqdm

from synapse.h5_util import get_all_keys_from_h5


def zero_large_white_components(raw: np.ndarray, min_size: int = 10_000) -> tuple[np.ndarray, int]:
    mask = ((raw == 255) | (raw == 254)).astype(np.uint8)
    if not mask.any():
        return raw, 0

    labeled, _ = nd_label(mask)
    sizes = np.bincount(labeled.ravel())  # index 0 = background

    large_labels = np.where(sizes >= min_size)[0]
    large_labels = large_labels[large_labels != 0]

    if len(large_labels) == 0:
        return raw, 0

    result = raw.copy()
    result[np.isin(labeled, large_labels)] = 0
    return result, len(large_labels)


def process_file(input_path: str, output_path: str, min_size: int,
                 raw_key: str, dry_run: bool) -> bool:
    keys = get_all_keys_from_h5(input_path)

    with h5py.File(input_path, "r") as f:
        actual_raw_key = raw_key if raw_key in f else next(
            (k for k in keys if "raw" in k), None
        )
        if actual_raw_key is None:
            print("  WARNING: no raw dataset found, skipping.")
            return False
        raw = f[actual_raw_key][()].astype(np.uint8)

    cleaned, n_zeroed = zero_large_white_components(raw, min_size=min_size)

    if n_zeroed == 0:
        print("  no large white regions found")
        return False

    before = int(((raw == 255) | (raw == 254)).sum())
    after  = int(((cleaned == 255) | (cleaned == 254)).sum())
    n_voxels = before - after
    print(f"  zeroed {n_zeroed} component(s), {n_voxels:,} voxels")

    if dry_run:
        return True

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        for k, v in f_in.attrs.items():
            f_out.attrs[k] = v
        for key in keys:
            ds = f_in[key]
            data = cleaned if key == actual_raw_key else ds[()]
            out_ds = f_out.create_dataset(key, data=data, compression="gzip", compression_opts=4)
            for k, v in ds.attrs.items():
                out_ds.attrs[k] = v

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True,
                        help="Input file or directory (*.h5, recursive).")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file or directory. "
                             "Defaults to input location with --suffix appended.")
    parser.add_argument("--suffix", default="_nofill",
                        help="Filename suffix when --output is omitted. (default: _nofill)")
    parser.add_argument("--min_size", type=int, default=10_000,
                        help="Min voxels for a white component to be zeroed. (default: 10000)")
    parser.add_argument("--raw_key", default="raw",
                        help="HDF5 key of the raw volume. (default: raw)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print what would be done without writing files.")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        files = [args.input]
        base = os.path.dirname(args.input) or "."
    elif os.path.isdir(args.input):
        files = sorted(glob(os.path.join(args.input, "**", "*.h5"), recursive=True))
        base = args.input
    else:
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    if not files:
        print("No .h5 files found.", file=sys.stderr)
        sys.exit(1)

    n_modified = 0
    for path in tqdm(files, desc="Processing"):
        if args.output and os.path.splitext(args.output)[1]:
            out_path = args.output
        else:
            rel = os.path.relpath(path, base)
            if args.output:
                out_path = os.path.join(args.output, rel)
            else:
                stem, ext = os.path.splitext(path)
                out_path = stem + args.suffix + ext

        print(f"\n{path}")
        if not args.dry_run and os.path.exists(out_path):
            print(f"  skipping, output exists: {out_path}")
            continue

        modified = process_file(path, out_path, min_size=args.min_size,
                                raw_key=args.raw_key, dry_run=args.dry_run)
        if modified:
            n_modified += 1
            if not args.dry_run:
                print(f"  -> {out_path}")

    print(f"\nDone. {n_modified}/{len(files)} files modified.")


if __name__ == "__main__":
    main()
