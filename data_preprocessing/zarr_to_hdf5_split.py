#!/usr/bin/env python3
"""
Split a 3‑D Zarr volume into many HDF5 files.

Each output file contains a sub‑volume whose number of voxels never exceeds
`max_voxels` (default = 500 * 1000 * 1000 ≈ 5 × 10⁸ voxels ≈ 200 × 1600 × 1600).

A *minimum* crop size can be enforced with `--min_shape`.  The default is
(128, 1024, 1024) – the same numbers you asked for.
"""

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import tqdm
import zarr
import z5py


# functions
# ----------------------------------------------------------------------
def _open_zarr(zarr_path: Path):
    """Open a Zarr or N5 store and return the root group."""
    if (zarr_path / "attributes.json").exists():
        store = zarr.N5Store(str(zarr_path))
    else:
        store = zarr.DirectoryStore(str(zarr_path))
    return zarr.open(store, mode="r")


def _calc_block_shape(volume_shape, max_voxels, min_shape):
    """
    Return a (z, y, x) block size that

    * fits into the voxel budget (`max_voxels`);
    * is **at least** `min_shape` in every dimension;
    * never exceeds the volume size.
    """
    z, y, x = volume_shape
    mz, my, mx = min_shape

    # start with the full XY size, shrink Z until we are under the budget
    block_z = min(z, max_voxels // (y * x))
    if block_z == 0:                     # XY alone already exceeds the budget
        block_z = 1
        block_y = min(y, int(math.sqrt(max_voxels / block_z)))
        block_x = min(x, max_voxels // (block_z * block_y))
    else:
        block_y, block_x = y, x

    # enforce the *minimum* shape
    block_z = max(block_z, mz)
    block_y = max(block_y, my)
    block_x = max(block_x, mx)

    # make sure we never exceed the budget (halve the largest dim if needed)
    while block_z * block_y * block_x > max_voxels:
        # reduce the largest dimension first
        if block_x >= block_y and block_x >= block_z:
            block_x = max(mz, block_x // 2)
        elif block_y >= block_x and block_y >= block_z:
            block_y = max(my, block_y // 2)
        else:
            block_z = max(mz, block_z // 2)

    # finally clamp to the actual volume size
    block_z = min(block_z, z)
    block_y = min(block_y, y)
    block_x = min(block_x, x)

    return block_z, block_y, block_x


def _write_block(h5_path: Path, block_data: np.ndarray, h5_key: str, block_idx: int):
    """Write a single block to an HDF5 file."""
    with h5py.File(h5_path, "w") as f:
        f.create_dataset(
            h5_key,
            data=block_data,
            compression="gzip",
            chunks=True,
        )
        f.attrs["block_index"] = block_idx   # optional meta‑data


# ----------------------------------------------------------------------
# Core splitter
# ----------------------------------------------------------------------
def split_zarr_to_h5(zarr_path: Path,
                     output_dir: Path,
                     max_voxels: int,
                     zarr_key: str,
                     h5_key: str,
                     min_shape: tuple):
    """Read a Zarr dataset and write it into many HDF5 files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Open Zarr and get the requested dataset
    zarr_root = _open_zarr(zarr_path)
    if zarr_key not in zarr_root:
        sys.exit(f"❌  Key '{zarr_key}' not found in {zarr_path}")
    arr = zarr_root[zarr_key]          # Zarr array – lazy, not loaded yet
    volume_shape = arr.shape            # (z, y, x)
    print(f"Volume shape (z, y, x): {volume_shape}")

    # 2️⃣ Determine a block shape that respects both the voxel budget
    #    and the minimum‑crop requirement.
    block_shape = _calc_block_shape(volume_shape, max_voxels, min_shape)
    bz, by, bx = block_shape
    print(f"Block shape (z, y, x): {block_shape} → {bz*by*bx:,} voxels per block")

    # 3️⃣ Loop over the grid and write each block.
    #    The inner loops are written so that the *last* block in each
    #    dimension is extended if the remaining slice would be smaller
    #    than the minimum shape.
    nz, ny, nx = volume_shape
    block_id = 0
    for z0 in tqdm.tqdm(range(0, nz, bz), desc="Z blocks"):
        # ensure the last Z block is not smaller than min_shape[0]
        z1 = min(z0 + bz, nz)
        if nz - z0 < min_shape[0] and z0 != 0:
            # extend the previous block to cover the tail
            z0 = max(0, nz - min_shape[0])
            z1 = nz

        for y0 in range(0, ny, by):
            y1 = min(y0 + by, ny)
            if ny - y0 < min_shape[1] and y0 != 0:
                y0 = max(0, ny - min_shape[1])
                y1 = ny

            for x0 in range(0, nx, bx):
                x1 = min(x0 + bx, nx)
                if nx - x0 < min_shape[2] and x0 != 0:
                    x0 = max(0, nx - min_shape[2])
                    x1 = nx

                # read the sub‑volume (lazy view)
                block = arr[z0:z1, y0:y1, x0:x1]
                block_np = np.asarray(block)          # materialise a small NumPy array

                # deterministic file name that encodes the coordinates
                h5_name = (
                    f"block_z{z0:06d}_{z1:06d}_"
                    f"y{y0:06d}_{y1:06d}_"
                    f"x{x0:06d}_{x1:06d}.h5"
                )
                h5_path = output_dir / h5_name

                _write_block(h5_path, block_np, h5_key, block_id)
                block_id += 1

    print(f"\n✅  Finished – wrote {block_id} HDF5 files to {output_dir}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_min_shape(s: str) -> tuple:
    """Parse a comma‑separated string like '128,1024,1024'."""
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "min_shape must be three comma‑separated integers, e.g. 128,1024,1024"
        )
    return tuple(int(p) for p in parts)


def main():
    parser = argparse.ArgumentParser(
        description="Split a 3‑D Zarr volume into many HDF5 files."
    )
    parser.add_argument(
        "--zarr_path",
        "-i",
        type=Path,
        default=Path(
            "/scratch-grete/projects/nim00007/data/mitochondria/moebius/volume_em/mobie/data/4007/images/ome-zarr/raw.ome.zarr"
        ),
        help="Path to the Zarr store (e.g. raw.ome.zarr).",
    )
    parser.add_argument(
        "--zarr_key",
        "-k",
        type=str,
        default="s0",
        help="Dataset key inside the Zarr store (default: 's0').",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        default="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/4007_split",
        help="Directory where the HDF5 blocks will be written.",
    )
    parser.add_argument(
        "--h5_key",
        "-hk",
        type=str,
        default="raw",
        help="Dataset name inside each HDF5 file (default: 'data').",
    )
    parser.add_argument(
        "--max_voxels",
        "-m",
        type=int,
        default=500 * 1000 * 1000,
        help="Maximum number of voxels per HDF5 file (default 5e8 ≈ 200×1600×1600).",
    )
    parser.add_argument(
        "--min_shape",
        "-ms",
        type=_parse_min_shape,
        default=(128, 1024, 1024),
        help="Minimum crop size as Z,Y,X (comma‑separated). "
             "Example: 128,1024,1024",
    )

    args = parser.parse_args()

    split_zarr_to_h5(
        zarr_path=args.zarr_path,
        output_dir=args.output_dir,
        max_voxels=args.max_voxels,
        zarr_key=args.zarr_key,
        h5_key=args.h5_key,
        min_shape=args.min_shape,
    )


if __name__ == "__main__":
    main()