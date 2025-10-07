#!/usr/bin/env python3
"""
Stitch a set of HDF5 blocks (produced by the splitter script) back into one
large TIFF file.

Usage
-----
    python stitch_h5_to_tif.py \
        -i /path/to/blocks_dir \
        -o /path/to/whole_volume.tif \
        -k data               # name of the dataset inside each HDF5 file
"""

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import tifffile as tif
import tqdm


# ----------------------------------------------------------------------
# Helper: parse the coordinates from the file name
# ----------------------------------------------------------------------
BLOCK_RE = re.compile(
    r"block_z(?P<z0>\d{6})_(?P<z1>\d{6})_"
    r"y(?P<y0>\d{6})_(?P<y1})_"
    r"x(?P<x0>\d{6})_(?P<x1>\d{6})\.h5$"
)


def parse_coords(fname: Path):
    """Return (z0, z1, y0, y1, x0, x1) as ints."""
    m = BLOCK_RE.search(fname.name)
    if not m:
        raise ValueError(f"File name does not match expected pattern: {fname}")
    return tuple(int(m.group(g)) for g in ("z0", "z1", "y01", "x0", "x1"))


# ----------------------------------------------------------------------
# Main stitching routine
# ----------------------------------------------------------------------
def stitch_blocks(
    blocks_dir: Path,
    out_tif: Path,
    h5_key: str = "data",
    dtype: np.dtype | None = None,
    compression: str = "lzw",
):
    """
    Parameters
    ----------
    blocks_dir : Path
        Directory that contains the ``block_*.h5`` files.
    out_tif : Path
        Destination TIFF file (will be overwritten if it exists).
    h5_key : str, optional
        Name of the dataset inside each HDF5 file (default: ``data``).
    dtype : np.dtype, optional
        Force a dtype for the output.  If ``None`` the dtype of the first
        block is used.
    compression : str, optional
        TIFF compression (``lzw`` works well for integer label volumes).
    """
    # ------------------------------------------------------------------
    # 1️⃣  Gather all block files and their coordinates
    # ------------------------------------------------------------------
    block_files = sorted(blocks_dir.glob("block_*.h5"))
    if not block_files:
        sys.exit(f"❌  No block_*.h5 files found in {blocks_dir}")

    coords = {}
    for bf in block_files:
        coords[bf] = parse_coords(bf)

    # ------------------------------------------------------------------
    # 2️⃣  Determine the full volume shape
    # ------------------------------------------------------------------
    max_z = max(c[1] for c in coords.values())
    max_y = max(c[3] for c in coords.values())
    max_x = max(c[5] for c in coords.values())
    volume_shape = (max_z, max_y, max_x)  # (Z, Y, X)

    print(f"Full volume shape (z, y, x): {volume_shape}")

    # ------------------------------------------------------------------
    # 3️⃣  Create a memmap (or a Zarr array) that will hold the whole volume
    # ------------------------------------------------------------------
    # Use a temporary .npy file as backing store – it is fast and works on
    # any filesystem.  The file is deleted automatically when the script
    # finishes.
    tmp_memmap_path = out_tif.with_suffix(".npy")
    if dtype is None:
        # Peek at the first block to get the dtype
        with h5py.File(block_files[0], "r") as f:
            dtype = f[h5_key].dtype
    volume = np.memmap(
        tmp_memmap_path,
        dtype=dtype,
        mode="w+",
        shape=volume_shape,
    )
    # Initialise with zeros (important for label volumes)
    volume[:] = 0

    # ------------------------------------------------------------------
    # 4️⃣  Fill the memmap block‑by‑block
    # ------------------------------------------------------------------
    for bf in tqdm.tqdm(block_files, desc="Stitching blocks"):
        z0, z1, y0, y1, x0, x1 = coords[bf]
        with h5py.File(bf, "r") as f:
            block = f[h5_key][:]
        # sanity check – block shape must match the coordinates
        if block.shape != (z1 - z0, y1 - y0, x1 - x0):
            raise RuntimeError(
                f"Shape mismatch in {bf}: "
                f"block.shape={block.shape}, expected={(z1 - z0, y1 - y0, x1 - x0)}"
            )
        volume[z0:z1, y0:y1, x0:x1] = block

    # ------------------------------------------------------------------
    # 5️⃣  Write the memmap to a single TIFF file
    # ------------------------------------------------------------------
    # tifffile can write directly from a NumPy array (including a memmap)
    print(f"Writing stitched volume to {out_tif} …")
    tif.imwrite(
        out_tif,
        volume,
        compression=compression,
        metadata=None,          # no OME metadata needed for a plain label stack
    )
    print("✅  Done.")

    # ------------------------------------------------------------------
    # 6️⃣  Clean up the temporary memmap file
    # ------------------------------------------------------------------
    del volume  # close the memmap
    tmp_memmap_path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stitch HDF5 blocks (produced by the splitter) into one TIFF."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing block_*.h5 files.",
    )
    parser.add_argument(
        "-o",
        "--output_tif",
        type=Path,
        required=True,
        help="Path of the stitched output TIFF.",
    )
    parser.add_argument(
        "-k",
        "--h5_key",
        type=str,
        default="data",
        help="Dataset name inside each HDF5 block (default: 'data').",
    )
    parser.add_argument(
        "-c",
        "--compression",
        type=str,
        default="lzw",
        help="TIFF compression (default: lzw).",
    )
    args = parser.parse_args()

    stitch_blocks(
        blocks_dir=args.input_dir,
        out_tif=args.output_tif,
        h5_key=args.h5_key,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()