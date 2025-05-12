import argparse
from glob import glob
import os
from elf.io import open_file
import h5py
import numpy as np
from tqdm import tqdm

desired_dtypes = {
    "raw": np.uint8,
    "labels/mitochondria": np.uint8,
}


def compress_hdf5(input_path, output_path, compression="gzip", compression_level=4, chunk_size=(64, 64, 64)):
    """
    Compress an HDF5 file using chunking and compression.

    Args:
        input_path (str): Path to the original HDF5 file.
        output_path (str): Path for the compressed HDF5 file.
        compression (str): Compression algorithm ('gzip', 'lzf', etc.).
        compression_level (int): Compression level for 'gzip' (1–9).
        chunk_size (tuple): Chunk size for datasets.
    """
    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        def copy_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                shape = data.shape
                chunk_size = (1, 64, 64, 64) if len(shape) == 4 else (64, 64, 64)
                # Adjust chunk size so it does not exceed data shape in any dimension
                adjusted_chunk = tuple(min(c, s) for c, s in zip(chunk_size, shape))
                f_out.create_dataset(
                    name,
                    data=data,
                    compression=compression,
                    compression_opts=compression_level if compression == "gzip" else None,
                    chunks=adjusted_chunk
                )
            elif isinstance(obj, h5py.Group):
                f_out.create_group(name)

        f_in.visititems(copy_dataset)


def main():
    parser = argparse.ArgumentParser()
    # /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer
    parser.add_argument("--path", "-p",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/wichmann/extracted", help="Path to the root data directory")
    parser.add_argument("--ext", "-e", type=str, default=".h5")
    parser.add_argument("--key", "-k",  type=str, default=None, help="Key for the dataset to be compressed")
    parser.add_argument("--output", "-o",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/wichmann/extracted_compressed", help="Path to the output data directory")
    args = parser.parse_args()

    paths = sorted(glob(os.path.join(args.path, "**", f"*{args.ext}"), recursive=True))#, reverse=True)

    for p in tqdm(paths):
        # Compute relative path to maintain directory structure
        rel_path = os.path.relpath(p, args.path)
        out_path = os.path.join(args.output, rel_path)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        if os.path.isfile(out_path):
            print(f"File already exists: {out_path}")
            continue

        print(f"Compressing: {p}")
        compress_hdf5(p, out_path)
        print(f"Saved compressed file to: {out_path}")


if __name__ == "__main__":
    main()