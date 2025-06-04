import argparse
from elf.io import open_file
import zarr
import numpy as np


def main(args):
    # Paths
    """
    Copy a dataset from N5 to Zarr, for a given input/output path.
    
    This script is used to copy the label data from N5 to Zarr format.
    cellmap data is available in zarr but labels are only avaible in n5.
    Before copying, it checks if the shapes of the N5 label dataset and the Zarr raw dataset are identical.
    If not, it stops with an error message.
    """
    n5_path = args.input
    zarr_path = args.output
    n5_label_path = "labels/mito-seg/s0"
    zarr_label_path = "labels/mito-seg/s0"  # <-- choose target location/group in Zarr
    zarr_raw_path = "recon-1/em/fibsem-uint8/s0"
    
    print("n5 path and zarr path:", "\n", n5_path, "\n", zarr_path)

    # Open source (N5) and target (Zarr)
    with open_file(n5_path, 'r') as n5f, open_file(zarr_path, 'a') as zf:
        n5_shape = n5f[n5_label_path].shape
        zarr_shape = zf[zarr_raw_path].shape

        print(f"N5 label dataset shape: {n5_shape}")
        print(f"Zarr raw dataset shape: {zarr_shape}")

        if n5_shape == zarr_shape:
            print("✅ Shapes are identical. Safe to copy!")
        else:
            print("❌ ERROR: Shapes are NOT identical!")
            return None  # Exit with error code

        n5data = n5f[n5_label_path]

        # Optionally: create intermediate groups in zarr if needed
        zarr_group = zf
        if "/" in zarr_label_path:
            for part in zarr_label_path.split("/")[:-1]:
                zarr_group = zarr_group.require_group(part)
            zarr_dataset_name = zarr_label_path.split("/")[-1]
        else:
            zarr_dataset_name = zarr_label_path

        # Copy the array (efficient for reasonable array sizes; for massive datasets, consider chunked copy)
        print(f"Copying data from N5:{n5_label_path} to Zarr:{zarr_label_path}")
        arr = np.array(n5data)  # Load the whole thing to memory

        # Use the same chunk shape and dtype, if possible
        zarr_group.create_dataset(
            zarr_dataset_name,
            data=arr,
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=n5data.chunks,
            compression='gzip'  # or set as needed
        )

    print("Done!")


if __name__ == "__main__":
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--input", "-i", type=str, default="/mnt/ceph-hdd/cold_store/projects/nim00007/cellmap/datasets/janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.n5")
    argsparse.add_argument("--output", "-o", type=str, default="/mnt/ceph-hdd/cold_store/projects/nim00007/cellmap/datasets/janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.zarr")
    args = argsparse.parse_args()
    
    main(args)
