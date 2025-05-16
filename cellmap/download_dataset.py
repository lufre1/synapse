import argparse
import zarr
import numpy as np
# from elf.io import open_file
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from synapse_net.file_utils import read_ome_zarr


def s3_join(*args, trailing_slash=False):
    path = '/'.join([str(a).strip('/').strip() for a in args if a])
    return path + '/' if trailing_slash and not path.endswith('/') else path


# def download_zarr_subtree(bucket, remote_group, local_dest):
#     b = q3.Bucket(f"s3://{bucket}")
#     b.fetch(remote_group, local_dest)
#     print(f"Downloaded s3://{bucket}/{remote_group} → {local_dest}")

def download_zarr_subtree(bucket, remote_group, local_dest, progress_bar=True):
    import quilt3 as q3
    import os
    from tqdm import tqdm
    # breakpoint()
    b = q3.Bucket(f"s3://{bucket}")
    keys = [k for k in b.keys() if k.startswith(remote_group)]
    if progress_bar:
        iterator = tqdm(keys, desc="Downloading zarr objects", total=len(keys))
    else:
        iterator = keys

    for k in iterator:
        rel = k[len(remote_group):].lstrip('/')
        dest_path = os.path.join(local_dest, rel)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        b.fetch(k, dest_path)
    print("Download finished.")


def main(args):
    bucket = args.bucket
    zarr_path = args.zarr_path
    dataset = args.dataset
    resolution = args.resolution
    local_dest = args.out

    # Compose S3 path
    # Example: jrc_hela-bfa.zarr/volumes/raw/s0
    # add trailing slash if not present
    remote_zarr_group = s3_join(zarr_path, dataset, resolution, trailing_slash=True)
    print(f"Downloading: s3://{bucket}/{remote_zarr_group} to {local_dest}")

    # Download the (sub)tree to the desired path (e.g. ./my_local_folder)
    download_zarr_subtree(bucket, remote_zarr_group, local_dest)

    # You can now open your freshly downloaded Zarr
    print(f"Done. To open in zarr: zarr.open('{local_dest}', mode='r')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, default="janelia-cosem-datasets")
    parser.add_argument("--zarr_path", type=str, default="jrc_mus-liver-zon-1/jrc_mus-liver-zon-1.zarr/",
                        help="e.g. jrc_hela-bfa.zarr")
    parser.add_argument("--dataset", type=str, default="recon-1/em/fibsem-uint8/",
                        help="e.g. volumes/raw")
    parser.add_argument("--resolution", type=str, default="s0")
    parser.add_argument("--out", type=str, default="/scratch-grete/projects/nim00007/data/cellmap/datasets/",
                        help="Destination folder to save the Zarr group, e.g. ./my_zarr_download")
    args = parser.parse_args()
    main(args)