import argparse
import zarr
import napari
import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser(description="View a Zarr dataset and optional label TIFF in napari")
    parser.add_argument("--zarr_path", "-p", default="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/cutout_1/images/ome-zarr/raw.ome.zarr", help="Path to the Zarr file")
    parser.add_argument("--dataset_key", "-k", default=0, help="Key to the Zarr /group/dataset")
    parser.add_argument("--label_path", "-lp", default=None, help="Path to the labels TIFF file")
    args = parser.parse_args()

    arr = zarr.open(args.zarr_path, mode='r')[args.dataset_key][:]
    viewer = napari.view_image(arr, name=str(args.dataset_key))

    if args.label_path is not None:
        labels = tifffile.imread(args.label_path)
        viewer.add_labels(labels, name='Labels')

    napari.run()


if __name__ == "__main__":
    main()
