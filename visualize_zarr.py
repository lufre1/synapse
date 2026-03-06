import argparse
import zarr
import napari
import numpy as np
import tifffile


def open_arr(path, key):
    store = zarr.DirectoryStore(path)          # zarr v2
    root = zarr.open(store=store, mode="r")    # group/array
    return root[key]


def main():
    parser = argparse.ArgumentParser(description="View a Zarr dataset and optional label TIFF in napari")
    parser.add_argument("--zarr_path", "-p", default="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/cutout_1/images/ome-zarr/raw.ome.zarr", help="Path to the Zarr file")
    parser.add_argument("--dataset_key", "-k", default=0, help="Key to the Zarr /group/dataset")
    parser.add_argument("--label_path", "-lp", default=None, help="Path to the labels TIFF file")
    parser.add_argument("--scale", "-s", type=int, default=1, help="Scale factor for the image")
    parser.add_argument("--second_zarr_path", "-sp", default=None, help="Path to the second Zarr file")
    parser.add_argument("--second_dataset_key", "-sk", default=None, help="Key to the Zarr /group/dataset")
    args = parser.parse_args()
    ndim = 3
    slicing = None
    if args.scale != 1:
        scale = args.scale
        slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))

    a1 = open_arr(args.zarr_path, args.dataset_key)
    if slicing is not None:
        a1 = a1[slicing]

    viewer = napari.Viewer()
    viewer.add_image(a1, name=f"zarr1:{args.dataset_key}")

    if args.second_zarr_path is not None:
        print("second_zarr_path", args.second_zarr_path)
        second_key = args.second_dataset_key
        if second_key is None:
            second_key = args.dataset_key  # default: same key as first
        if slicing is not None:
            a2 = open_arr(args.second_zarr_path, second_key)[slicing]
        else:
            a2 = open_arr(args.second_zarr_path, second_key)
        viewer.add_image(a2, name=f"zarr2:{second_key}")

    if args.label_path is not None:
        labels = tifffile.imread(args.label_path)
        if slicing is not None and labels.shape != a1.shape:
            print("slicing", slicing)
            labels = labels[slicing]
        viewer.add_labels(labels, name="Labels")

    napari.run()


if __name__ == "__main__":
    main()
