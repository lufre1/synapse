import argparse
import zarr
import napari
import numpy as np
import tifffile


def open_arr(path, key):
    store = zarr.DirectoryStore(path)          # zarr v2
    root = zarr.open(store=store, mode="r")    # group/array
    return root[key]


def filter_labels(labels, ids):
    """Keep only specified label IDs, set all others to 0."""
    mask = np.isin(labels, ids)
    return labels * mask


def main():
    parser = argparse.ArgumentParser(description="View a Zarr dataset and optional label TIFF in napari")
    parser.add_argument("--zarr_path", "-p", default="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/cutout_1/images/ome-zarr/raw.ome.zarr", help="Path to the Zarr file")
    parser.add_argument("--dataset_key", "-k", default=0, help="Key to the Zarr /group/dataset")
    parser.add_argument("--is_segmentation", "-seg", "--seg", default=False, action="store_true")
    parser.add_argument("--label_path", "-lp", default=None, help="Path to the labels TIFF file")
    parser.add_argument("--scale", "-s", type=int, default=1, help="Scale factor for the image")
    parser.add_argument("--second_zarr_path", "-sp", default=None, help="Path to the second Zarr file")
    parser.add_argument("--second_dataset_key", "-sk", default=None, help="Key to the Zarr /group/dataset")
    parser.add_argument(
        "--voxel_size", "-vs",
        type=lambda x: tuple(map(float, x.split(','))) if ',' in x else (float(x),) * 3,  # Always return a tuple
        default=None,
        help="Voxel size in nm, either a single float (e.g., 12) or a tuple (e.g., 12,12,12)"
    )
    parser.add_argument("--filter_ids", "-fid", type=str, default=None,
                        help="Comma-separated list of label IDs to display (e.g., '1,3,5,7')")
    args = parser.parse_args()
    voxel_size = None
    if args.voxel_size is not None:
        voxel_size = args.voxel_size
    
    filter_ids = None
    if args.filter_ids is not None:
        filter_ids = [int(x.strip()) for x in args.filter_ids.split(',')]
    
    ndim = 3
    slicing = None
    if args.scale != 1:
        scale = args.scale
        slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))

    a1 = open_arr(args.zarr_path, args.dataset_key)
    if slicing is not None:
        a1 = a1[slicing]
    viewer = napari.Viewer()
    if not args.is_segmentation:
        viewer.add_image(a1, name=f"zarr1:{args.dataset_key}", scale=voxel_size)
    else:
        seg_data = a1
        if filter_ids is not None:
            seg_data = filter_labels(seg_data, filter_ids)
            if not np.any(seg_data):
                raise ValueError("No labels found after filtering")
        viewer.add_labels(seg_data, name=f"zarr1:{args.dataset_key}", scale=voxel_size)

    if args.second_zarr_path is not None:
        print("second_zarr_path", args.second_zarr_path)
        second_key = args.second_dataset_key
        if second_key is None:
            second_key = args.dataset_key  # default: same key as first
        if slicing is not None:
            a2 = open_arr(args.second_zarr_path, second_key)[slicing]
        else:
            a2 = open_arr(args.second_zarr_path, second_key)
        viewer.add_image(a2, name=f"zarr2:{second_key}", scale=voxel_size)

    if args.label_path is not None:
        labels = tifffile.imread(args.label_path)
        if slicing is not None and labels.shape != a1.shape:
            print("slicing", slicing)
            labels = labels[slicing]
        if filter_ids is not None:
            labels = filter_labels(labels, filter_ids)
        viewer.add_labels(labels, name="Labels", scale=voxel_size)

    napari.run()


if __name__ == "__main__":
    main()
