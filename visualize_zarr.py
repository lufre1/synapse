import argparse
import zarr
import napari
import numpy as np
import tifffile
from skimage.transform import resize


def open_arr(path, key):
    store = zarr.DirectoryStore(path)
    root = zarr.open(store=store, mode="r")
    return root[key]


def filter_labels(labels, ids):
    """Keep only specified label IDs, set all others to 0."""
    mask = np.isin(labels, ids)
    return labels * mask


def auto_scale(arr, max_gb=2.0, z_start=None, z_end=None):
    """Return the smallest integer scale that keeps the loaded array under max_gb."""
    import math
    shape = list(arr.shape)
    if arr.ndim >= 3:
        z_size = shape[-3]
        shape[-3] = (z_end if z_end is not None else z_size) - (z_start or 0)
    nbytes = np.prod(shape) * np.dtype(arr.dtype).itemsize
    if nbytes <= max_gb * 1024 ** 3:
        return 1
    scale = math.ceil((nbytes / (max_gb * 1024 ** 3)) ** (1 / 3))
    return max(2, scale)


def lazy_downscale(arr, factor, z_start=None, z_end=None, no_z_scale=False):
    """Stride-based downscale that reads one Z-slab at a time.

    Bounds peak RAM to one decompressed slab instead of the full volume.
    Works for both zarr arrays and memory-mapped TIFFs.

    z_start / z_end are optional crop bounds in the *original* z coordinates.
    Only data within [z_start, z_end) is ever read from disk.

    no_z_scale: if True, keep all z slices and only downsample y/x.
    """
    ndim = arr.ndim
    z_size = arr.shape[-3]
    z_lo = z_start if z_start is not None else 0
    z_hi = min(z_end, z_size) if z_end is not None else z_size
    z_lo = max(0, z_lo)
    z_step = 1 if no_z_scale else factor

    if factor <= 1:
        if ndim == 3:
            return np.asarray(arr[z_lo:z_hi])
        else:
            return np.asarray(arr[:, z_lo:z_hi])

    # Use the array's native Z-chunk size as slab height (fall back to 64).
    if hasattr(arr, "chunks") and arr.chunks is not None:
        slab_z = int(arr.chunks[-3]) if ndim >= 3 else 1
    else:
        slab_z = 64
    slab_z = max(slab_z, factor)  # need at least `factor` rows to emit one output row

    z_cropped = z_hi - z_lo
    out_z_len = len(range(0, z_cropped, z_step))
    if ndim == 3:
        out_shape = (out_z_len,) + tuple(len(range(0, s, factor)) for s in arr.shape[-2:])
    else:
        out_shape = (arr.shape[0], out_z_len) + tuple(len(range(0, s, factor)) for s in arr.shape[-2:])
    out = np.empty(out_shape, dtype=arr.dtype)

    out_z = 0
    for z0 in range(z_lo, z_hi, slab_z):
        z1 = min(z0 + slab_z, z_hi)
        if no_z_scale:
            z_first = z0
        else:
            # Align slab start to the crop-relative ::factor grid.
            offset = (z0 - z_lo) % factor
            z_first = z0 + ((-offset) % factor)
        if z_first >= z1:
            continue
        if ndim == 3:
            slab = np.asarray(arr[z_first:z1:z_step, ::factor, ::factor])
            n = slab.shape[0]
            out[out_z:out_z + n] = slab
        else:                            # 4-D channel-first (C, Z, Y, X)
            slab = np.asarray(arr[:, z_first:z1:z_step, ::factor, ::factor])
            n = slab.shape[1]
            out[:, out_z:out_z + n] = slab
        out_z += n

    return out


def chunked_resize(arr, target_shape, order=1, slab_size=64):
    """Resize a 3D array in Z-slabs to bound memory usage.

    Args:
        arr: Input array (3D)
        target_shape: Target output shape
        order: Interpolation order (0=nearest, 1=linear, etc.)
        slab_size: Number of input Z-slices to process at once

    Returns:
        Resized array with target_shape
    """
    from skimage.transform import resize

    if arr.shape == target_shape:
        return np.asarray(arr)

    z_in, y_in, x_in = arr.shape
    z_out, y_out, x_out = target_shape

    out = np.empty(target_shape, dtype=arr.dtype)

    z_scale = z_out / z_in

    for z0 in range(0, z_in, slab_size):
        z1 = min(z0 + slab_size, z_in)
        slab = arr[z0:z1]

        out_z_start = int(round(z0 * z_scale))
        out_z_end = int(round(z1 * z_scale))
        slab_z_out = out_z_end - out_z_start

        if slab_z_out == 0:
            continue

        slab_target = (slab_z_out, y_out, x_out)
        resized_slab = resize(
            slab, slab_target,
            order=order,
            preserve_range=True,
            anti_aliasing=(order > 0)
        ).astype(arr.dtype)

        out[out_z_start:out_z_end] = resized_slab

    return out


def main():
    parser = argparse.ArgumentParser(description="View a Zarr dataset and optional label TIFF in napari")
    parser.add_argument("--zarr_path", "-p", default="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/cutout_1/images/ome-zarr/raw.ome.zarr", help="Path to the Zarr file")
    parser.add_argument("--dataset_key", "-k", default=0, help="Key to the Zarr /group/dataset")
    parser.add_argument("--seg1", "-seg1", "--seg", "-seg", "--is_segmentation", default=False, action="store_true", help="Treat zarr1 as a segmentation (labels layer)")
    parser.add_argument("--seg2", "-seg2", default=False, action="store_true", help="Treat zarr2 as a segmentation (labels layer)")
    parser.add_argument("--seg3", "-seg3", default=False, action="store_true", help="Treat zarr3 as a segmentation (labels layer)")
    parser.add_argument("--label_path", "-lp", default=None, help="Path to the labels TIFF file")
    parser.add_argument("--scale", "-s", type=int, default=1,
                        help="Integer downscale factor applied to every layer before display "
                             "(images: local-mean pooling; segmentations: nearest-neighbour)")
    parser.add_argument("--second_zarr_path", "-sp", default=None, help="Path to the second Zarr file")
    parser.add_argument("--second_dataset_key", "-sk", default=None, help="Key to the Zarr /group/dataset")
    parser.add_argument("--third_zarr_path", "-tp", default=None, help="Path to the third Zarr file")
    parser.add_argument("--third_dataset_key", "-tk", default=None, help="Key to the Zarr /group/dataset")
    parser.add_argument(
        "--voxel_size", "-vs",
        type=lambda x: tuple(map(float, x.split(','))) if ',' in x else (float(x),) * 3,
        default=None,
        help="Voxel size in nm, either a single float (e.g., 12) or a tuple (e.g., 12,12,12)"
    )
    parser.add_argument("--filter_ids", "-fid", type=str, default=None,
                        help="Comma-separated list of label IDs to display (e.g., '1,3,5,7')")
    parser.add_argument("--z_start", "-zs", type=int, default=None,
                        help="First z-slice to load (original resolution, inclusive)")
    parser.add_argument("--z_end", "-ze", type=int, default=None,
                        help="Last z-slice to load (original resolution, exclusive)")
    parser.add_argument("--no_z_scale", "-nzs", default=False, action="store_true",
                        help="Keep all z slices at original resolution; only downsample y/x")
    args = parser.parse_args()

    scale = args.scale
    z_start = args.z_start
    z_end = args.z_end
    no_z_scale = args.no_z_scale
    voxel_size = args.voxel_size
    if voxel_size is not None and scale > 1:
        voxel_size = tuple(v * scale for v in voxel_size)

    filter_ids = None
    if args.filter_ids is not None:
        filter_ids = [int(x.strip()) for x in args.filter_ids.split(',')]

    # --- load arrays (lazy downscale: slice zarr on disk, never load full volume) ---
    raw1 = open_arr(args.zarr_path, args.dataset_key)
    if z_start is not None or z_end is not None:
        z_total = raw1.shape[-3]
        print(f"Z crop: [{z_start or 0}, {z_end or z_total}) of {z_total} slices")
    scale1 = scale if scale > 1 else auto_scale(raw1, z_start=z_start, z_end=z_end)
    if scale1 > 1 and scale == 1:
        print(f"Auto-scaling a1 by {scale1}x "
              f"({np.prod(raw1.shape) * np.dtype(raw1.dtype).itemsize / 1024**3:.1f} GB)")
    a1 = lazy_downscale(raw1, scale1, z_start=z_start, z_end=z_end, no_z_scale=no_z_scale)
    print(f"a1 loaded: {a1.shape}")

    # Load second and third arrays early so we can match shapes afterwards.
    a2 = None
    if args.second_zarr_path is not None:
        second_key = args.second_dataset_key or args.dataset_key
        raw2 = open_arr(args.second_zarr_path, second_key)
        scale2 = scale if scale > 1 else auto_scale(raw2, z_start=z_start, z_end=z_end)
        if scale2 > 1 and scale == 1:
            print(f"Auto-scaling a2 by {scale2}x "
                  f"({np.prod(raw2.shape) * np.dtype(raw2.dtype).itemsize / 1024**3:.1f} GB)")
        a2 = lazy_downscale(raw2, scale2, z_start=z_start, z_end=z_end, no_z_scale=no_z_scale)
        print(f"a2 loaded: {a2.shape}")

    a3 = None
    if args.third_zarr_path is not None:
        third_key = args.third_dataset_key or args.dataset_key
        raw3 = open_arr(args.third_zarr_path, third_key)
        scale3 = scale if scale > 1 else auto_scale(raw3, z_start=z_start, z_end=z_end)
        if scale3 > 1 and scale == 1:
            print(f"Auto-scaling a3 by {scale3}x "
                  f"({np.prod(raw3.shape) * np.dtype(raw3.dtype).itemsize / 1024**3:.1f} GB)")
        a3 = lazy_downscale(raw3, scale3, z_start=z_start, z_end=z_end, no_z_scale=no_z_scale)
        print(f"a3 loaded: {a3.shape}")

    # --- match shapes: raw (a2) is the reference; resize labels to match it ---
    # If there is no a2, fall back to a1 as reference.
    # Segmentations always use order=0 (nearest-neighbour) with no anti-aliasing.
    ref_shape = a2.shape if a2 is not None else a1.shape

    if a1.shape != ref_shape:
        print(f"Resizing a1 {a1.shape} → {ref_shape} (nearest-neighbour)")
        a1 = chunked_resize(a1, ref_shape, order=0, slab_size=64)

    if a2 is not None and a2.shape != ref_shape:
        order2 = 0 if args.seg2 else 1
        print(f"Resizing a2 {a2.shape} → {ref_shape} ({'nearest-neighbour' if args.seg2 else 'linear'})")
        a2 = chunked_resize(a2, ref_shape, order=order2, slab_size=64)

    if a3 is not None and a3.shape != ref_shape:
        order3 = 0 if args.seg3 else 1
        print(f"Resizing a3 {a3.shape} → {ref_shape} ({'nearest-neighbour' if args.seg3 else 'linear'})")
        a3 = chunked_resize(a3, ref_shape, order=order3, slab_size=64)

    # --- build viewer ---
    viewer = napari.Viewer()

    if not args.seg1:
        viewer.add_image(a1, name=f"zarr1:{args.dataset_key}", scale=voxel_size)
    else:
        seg_data = a1
        if filter_ids is not None:
            seg_data = filter_labels(seg_data, filter_ids)
            if not np.any(seg_data):
                raise ValueError("No labels found after filtering")
        viewer.add_labels(seg_data, name=f"zarr1:{args.dataset_key}", scale=voxel_size)

    if a2 is not None:
        if not args.seg2:
            viewer.add_image(a2, name=f"zarr2:{second_key}", scale=voxel_size)
        else:
            viewer.add_labels(a2, name=f"zarr2:{second_key}", scale=voxel_size)

    if a3 is not None:
        if not args.seg3:
            viewer.add_image(a3, name=f"zarr3:{third_key}", scale=voxel_size)
        else:
            viewer.add_labels(a3, name=f"zarr3:{third_key}", scale=voxel_size)

    if args.label_path is not None:
        labels = tifffile.memmap(args.label_path, mode="r")  # memory-mapped: no full load
        labels = lazy_downscale(labels, scale, z_start=z_start, z_end=z_end, no_z_scale=no_z_scale)
        if scale > 1:
            print(f"label TIFF downscaled to {labels.shape}")
        if filter_ids is not None:
            labels = filter_labels(labels, filter_ids)
        viewer.add_labels(labels, name="Labels", scale=voxel_size)

    napari.run()


if __name__ == "__main__":
    main()
