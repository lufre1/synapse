import argparse
import zarr
import napari
import numpy as np
import tifffile
from scipy.ndimage import binary_dilation
from skimage.transform import resize
from napari.utils.colormaps.colormap import CyclicLabelColormap


def open_arr(path, key):
    store = zarr.DirectoryStore(path)
    root = zarr.open(store=store, mode="r")
    return root[key]


def _fixed_colormap(color_rgba):
    """Return a CyclicLabelColormap where every non-zero label maps to *color_rgba*.

    Uses 50 slots (napari default GPU texture size). All slots are the target
    color; napari's Labels layer always renders label-0 as transparent regardless
    of the colormap, so slot 0 being colored has no effect on the background.
    """
    colors = np.tile(np.array(color_rgba, dtype=np.float32), (50, 1))
    return CyclicLabelColormap(colors=colors)


def filter_labels(labels, ids):
    """Zero out all labels not in *ids* — in-place, no full copy."""
    labels[np.isin(labels, ids, invert=True)] = 0
    return labels


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


def nonzero_bbox(*arrays):
    """Return (slice_z, slice_y, slice_x) covering the union bounding box of non-zero content.

    Only considers 3D arrays. Returns None if all arrays are empty.
    """
    z0 = y0 = x0 = np.inf
    z1 = y1 = x1 = -np.inf
    found = False
    for arr in arrays:
        if arr is None or arr.ndim != 3:
            continue
        nz = np.nonzero(arr)
        if len(nz[0]) == 0:
            continue
        found = True
        z0 = min(z0, int(nz[0].min())); z1 = max(z1, int(nz[0].max()))
        y0 = min(y0, int(nz[1].min())); y1 = max(y1, int(nz[1].max()))
        x0 = min(x0, int(nz[2].min())); x1 = max(x1, int(nz[2].max()))
    if not found:
        return None
    return (slice(z0, z1 + 1), slice(y0, y1 + 1), slice(x0, x1 + 1))


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
    parser.add_argument("--no_downscale", "-nd", default=False, action="store_true",
                        help="Load all arrays at their original resolution — no downsampling "
                             "and no shape-matching resize. WARNING: may load tens of GB into RAM.")
    parser.add_argument("--filter_mito_by_axon", "-fma", default=False, action="store_true",
                        help="Only show mitos (third zarr, --seg3) that overlap with or touch "
                             "the axons kept by --filter_ids. Requires --seg and --seg3.")
    parser.add_argument("--fixed_colors", "-fc", default=False, action="store_true",
                        help="Color all axons (first zarr when --seg) yellow and all mitos "
                             "(third zarr when --seg3) blue.")
    parser.add_argument("--base_plane", "-bp", default=False, action="store_true",
                        help="Show raw image (second zarr) as a single flat plane using only "
                             "the first z-slice. Useful for a 3D view where segmentations "
                             "float above a 2D base image.")
    parser.add_argument("--export_path", "-ep", default=None,
                        help="Directory to export all napari layers as zarr files after closing "
                             "the viewer. Each layer is saved as <name>.zarr with key 'seg' "
                             "(Labels) or 'raw' (Image).")
    parser.add_argument("--scale_bar", "-sb", default=False, action="store_true",
                        help="Show a scale bar in the viewer (uses --voxel_size unit as 'nm')")
    parser.add_argument("--axes", "-ax", default=False, action="store_true",
                        help="Show 3D axis indicators in the viewer")
    parser.add_argument("--rotation_widget", "-rw", default=False, action="store_true",
                        help="Add an interactive rotation panel with elevation/azimuth/roll sliders")
    args = parser.parse_args()

    scale = args.scale
    z_start = args.z_start
    z_end = args.z_end
    no_z_scale = args.no_z_scale
    no_downscale = args.no_downscale
    voxel_size = args.voxel_size

    filter_ids = None
    if args.filter_ids is not None:
        filter_ids = [int(x.strip()) for x in args.filter_ids.split(',')]

    # --- load arrays ---
    raw1 = open_arr(args.zarr_path, args.dataset_key)
    if z_start is not None or z_end is not None:
        z_total = raw1.shape[-3]
        print(f"Z crop: [{z_start or 0}, {z_end or z_total}) of {z_total} slices")
    if no_downscale:
        scale1 = 1
        print(f"a1 original resolution: {raw1.shape} "
              f"({np.prod(raw1.shape) * np.dtype(raw1.dtype).itemsize / 1024**3:.1f} GB)")
    else:
        scale1 = scale if scale > 1 else auto_scale(raw1, z_start=z_start, z_end=z_end)
        if scale1 > 1 and scale == 1:
            print(f"Auto-scaling a1 by {scale1}x "
                  f"({np.prod(raw1.shape) * np.dtype(raw1.dtype).itemsize / 1024**3:.1f} GB)")
    a1 = lazy_downscale(raw1, scale1, z_start=z_start, z_end=z_end, no_z_scale=no_z_scale)
    if args.seg1 and a1.dtype == np.uint64:
        a1 = a1.astype(np.uint32)
    print(f"a1 loaded: {a1.shape} {a1.dtype}")

    a2 = None
    if args.second_zarr_path is not None:
        second_key = args.second_dataset_key or args.dataset_key
        raw2 = open_arr(args.second_zarr_path, second_key)
        if no_downscale:
            scale2 = 1
        else:
            scale2 = scale if scale > 1 else auto_scale(raw2, z_start=z_start, z_end=z_end)
            if scale2 > 1 and scale == 1:
                print(f"Auto-scaling a2 by {scale2}x "
                      f"({np.prod(raw2.shape) * np.dtype(raw2.dtype).itemsize / 1024**3:.1f} GB)")
        a2 = lazy_downscale(raw2, scale2, z_start=z_start, z_end=z_end, no_z_scale=no_z_scale)
        if args.seg2 and a2.dtype == np.uint64:
            a2 = a2.astype(np.uint32)
        print(f"a2 loaded: {a2.shape} {a2.dtype}")

    a3 = None
    if args.third_zarr_path is not None:
        third_key = args.third_dataset_key or args.dataset_key
        raw3 = open_arr(args.third_zarr_path, third_key)
        if no_downscale:
            scale3 = 1
        else:
            scale3 = scale if scale > 1 else auto_scale(raw3, z_start=z_start, z_end=z_end)
            if scale3 > 1 and scale == 1:
                print(f"Auto-scaling a3 by {scale3}x "
                      f"({np.prod(raw3.shape) * np.dtype(raw3.dtype).itemsize / 1024**3:.1f} GB)")
        a3 = lazy_downscale(raw3, scale3, z_start=z_start, z_end=z_end, no_z_scale=no_z_scale)
        if args.seg3 and a3.dtype == np.uint64:
            a3 = a3.astype(np.uint32)
        print(f"a3 loaded: {a3.shape} {a3.dtype}")

    # --- adjust voxel_size for effective downscale factor ---
    # The display scale factor is the one applied to the reference array (a2 if present, else a1).
    # All arrays are resized to match the reference shape, so one effective factor covers all.
    if voxel_size is not None and not no_downscale:
        display_scale = scale2 if a2 is not None else scale1
        if display_scale > 1:
            if no_z_scale:
                # Z was not downsampled; only Y/X were
                voxel_size = (voxel_size[0], voxel_size[1] * display_scale, voxel_size[2] * display_scale)
            else:
                voxel_size = tuple(v * display_scale for v in voxel_size)

    # --- record original shapes + scale factors for export upsampling ---
    _layer_meta = {
        f"zarr1:{args.dataset_key}": (tuple(raw1.shape[-3:]), scale1),
    }
    if a2 is not None:
        _layer_meta[f"zarr2:{second_key}"] = (tuple(raw2.shape[-3:]), scale2)
    if a3 is not None:
        _layer_meta[f"zarr3:{third_key}"] = (tuple(raw3.shape[-3:]), scale3)

    # --- apply filter_ids to seg1 before any resize (avoids large array before filter) ---
    seg1_data = a1
    if args.seg1 and filter_ids is not None:
        filter_labels(a1, filter_ids)   # in-place: a1 is modified, seg1_data is the same object
        if not np.any(a1):
            raise ValueError("No labels found after filtering")

    # --- filter mitos by axon contact BEFORE shape-match resize ---
    #     Stride-sample a3 down to a1's resolution (zero-copy numpy view) so
    #     binary_dilation and the ID lookup run on the smaller array.
    if args.filter_mito_by_axon and a3 is not None and args.seg1 and args.seg3:
        # Stride both arrays to their per-axis minimum shape (zero-copy views).
        min_shape = tuple(min(a3.shape[i], seg1_data.shape[i]) for i in range(3))
        st3   = tuple(max(1, round(a3.shape[i]        / min_shape[i])) for i in range(3))
        st_s1 = tuple(max(1, round(seg1_data.shape[i] / min_shape[i])) for i in range(3))
        a3_coarse   = a3[        ::st3[0],   ::st3[1],   ::st3[2]]
        seg1_coarse = seg1_data[ ::st_s1[0], ::st_s1[1], ::st_s1[2]]
        # Clip to identical shape in case of rounding edge cases
        clip = tuple(min(a3_coarse.shape[i], seg1_coarse.shape[i]) for i in range(3))
        a3_coarse   = a3_coarse[  :clip[0], :clip[1], :clip[2]]
        seg1_coarse = seg1_coarse[:clip[0], :clip[1], :clip[2]]
        if any(s > 1 for s in st3) or any(s > 1 for s in st_s1):
            print(f"  Contact filter: a3 {a3.shape}→{a3_coarse.shape}, seg1 {seg1_data.shape}→{seg1_coarse.shape}")
        axon_contact_mask = binary_dilation(seg1_coarse > 0)
        contact_mito_ids = np.unique(a3_coarse[axon_contact_mask])
        del axon_contact_mask, a3_coarse, seg1_coarse
        contact_mito_ids = contact_mito_ids[contact_mito_ids != 0]
        print(f"Mito contact filter: {len(contact_mito_ids)} mito(s) touch/overlap filtered axons.")
        filter_labels(a3, contact_mito_ids)   # in-place

    # --- match shapes: raw (a2) is the reference; resize labels to match it.
    #     Skipped with --no_downscale so arrays stay at their original resolutions.
    if not no_downscale:
        ref_shape = a2.shape if a2 is not None else a1.shape

        if a1.shape != ref_shape:
            print(f"Resizing a1 {a1.shape} → {ref_shape} (nearest-neighbour)")
            a1 = chunked_resize(a1, ref_shape, order=0, slab_size=64)
            seg1_data = a1

        if a2 is not None and a2.shape != ref_shape:
            order2 = 0 if args.seg2 else 1
            print(f"Resizing a2 {a2.shape} → {ref_shape} ({'nearest-neighbour' if args.seg2 else 'linear'})")
            a2 = chunked_resize(a2, ref_shape, order=order2, slab_size=64)

        if a3 is not None and a3.shape != ref_shape:
            order3 = 0 if args.seg3 else 1
            print(f"Resizing a3 {a3.shape} → {ref_shape} ({'nearest-neighbour' if args.seg3 else 'linear'})")
            a3 = chunked_resize(a3, ref_shape, order=order3, slab_size=64)

    # --- tight bbox crop when --axes is active so axes sit close to actual data ---
    if args.axes:
        seg_arrays = [
            a1 if args.seg1 else None,
            a2 if args.seg2 else None,
            a3 if args.seg3 else None,
        ]
        bbox = nonzero_bbox(*seg_arrays)
        if bbox is None:
            bbox = nonzero_bbox(a1, a2, a3)
        if bbox is not None:
            print(f"Axes bbox crop: z={bbox[0]}, y={bbox[1]}, x={bbox[2]}")
            if a1 is not None and a1.ndim == 3:
                a1 = a1[bbox]
                seg1_data = a1
            if a2 is not None and a2.ndim == 3:
                a2 = a2[bbox]
            if a3 is not None and a3.ndim == 3:
                a3 = a3[bbox]

    # --- base plane: reduce raw (a2) to first z-slice only ---
    a2_scale = voxel_size  # default scale for a2
    if args.base_plane and a2 is not None and not args.seg2:
        a2 = a2[0]  # (Y, X) — napari displays 2D images as a flat plane in 3D mode
        a2_scale = voxel_size[1:] if (voxel_size is not None and len(voxel_size) == 3) else voxel_size
        print(f"Base-plane mode: raw reduced to first z-slice {a2.shape}")

    # --- build viewer ---
    viewer = napari.Viewer()

    if args.scale_bar:
        viewer.scale_bar.visible = True
        if voxel_size is not None:
            viewer.scale_bar.unit = "nm"
    if args.axes:
        viewer.axes.visible = True

    if not args.seg1:
        viewer.add_image(a1, name=f"zarr1:{args.dataset_key}", scale=voxel_size)
    else:
        layer1 = viewer.add_labels(seg1_data, name=f"zarr1:{args.dataset_key}", scale=voxel_size)
        if args.fixed_colors:
            layer1.colormap = _fixed_colormap([1.0, 1.0, 0.0, 1.0])  # yellow

    if a2 is not None:
        if not args.seg2:
            viewer.add_image(a2, name=f"zarr2:{second_key}", scale=a2_scale)
        else:
            viewer.add_labels(a2, name=f"zarr2:{second_key}", scale=voxel_size)

    if a3 is not None:
        if not args.seg3:
            viewer.add_image(a3, name=f"zarr3:{third_key}", scale=voxel_size)
        else:
            layer3 = viewer.add_labels(a3, name=f"zarr3:{third_key}", scale=voxel_size)
            if args.fixed_colors:
                layer3.colormap = _fixed_colormap([0.0, 0.0, 1.0, 1.0])  # blue

    if args.label_path is not None:
        labels = tifffile.memmap(args.label_path, mode="r")  # memory-mapped: no full load
        _layer_meta["Labels"] = (tuple(labels.shape[-3:]), scale)
        labels = lazy_downscale(labels, scale, z_start=z_start, z_end=z_end, no_z_scale=no_z_scale)
        if scale > 1:
            print(f"label TIFF downscaled to {labels.shape}")
        if filter_ids is not None:
            labels = filter_labels(labels, filter_ids)
        viewer.add_labels(labels, name="Labels", scale=voxel_size)

    if args.rotation_widget:
        from magicgui import magicgui

        _syncing = [False]  # guard against slider↔camera feedback loop

        @magicgui(
            elevation={"widget_type": "FloatSlider", "min": -180, "max": 180, "value": 0},
            azimuth={"widget_type": "FloatSlider", "min": -180, "max": 180, "value": 0},
            roll={"widget_type": "FloatSlider", "min": -180, "max": 180, "value": 0},
            auto_call=True,
            call_button=False,
        )
        def _rotation_widget(elevation: float = 0, azimuth: float = 0, roll: float = 0):
            if _syncing[0]:
                return
            _syncing[0] = True
            viewer.camera.angles = (elevation, azimuth, roll)
            _syncing[0] = False

        def _on_camera_angles(event):
            if _syncing[0]:
                return
            _syncing[0] = True
            e, a, r = viewer.camera.angles
            _rotation_widget.elevation.value = e
            _rotation_widget.azimuth.value = a
            _rotation_widget.roll.value = r
            _syncing[0] = False

        viewer.camera.events.angles.connect(_on_camera_angles)
        viewer.window.add_dock_widget(_rotation_widget, name="Rotation", area="right")

    napari.run()

    # --- export layers to zarr after viewer closes ---
    if args.export_path is not None:
        import os
        os.makedirs(args.export_path, exist_ok=True)
        for layer in viewer.layers:
            safe_name = (
                layer.name.replace(":", "_").replace("/", "_").replace(" ", "_")
            )
            out_zarr = os.path.join(args.export_path, f"{safe_name}.zarr")
            data = np.asarray(layer.data)

            # Upsample back to original input resolution if the layer was downscaled.
            orig_shape, sf = _layer_meta.get(layer.name, (None, 1))
            if sf > 1 and orig_shape is not None and data.shape[-3:] != orig_shape:
                is_seg = isinstance(layer, napari.layers.Labels)
                order = 0 if is_seg else 1
                interp = "nearest-neighbour" if is_seg else "linear"
                print(f"  Upsampling {layer.name!r} {data.shape} → {orig_shape} ({interp})")
                if data.ndim == 3:
                    data = chunked_resize(data, orig_shape, order=order, slab_size=64)
                elif data.ndim == 4:
                    # channel-first: upsample each channel independently
                    out4 = np.empty((data.shape[0],) + orig_shape, dtype=data.dtype)
                    for c in range(data.shape[0]):
                        out4[c] = chunked_resize(data[c], orig_shape, order=order, slab_size=64)
                    data = out4

            if data.ndim == 3:
                chunks = (min(64, data.shape[0]), min(256, data.shape[1]), min(256, data.shape[2]))
            elif data.ndim == 4:
                chunks = (data.shape[0], min(64, data.shape[1]), min(256, data.shape[2]), min(256, data.shape[3]))
            else:
                chunks = True
            key = "seg" if isinstance(layer, napari.layers.Labels) else "raw"
            store = zarr.DirectoryStore(out_zarr)
            root = zarr.open(store, mode="w")
            ds = root.create_dataset(
                key, data=data, chunks=chunks, dtype=data.dtype,
                compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
                overwrite=True,
            )
            if args.voxel_size is not None:
                ds.attrs["voxel_size_zyx_nm"] = list(args.voxel_size)
            print(f"Exported {layer.name!r} ({data.shape} {data.dtype}) → {out_zarr}[{key!r}]")


if __name__ == "__main__":
    main()
