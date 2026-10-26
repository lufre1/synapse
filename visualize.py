# import time
from typing import Any, Dict
# from config import *
import h5py
import zarr
import argparse
from contextlib import ExitStack
import multiprocessing
import os
from glob import glob
import numpy as np
import napari
# import elf.parallel as parallel
from elf.io import open_file
from elf.parallel import label
from tqdm import tqdm
import z5py
from tifffile import TiffFile
from skimage.transform import resize
import dask.array as da


def _segment(foreground, boundary, block_shape=(128, 128, 128), threshold=0.5):
    foreground_mask = np.where(foreground > threshold, 1, 0)
    boundary_mask = np.where(boundary > threshold, 1, 0)
    mask = np.logical_or(foreground_mask, np.logical_and(foreground_mask, boundary_mask))
    seg = label(mask, block_shape=block_shape)
    return seg


def get_file_paths(path, ext=".h5", reverse=False):
    if ext in path:
        return [path]
    else:
        paths = sorted(glob(os.path.join(path, "**", f"*{ext}"), recursive=True), reverse=reverse)
        return paths


def visualize_data(data, name=None, offset_z=None, args=None, voxel_size=None):
    viewer = napari.Viewer()
    viewer.title = name or ""
    if args is not None and args.voxel_size is not None:
        voxel_size = args.voxel_size
    if voxel_size is not None:
        viewer.dims.axis_labels = ("z", "y", "x")

    for key, value in data.items():
        if "raw" in key or key == "0":
            viewer.add_image(value, name=key, scale=voxel_size)
        elif key == "prediction" or "pred" in key or "dist" in key or "fore" in key or "bound" in key:
            viewer.add_image(value, name=key, blending="additive", scale=voxel_size)
        else:
            # avoid value.astype(...) here if value is lazy (it will compute)
            if offset_z is not None:
                viewer.add_labels(value, name=key, translate=(0, 0, offset_z), scale=voxel_size)
            else:
                viewer.add_labels(value, name=key, scale=voxel_size)
    # Get the "raw" layer
    raw_layer = next((layer for layer in viewer.layers if "raw" in layer.name), None)
    if raw_layer:
        # Remove the "raw" layer from its current position
        viewer.layers.remove(raw_layer)
        # Add the "raw" layer to the beginning of the layer list
        viewer.layers.insert(0, raw_layer)

    napari.run()


def _read(ds, scale: int = 1, block: int = 64):
    """Read a dataset, keeping every scale-th voxel along the last three axes.

    h5py strided reads are ~2.5x slower than plain ones, so z-blocks are read whole and strided in numpy.
    """
    if scale == 1:
        return ds[:]
    slicing = tuple(slice(None, None, scale) if i >= ds.ndim - 3 else slice(None) for i in range(ds.ndim))
    ax, step = max(ds.ndim - 3, 0), block * scale  # step is a multiple of scale, so the stride stays in phase
    blocks = [ds[(slice(None),) * ax + (slice(i, i + step),)][slicing] for i in range(0, ds.shape[ax], step)]
    return np.concatenate(blocks, axis=ax)


def _as_lazy(item, scale: int = 1) -> da.Array:
    """Wrap a dataset as a lazy dask array, keeping every scale-th voxel along the last three axes."""
    # unchunked data (mrc memmaps, contiguous h5) reads any plane cheaply, so dask gets one chunk per z-plane
    chunks = getattr(item, "chunks", None) or (1,) * (item.ndim - 2) + item.shape[-2:]
    slicing = tuple(slice(None, None, scale) if i >= item.ndim - 3 else slice(None) for i in range(item.ndim))
    return da.from_array(item, chunks=chunks)[slicing]


def extract_data(group: Any, data: Dict[str, Any], prefix: str = "", scale: int = 1, lazy: bool = False):
    """
    Recursively extract datasets from a group and store them in a dictionary.
    """
    for key, item in group.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(item, (zarr.Group, h5py.Group, z5py.Group)):
            # Recursively extract data from subgroups
            extract_data(item, data, prefix=full_key, scale=scale, lazy=lazy)
        else:
            data[full_key] = _as_lazy(item, scale) if lazy else _read(item, scale)


def upsample_data(data, factor):
    """Upsample a 3D dataset in chunks to avoid memory overload."""
    upsampled_data = np.zeros(tuple(dim * factor for dim in data.shape), dtype=data.dtype)
    for z in range(data.shape[0]):
        upsampled_data[z * factor: (z + 1) * factor] = resize(
            data[z],
            (factor * data.shape[1], factor * data.shape[2]),
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(data.dtype)
    return upsampled_data


def _read_tif_labels(path, scale=1, block=64):
    """Read a label tif into the smallest integer dtype that holds its values.

    Predictions are often saved as (u)int64, which costs 8x the RAM of uint8 for a few dozen ids, so z-stacks
    are read in blocks of pages (decoded multithreaded by tifffile) and never held in full as int64.
    """
    with TiffFile(path) as tif:
        s = tif.series[0]
        if s.ndim != 3 or len(s.pages) != s.shape[0]:  # ponytail: only plain z-stacks are shrunk, rest loads as is
            data = tif.asarray()
            return data[tuple(slice(None, None, scale) if i >= data.ndim - 3 else slice(None)
                              for i in range(data.ndim))]
        blocks = [slice(z, min(z + block * scale, s.shape[0]), scale) for z in range(0, s.shape[0], block * scale)]

        def read(key):  # a single page comes back 2D
            return tif.asarray(key=key, series=0).reshape(-1, *s.shape[1:])[:, ::scale, ::scale]

        dtype = s.dtype
        if np.issubdtype(dtype, np.integer):
            lo = hi = 0
            for key in blocks:
                b = read(key)
                lo, hi = min(lo, b.min()), max(hi, b.max())
            dtype = np.result_type(np.min_scalar_type(lo), np.min_scalar_type(hi))
        out = np.empty(tuple(len(range(0, n, scale)) for n in s.shape), dtype)
        for j, key in enumerate(blocks):
            out[j * block:(j + 1) * block] = read(key)
    return out


def show_set(path, label_files, scale=1, upsample=False, segment=False, args=None):
    """Load one file plus its (name, path) label files and show them in napari; blocks until the window closes."""
    lazy = getattr(args, "lazy", False)
    read = _as_lazy if lazy else _read
    # files stay open until the viewer closes, lazy layers read from them while browsing
    with ExitStack() as files:
        f = files.enter_context(open_file(path, mode="r"))
        if "voxel_size" in f.attrs:
            voxel_size = tuple(float(x) for x in f.attrs["voxel_size"])
        elif "raw" in f and "voxel_size" in f["raw"].attrs:
            voxel_size = tuple(float(x) for x in f["raw"].attrs["voxel_size"])
        else:
            voxel_size = None
        data = {}
        for label_name, label_path in label_files:
            print("Loading label data from", label_path)
            if label_path.endswith(".tif"):
                data[label_name] = _read_tif_labels(label_path, scale)
            else:  # h5 / zarr: load every dataset, prefixed with the label folder name
                lf = files.enter_context(open_file(label_path, mode="r"))
                extract_data(lf, data, prefix=label_name, scale=scale, lazy=lazy)
            # import skimage as ski
            # data["label"] = ski.morphology.remove_small_objects(data["label"], min_size=1000)
        if not label_files:
            print("No specific label path loaded.")
        if ".mrc" in path or ".rec" in path:
            data["raw"] = read(f["data"], scale)
        elif ".tif" in path:
            data["label"] = read(f[""], scale)  # tif has no keys
        else:
            print(f.keys())
            for key in f.keys():
                if isinstance(f[key], (zarr.Group, h5py.Group, z5py.Group)):
                    print(f"Loading group: {key}")
                    extract_data(f[key], data, scale=scale, prefix=key, lazy=lazy)
                    continue
                data[key] = read(f[key], scale)
        for key in ["label"] + [name for name, _ in label_files]:
            if key in data and "raw" in data and data[key].shape != data["raw"].shape:
                print(f"Resizing {key} to match raw data shape")
                data[key] = resize(
                    data[key], data["raw"].shape, preserve_range=True, order=0,
                    anti_aliasing=False).astype(data[key].dtype)
        if upsample:
            del data["pred"]
            del data["raw"]

            for key in data.keys():
                data[key] = upsample_data(data[key], upsample)
        # breakpoint()
        if "raw_mitos_combined" in data.keys() and data["raw_mitos_combined"].ndim == 4:
            for dim in range(data["raw_mitos_combined"].shape[0]):
                if dim == 0:
                    data[f"raw_{dim}"] = data["raw_mitos_combined"][dim]
                if dim == 1:
                    data[f"mitos_{dim}"] = data["raw_mitos_combined"][dim].astype(np.uint8)
            del data["raw_mitos_combined"]
        if segment:
            # get foreground and boundary
            new_seg = _segment(data["pred/foreground"], data["pred/boundary"])
            data["new_seg"] = new_seg
        visualize_data(data, name=os.path.basename(path), args=args, voxel_size=voxel_size)


def main(root_path: str, ext: str = None, scale: int = 1, upsample: bool = False,
         root_label_path: str = None, segment: bool = False,
         offset_z: int = None, args=None):
    if ext is None:
        if os.path.isfile(root_path):
            ext = os.path.splitext(root_path)[1]
            paths = get_file_paths(root_path, ext)
        else:
            print("Loading h5, n5 and zarr files")
            paths = get_file_paths(root_path, ".h5")
            paths.extend(get_file_paths(root_path, ".n5"))
            paths.extend(get_file_paths(root_path, ".zarr"))
            paths.extend(get_file_paths(root_path, ".mrc"))
            paths.extend(get_file_paths(root_path, ".rec"))
    else:
        paths = get_file_paths(root_path, ext)
    if isinstance(root_label_path, str):
        root_label_path = [root_label_path]
    # one layer per label root, named after the root folder (e.g. "mitochondria2", "cristae5")
    label_path_lists = [(os.path.basename(os.path.normpath(r)),
                         [p for e in (".tif", ".h5", ".zarr") for p in get_file_paths(r, e)])
                        for r in (root_label_path or [])]
    print("Found files:", len(paths))
    # not at module level: it pulls in torch, which every spawned viewer process would re-import (~3.5 s)
    import synapse.util as util
    # ponytail: napari 0.6 never frees a viewer closed via its X button, so each set runs in its own process and
    # the OS reclaims the RAM; costs ~5 s of imports per set, a single viewer with a 'next' key would avoid that
    ctx = multiprocessing.get_context("spawn")
    proceed = False
    for path in tqdm(paths):
        if args is not None and not proceed:
            if args.start_pattern is not None:
                if args.start_pattern not in path:
                    continue
                else:
                    proceed = True
        print("\n", path)
        label_files = [(name, util.find_label_file(path, lps) if len(lps) > 1 else lps[0])
                       for name, lps in label_path_lists if lps]
        label_files = [(name, lp) for name, lp in label_files if lp is not None]
        p = ctx.Process(target=show_set, args=(path, label_files, scale, upsample, segment, args))
        p.start()
        p.join()
        if p.exitcode:
            print(f"Viewer for {path} exited with code {p.exitcode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str)
    parser.add_argument("--ext", "-e", type=str, default=None)
    parser.add_argument("--scale", "-s", type=int, default=1)
    parser.add_argument("--upsample", "-u", type=int, default=None)
    parser.add_argument("--label_path", "-lp", type=str, nargs="+", default=None,
                        help="One or more label folders; layers are prefixed with the folder name")
    parser.add_argument("--segment", "-seg", default=False, action="store_true")
    parser.add_argument("--lazy", action="store_true",
                        help="Open instantly and read slices on demand (slower scrolling, avoid 3D view); "
                             "label resizing, --upsample and --segment still load their data")
    parser.add_argument("--offset_z", "-o", type=int, default=None, help="Offset in z direction")
    parser.add_argument("--start_pattern", "-sp", type=str, default=None, help="Pattern to start from")
    parser.add_argument(
        "--voxel_size", "-vs",
        type=lambda x: tuple(map(float, x.split(','))) if ',' in x else (float(x),) * 3,  # Always return a tuple
        default=None,
        help="Voxel size in nm, either a single float (e.g., 12) or a tuple (e.g., 12,12,12)"
    )
    args = parser.parse_args()
    path = args.path
    ext = args.ext
    scale = args.scale
    upsample = args.upsample
    label_path = args.label_path
    main(path, ext, scale, upsample, label_path, segment=args.segment, offset_z=args.offset_z, args=args)
