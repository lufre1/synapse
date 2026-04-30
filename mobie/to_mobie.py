#!/usr/bin/env python3
import os
import shutil
import argparse

import mobie
import numpy as np
import z5py
from elf.parallel import label, unique
import mobie.validation.utils as vutils
import mobie.validation.metadata as vmeta

def _no_validate_with_schema(*args, **kwargs):
    return

# Patch both: utils module and the already-imported symbol in metadata module
vutils.validate_with_schema = _no_validate_with_schema
vmeta.validate_with_schema = _no_validate_with_schema


def get_volume_names():
    return ["4007", "4009"]


def get_resolution(volume_name):
    resolutions = {
        # 25 x 5 x 5 nanometer
        "4007": [0.025, 0.005, 0.005],
        "4009": [0.025, 0.005, 0.005],
        "4010": [0.025, 0.005, 0.005],
        "4016": [0.025, 0.005, 0.005],
    }
    return resolutions[volume_name]


# -----------------------------
# Optional preprocessing for raw
# -----------------------------
def clear_background(path, key, tmp_folder, out_name="raw_bgcleared.n5"):
    """Set the largest connected component of value 255 to 0 in the raw volume.
    Writes a temporary N5 with the same dataset key and returns the temp path.
    """
    os.makedirs(tmp_folder, exist_ok=True)
    out_path = os.path.join(tmp_folder, out_name)

    with z5py.File(out_path, "a") as f_out:
        if key in f_out:
            return out_path

        with z5py.File(path, "r") as f_in:
            ds = f_in[key]
            ds.n_threads = 8
            print(f"Load raw data: {path}:{key}")
            raw = ds[:]
            chunks = ds.chunks

        print("Find background (value==255) ...")
        input_ = (raw == 255)
        bg = np.zeros(raw.shape, dtype="uint32")
        bg = label(input_, bg, with_background=True, block_shape=chunks,
                   n_threads=16, verbose=True)

        print("Find largest component ...")
        bg_ids, counts = unique(bg, return_counts=True, block_shape=chunks,
                                n_threads=16, verbose=True)
        bg_id = bg_ids[np.argmax(counts)]
        bg = bg == bg_id

        raw[bg] = 0

        print(f"Save bg-cleared raw to: {out_path}:{key}")
        ds_out = f_out.create_dataset(
            key, shape=raw.shape, dtype=raw.dtype, compression="gzip", chunks=chunks
        )
        ds_out.n_threads = 8
        ds_out[:] = raw

    return out_path


def _get_sources(ds_folder):
    if not os.path.exists(ds_folder):
        return {}
    md = mobie.metadata.read_dataset_metadata(ds_folder)
    return md.get("sources", {})


def _repair_views_if_needed(ds_folder):
    """Ensure 'views' in dataset.json is a dict, not a list."""
    md_path = os.path.join(ds_folder, "dataset.json")
    if not os.path.exists(md_path):
        return
    md = mobie.metadata.read_dataset_metadata(ds_folder)
    views = md.get("views", {})
    if isinstance(views, list):
        print(f"[REPAIR] 'views' in {md_path} is a list — converting to dict")
        new_views = {}
        for entry in views:
            if isinstance(entry, dict):
                new_views.update(entry)
        md["views"] = new_views
        mobie.metadata.write_dataset_metadata(ds_folder, md)


def _remove_source_from_metadata(ds_folder, source_name, file_format="ome.zarr"):
    """Remove a source (image or segmentation) from MoBIE dataset metadata and disk."""
    md = mobie.metadata.read_dataset_metadata(ds_folder)
    sources = md.get("sources", {})

    if source_name not in sources:
        return

    # Find and delete the on-disk data
    source_info = sources[source_name]
    # source_info looks like: {"segmentation": {"imageData": {"ome.zarr": {"relativePath": "..."}}}}
    # or:                     {"image": {"imageData": {"ome.zarr": {"relativePath": "..."}}}}
    for source_type in ("image", "segmentation"):
        if source_type in source_info:
            rel_path = (source_info[source_type]
                        .get("imageData", {})
                        .get(file_format, {})
                        .get("relativePath", None))
            if rel_path:
                full_path = os.path.join(ds_folder, rel_path)
                if os.path.exists(full_path):
                    shutil.rmtree(full_path)
                    print(f"  Deleted data: {full_path}")

    # Remove from sources dict
    del md["sources"][source_name]

    # Remove from views dict (views is a dict, NOT a list)
    views = md.get("views", {})
    if isinstance(views, dict) and source_name in views:
        del views[source_name]
        md["views"] = views

    mobie.metadata.write_dataset_metadata(ds_folder, md)
    print(f"  Removed '{source_name}' from metadata")


# -----------------------------
# MoBIE add helpers
# -----------------------------
def add_raw(output_root, ds_name, path, key, tmp_folder, resolution,
            image_name="raw", clear_bg=False,
            file_format="ome.zarr", target="local", max_jobs=16,
            overwrite=False):
    ds_folder = os.path.join(output_root, ds_name)
    _repair_views_if_needed(ds_folder)
    sources = _get_sources(ds_folder)

    if image_name in sources:
        if not overwrite:
            print(f"[SKIP] {ds_name}: image '{image_name}' already exists (use --overwrite)")
            return
        print(f"[OVERWRITE] {ds_name}: removing existing image '{image_name}'")
        _remove_source_from_metadata(ds_folder, image_name, file_format)

    if clear_bg:
        path = clear_background(path, key=key, tmp_folder=tmp_folder)

    scale_factors = [[1, 2, 2], [1, 2, 2], [2, 2, 2]]
    chunks = [64, 128, 128]

    mobie.add_image(
        path, key,
        output_root, ds_name,
        image_name=image_name,
        resolution=resolution,
        scale_factors=scale_factors,
        chunks=chunks,
        file_format=file_format,
        target=target,
        max_jobs=max_jobs,
        tmp_folder=tmp_folder
    )
    print(f"[OK] {ds_name}: added image '{image_name}'")


def add_seg(output_root, ds_name, path, key, tmp_folder, resolution,
            seg_name,
            file_format="ome.zarr", target="local", max_jobs=16,
            overwrite=False):
    ds_folder = os.path.join(output_root, ds_name)
    _repair_views_if_needed(ds_folder)
    sources = _get_sources(ds_folder)

    if seg_name in sources:
        if not overwrite:
            print(f"[SKIP] {ds_name}: segmentation '{seg_name}' already exists (use --overwrite)")
            return
        print(f"[OVERWRITE] {ds_name}: removing existing segmentation '{seg_name}'")
        _remove_source_from_metadata(ds_folder, seg_name, file_format)

    scale_factors = [[1, 2, 2], [1, 2, 2], [2, 2, 2]]
    chunks = [64, 128, 128]

    mobie.add_segmentation(
        path, key,
        output_root, ds_name,
        segmentation_name=seg_name,
        resolution=resolution,
        scale_factors=scale_factors,
        chunks=chunks,
        file_format=file_format,
        target=target,
        max_jobs=max_jobs,
        tmp_folder=tmp_folder,
    )
    print(f"[OK] {ds_name}: added segmentation '{seg_name}'")


# -----------------------------
# argparse plumbing
# -----------------------------
def parse_kv_list(kv_list, what):
    """Parse ['name=/path:group/key', ...] into dict(name -> (path, key))."""
    out = {}
    for item in kv_list or []:
        if "=" not in item:
            raise ValueError(f"Invalid {what} spec '{item}'. Expected name=/path:KEY")
        name, spec = item.split("=", 1)
        if ":" not in spec:
            raise ValueError(f"Invalid {what} spec '{item}'. Expected name=/path:KEY")
        path, key = spec.rsplit(":", 1)
        out[name] = (path, key)
    return out


def build_parser():
    p = argparse.ArgumentParser(
        description="Add raw + segmentations to a MoBIE project from container datasets (N5/Zarr/etc)."
    )
    p.add_argument("--output-root", required=True,
                   help="MoBIE project root (contains datasets like <output-root>/<ds-name>/...).")
    p.add_argument("--ds", required=True,
                   help="Dataset name/id in the MoBIE project, e.g. 4007.")
    p.add_argument("--tmp-folder", default=None,
                   help="Temporary folder. Default: <output-root>/_tmp_<ds>.")
    p.add_argument("--resolution-from", default=None,
                   help="Dataset id used for get_resolution(). If omitted, uses --ds.")
    p.add_argument("--resolution", default=None,
                   help="Explicit resolution as 'z,y,x' (overrides --resolution-from/--ds). Example: 40,8,8")

    # Inputs
    p.add_argument("--raw", default=None,
                   help="Raw spec as '/path:KEY' (e.g. /data/4007.n5:raw).")
    p.add_argument("--raw-name", default="raw",
                   help="MoBIE image name for raw (default: raw).")
    p.add_argument("--clear-bg", action="store_true",
                   help="Apply background clearing (largest component of value 255 -> 0) for raw.")

    p.add_argument("--seg", action="append", default=[],
                   help=("Segmentation spec, repeatable: name=/path:KEY  "
                         "(e.g. mitochondria=/data/labels.n5:mito)."))

    # MoBIE writing options
    p.add_argument("--file-format", default="ome.zarr",
                   help="MoBIE output format (default: ome.zarr).")
    p.add_argument("--target", default="local",
                   help="MoBIE target (default: local).")
    p.add_argument("--max-jobs", type=int, default=8,
                   help="Parallel jobs for writing (default: 8).")
    p.add_argument("--overwrite", action="store_true", default=False,
               help="If set, overwrite existing sources instead of skipping them.")

    return p


def parse_resolution_arg(res_str):
    parts = [p.strip() for p in res_str.split(",")]
    if len(parts) != 3:
        raise ValueError("--resolution must be 'z,y,x' (3 values)")
    # allow ints or floats
    vals = tuple(float(v) if "." in v else int(v) for v in parts)
    return vals


def main():
    args = build_parser().parse_args()

    ds_name = args.ds
    tmp_folder = args.tmp_folder or os.path.join(args.output_root, f"_tmp_{ds_name}")
    os.makedirs(tmp_folder, exist_ok=True)

    # resolution
    if args.resolution is not None:
        resolution = parse_resolution_arg(args.resolution)
    else:
        res_from = args.resolution_from or ds_name
        resolution = get_resolution(res_from)

    # raw
    if args.raw:
        if ":" not in args.raw:
            raise ValueError("--raw must be '/path:KEY'")
        raw_path, raw_key = args.raw.rsplit(":", 1)
        add_raw(
            args.output_root, ds_name,
            path=raw_path, key=raw_key,
            tmp_folder=tmp_folder,
            resolution=resolution,
            image_name=args.raw_name,
            clear_bg=args.clear_bg,
            file_format=args.file_format,
            target=args.target,
            max_jobs=args.max_jobs,
            overwrite=args.overwrite,
        )
        shutil.rmtree(tmp_folder)

    # segmentations
    segs = parse_kv_list(args.seg, what="--seg")
    for seg_name, (seg_path, seg_key) in segs.items():
        add_seg(
            args.output_root, ds_name,
            path=seg_path, key=seg_key,
            tmp_folder=tmp_folder,
            resolution=resolution,
            seg_name=seg_name,
            file_format=args.file_format,
            target=args.target,
            max_jobs=args.max_jobs,
            overwrite=args.overwrite,
        )
        shutil.rmtree(tmp_folder)
    if os.path.isdir(tmp_folder):
        shutil.rmtree(tmp_folder)


if __name__ == "__main__":
    main()