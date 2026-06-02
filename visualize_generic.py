"""visualize_generic.py — auto-detecting multi-format 3D visualizer for EM data.

Supports H5, Zarr, N5, MRC/REC, TIFF.
- Recursively discovers all datasets inside a file.
- Skips multiscale siblings: only the highest-resolution level is loaded.
- For TIFF and MRC/REC: searches sibling files/directories for annotations
  (TIFF segmentations or IMOD .mod/.imod files).
"""

import os
import re
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import zarr
import z5py
import numpy as np
import napari
from elf.io import open_file
from tifffile import imread


# ── Keyword sets ────────────────────────────────────────────────────────────
MULTISCALE_RE = re.compile(r"^s?\d+$")          # s0, s1, 0, 1, 2, …
ANNOTATION_KW = {"seg", "label", "annot", "mask", "gt"}
ANNOTATION_SUFFIXES = ["_seg", "_labels", "_label", "_annotation", "_mask", "_gt"]
ANNOTATION_EXTS = [".tif", ".tiff", ".mod", ".imod"]
ADDITIVE_KW = ("pred", "dist", "fore", "bound")
LABEL_KW = ("label", "seg", "mask", "gt", "annot")

# Tried when a recursive walk yields nothing
COMMON_KEYS = [
    "raw", "data", "image",
    "labels", "seg", "segmentation",
    "labels/mitochondria", "labels/cristae", "labels/mitos",
    "prediction", "pred", "foreground", "boundary",
]


# ── Format detection ─────────────────────────────────────────────────────────
def detect_format(path: str) -> str:
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".h5", ".hdf5"):
        return "h5"
    if ext in (".zarr",) or (p.is_dir() and ((p / ".zgroup").exists() or (p / ".zarray").exists())):
        return "zarr"
    if ext == ".n5" or (p.is_dir() and (p / "attributes.json").exists() and not (p / ".zgroup").exists()):
        return "n5"
    if ext in (".mrc", ".rec"):
        return "mrc"
    if ext in (".tif", ".tiff"):
        return "tif"
    return "unknown"


# ── Multiscale helpers ───────────────────────────────────────────────────────
def _best_scale_key(group) -> Optional[str]:
    """
    Return the highest-resolution key if this group is a multiscale pyramid.
    Returns None for plain groups.
    """
    try:
        attrs = dict(group.attrs)
    except Exception:
        attrs = {}

    # OME-Zarr / OME-NGFF: 'multiscales' attribute lists dataset paths
    if "multiscales" in attrs:
        try:
            datasets = attrs["multiscales"][0]["datasets"]
            path = datasets[0]["path"]
            if path in group:
                return path
        except (KeyError, IndexError, TypeError):
            pass

    # N5-style or plain pyramid: ALL children match s?\d+
    keys = list(group.keys())
    scale_keys = [k for k in keys if MULTISCALE_RE.match(k)]
    if len(scale_keys) >= 2 and len(scale_keys) == len(keys):
        for candidate in ("s0", "0"):
            if candidate in scale_keys:
                return candidate
        return sorted(scale_keys, key=lambda k: int(re.sub(r"[^0-9]", "", k)))[0]

    return None


def _load_array(item, scale: int = 1) -> np.ndarray:
    ndim = item.ndim
    slicing = tuple(
        slice(None, None, scale) if i >= ndim - 3 else slice(None)
        for i in range(ndim)
    )
    return item[slicing] if scale > 1 else item[:]


def _walk_group(group, data: Dict[str, Any], prefix: str = "", scale: int = 1):
    """
    Recursively collect datasets from a group.
    When all children are scale levels, only the highest-resolution one is kept.
    """
    best = _best_scale_key(group)
    if best is not None:
        item = group[best]
        key_out = prefix or "raw"
        if isinstance(item, (zarr.Group, h5py.Group, z5py.Group)):
            _walk_group(item, data, prefix=key_out, scale=scale)
        else:
            data[key_out] = _load_array(item, scale)
        return

    for k in group.keys():
        item = group[k]
        full_key = f"{prefix}/{k}" if prefix else k
        if isinstance(item, (zarr.Group, h5py.Group, z5py.Group)):
            _walk_group(item, data, prefix=full_key, scale=scale)
        else:
            data[full_key] = _load_array(item, scale)


# ── File loaders ─────────────────────────────────────────────────────────────
def load_structured(path: str, scale: int = 1) -> Tuple[Dict, Optional[tuple]]:
    """Load H5 / Zarr / N5."""
    data: Dict[str, Any] = {}
    voxel_size = None
    with open_file(path, mode="r") as f:
        attrs = dict(f.attrs)
        if "voxel_size" in attrs:
            voxel_size = tuple(float(x) for x in attrs["voxel_size"])
        elif "raw" in f and "voxel_size" in dict(getattr(f["raw"], "attrs", {})):
            voxel_size = tuple(float(x) for x in f["raw"].attrs["voxel_size"])

        _walk_group(f, data, scale=scale)

        # Fallback: try well-known keys if recursive walk found nothing
        if not data:
            print("  [warn] Recursive walk found nothing, trying common keys …")
            for key in COMMON_KEYS:
                try:
                    if key in f:
                        data[key] = _load_array(f[key], scale)
                except Exception:
                    pass

    return data, voxel_size


def load_mrc(path: str, scale: int = 1) -> Tuple[Dict, Optional[tuple]]:
    data: Dict[str, Any] = {}
    with open_file(path, mode="r") as f:
        data["raw"] = _load_array(f["data"], scale)
    return data, None


def load_tif_file(path: str, scale: int = 1) -> Tuple[Dict, Optional[tuple]]:
    arr = imread(path)
    if scale > 1:
        ndim = arr.ndim
        slicing = tuple(
            slice(None, None, scale) if i >= ndim - 3 else slice(None)
            for i in range(ndim)
        )
        arr = arr[slicing]
    # Classify by filename: annotation keywords → labels, otherwise raw
    stem_lower = Path(path).stem.lower()
    key = "labels" if any(kw in stem_lower for kw in ANNOTATION_KW) else "raw"
    return {key: arr}, None


# ── Annotation discovery ─────────────────────────────────────────────────────
def find_annotation_files(path: str) -> List[str]:
    """Search for companion annotation files next to a TIF or MRC file."""
    p = Path(path)
    stem, directory = p.stem, p.parent
    found: List[str] = []

    def _add(candidate: Path):
        s = str(candidate)
        if s not in found:
            found.append(s)

    # 1. Same stem + annotation suffix
    for suffix in ANNOTATION_SUFFIXES:
        for ext in ANNOTATION_EXTS:
            c = directory / f"{stem}{suffix}{ext}"
            if c.exists():
                _add(c)

    # 2. Any sibling whose name contains annotation keywords
    for ext in ANNOTATION_EXTS:
        for f in directory.glob(f"*{ext}"):
            if str(f) == path:
                continue
            if any(kw in f.stem.lower() for kw in ANNOTATION_KW):
                _add(f)

    # 3. Sibling sub-directories commonly used for annotations
    for sub in ("labels", "seg", "segmentation", "annotations", "gt"):
        sub_dir = directory / sub
        if sub_dir.is_dir():
            for ext in (".tif", ".tiff"):
                for f in sub_dir.glob(f"*{ext}"):
                    _add(f)

    return found


def load_annotation(ann_path: str, scale: int = 1) -> Optional[np.ndarray]:
    ext = Path(ann_path).suffix.lower()
    if ext in (".tif", ".tiff"):
        arr = imread(ann_path)
        if scale > 1:
            ndim = arr.ndim
            slicing = tuple(
                slice(None, None, scale) if i >= ndim - 3 else slice(None)
                for i in range(ndim)
            )
            arr = arr[slicing]
        return arr
    if ext in (".mod", ".imod"):
        return _load_mod(ann_path)
    return None


def _load_mod(mod_path: str) -> Optional[np.ndarray]:
    """Load an IMOD .mod file by converting to TIFF via imod2tif."""
    tmp = tempfile.mktemp(suffix=".tif")
    try:
        result = subprocess.run(
            ["imod2tif", mod_path, tmp],
            capture_output=True,
            timeout=30,
        )
        if result.returncode == 0 and os.path.exists(tmp):
            arr = imread(tmp)
            os.remove(tmp)
            return arr
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    print(f"  [warn] Cannot load {mod_path}: imod2tif not available.")
    return None


# ── Napari visualizer ────────────────────────────────────────────────────────
def visualize(data: Dict[str, Any], name: str = "", voxel_size=None):
    viewer = napari.Viewer()
    viewer.title = name

    for key, arr in data.items():
        kl = key.lower()
        if any(kw in kl for kw in LABEL_KW):
            viewer.add_labels(arr, name=key, scale=voxel_size)
        elif any(kw in kl for kw in ADDITIVE_KW):
            viewer.add_image(arr, name=key, blending="additive", scale=voxel_size)
        else:
            viewer.add_image(arr, name=key, scale=voxel_size)

    # Raw data goes to the bottom of the layer stack
    raw_layer = next((la for la in viewer.layers if "raw" in la.name.lower()), None)
    if raw_layer:
        viewer.layers.remove(raw_layer)
        viewer.layers.insert(0, raw_layer)

    napari.run()


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Auto-detecting 3D visualizer for EM data (H5, Zarr, N5, MRC, TIFF)."
    )
    ap.add_argument("path", help="File or directory to visualize")
    ap.add_argument("--scale", "-s", type=int, default=1,
                    help="Spatial downsampling factor (applied to last 3 axes)")
    ap.add_argument("--annotation", "-a", default=None,
                    help="Explicit annotation file path (disables auto-discovery)")
    ap.add_argument("--no_annotations", action="store_true",
                    help="Disable automatic annotation file search for TIFF/MRC")
    ap.add_argument(
        "--voxel_size", "-vs",
        type=lambda x: tuple(map(float, x.split(","))) if "," in x else (float(x),) * 3,
        default=None,
        help="Voxel size in nm — single value or z,y,x triple (e.g. 12 or 14,8,8)",
    )
    args = ap.parse_args()

    path = args.path
    fmt = detect_format(path)
    print(f"Format: {fmt}  →  {path}")

    if fmt in ("h5", "zarr", "n5"):
        data, voxel_size = load_structured(path, args.scale)
    elif fmt == "mrc":
        data, voxel_size = load_mrc(path, args.scale)
    elif fmt == "tif":
        data, voxel_size = load_tif_file(path, args.scale)
    else:
        sys.exit(f"Unsupported format '{fmt}' for path: {path}")

    # Annotation search (only for flat formats that carry no internal labels)
    if fmt in ("tif", "mrc") and not args.no_annotations:
        if args.annotation:
            ann_files = [args.annotation]
        else:
            ann_files = find_annotation_files(path)
            if ann_files:
                print(f"Found annotation files: {ann_files}")

        for ann_path in ann_files:
            print(f"  Loading annotation: {ann_path}")
            arr = load_annotation(ann_path, args.scale)
            if arr is not None:
                data[f"labels/{Path(ann_path).stem}"] = arr

    if not data:
        sys.exit("No data loaded — nothing to visualize.")

    if args.voxel_size is not None:
        voxel_size = args.voxel_size

    print(f"Layers: {list(data.keys())}")
    visualize(data, name=Path(path).name, voxel_size=voxel_size)


if __name__ == "__main__":
    main()
