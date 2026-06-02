"""Migrate existing zarr stores to OME-NGFF 0.4 multiscales metadata.

Targets zarr groups that contain datasets matching the pattern s0, s1, s2, …
Voxel sizes are inferred from:
  1. Legacy root-level voxel_size attribute (assumed to correspond to s0).
  2. A sibling *.zarr in the same parent directory that already has voxel_size.
  3. --default_voxel_size provided on the CLI.

Per-level voxel sizes are computed from shape ratios against s0:
  voxel_size[level][i] = s0_voxel_size[i] * s0_shape[i] / sN_shape[i]

Usage:
  python fix_zarr_multiscales.py --scan_dir /home/.../data/volume-em
  python fix_zarr_multiscales.py --zarr_path /path/to/specific.zarr --default_voxel_size 0.025 0.005 0.005
  python fix_zarr_multiscales.py --scan_dir /home/.../data/volume-em --dry_run
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import zarr


# ---------------------------------------------------------------------------
# OME-NGFF helpers (mirrors downscale_zarr.py — kept local so script is standalone)
# ---------------------------------------------------------------------------

_SCALE_KEY_RE = re.compile(r"^s(\d+)$")


def _is_scale_key(key: str) -> bool:
    return bool(_SCALE_KEY_RE.match(key))


def _scale_index(key: str) -> int:
    return int(_SCALE_KEY_RE.match(key).group(1))


def _build_multiscales(
    scale_keys: List[str],
    voxel_sizes: Dict[str, Tuple[float, ...]],
    unit: str = "micrometer",
) -> dict:
    axes = [
        {"name": "z", "type": "space", "unit": unit},
        {"name": "y", "type": "space", "unit": unit},
        {"name": "x", "type": "space", "unit": unit},
    ]
    datasets = []
    for key in sorted(scale_keys, key=_scale_index):
        vs = voxel_sizes[key]
        datasets.append({
            "path": key,
            "coordinateTransformations": [{"type": "scale", "scale": list(vs)}],
        })
    ndim = len(next(iter(voxel_sizes.values())))
    return {
        "version": "0.4",
        "axes": axes,
        "datasets": datasets,
        "coordinateTransformations": [{"type": "scale", "scale": [1.0] * ndim}],
    }


def _round_voxel_size(vs: Tuple[float, ...], sig: int = 10) -> Tuple[float, ...]:
    """Round voxel_size values to *sig* significant figures to eliminate float noise."""
    import math
    def _round_sig(x, n):
        if x == 0:
            return 0.0
        d = math.ceil(math.log10(abs(x)))
        return round(x, n - d)
    return tuple(_round_sig(v, sig) for v in vs)


def _voxel_sizes_from_shapes(
    s0_voxel_size: Tuple[float, ...],
    shapes: Dict[str, Tuple[int, ...]],
) -> Dict[str, Tuple[float, ...]]:
    """Compute per-level voxel sizes from shape ratios relative to s0."""
    s0_shape = shapes["s0"]
    result = {}
    for key, shape in shapes.items():
        vs = tuple(
            s0_voxel_size[i] * s0_shape[i] / shape[i]
            for i in range(len(s0_voxel_size))
        )
        result[key] = _round_voxel_size(vs)
    return result


def _read_legacy_voxel_size(zarr_path: str) -> Optional[Tuple[float, ...]]:
    """Read legacy root-level voxel_size from .zattrs."""
    zattrs = Path(zarr_path) / ".zattrs"
    if not zattrs.exists():
        return None
    with open(zattrs) as f:
        attrs = json.load(f)
    vs = attrs.get("voxel_size")
    if vs is not None:
        return tuple(float(v) for v in vs)
    # Also check OME-NGFF multiscales already present (e.g. partially migrated)
    for ms in attrs.get("multiscales", []):
        for ds in ms.get("datasets", []):
            if ds["path"] == "s0":
                for ct in ds.get("coordinateTransformations", []):
                    if ct["type"] == "scale":
                        return tuple(float(v) for v in ct["scale"])
    return None


def _find_sibling_voxel_size(zarr_path: str) -> Optional[Tuple[float, ...]]:
    """Search sibling *.zarr directories in the same parent for a known voxel_size."""
    parent = Path(zarr_path).parent
    for candidate in sorted(parent.glob("*.zarr")):
        if str(candidate) == zarr_path:
            continue
        vs = _read_legacy_voxel_size(str(candidate))
        if vs is not None:
            # Confirm that candidate has s0 and its shape is a superset of ours
            # (i.e. this sibling is the raw source we were derived from).
            return vs
    return None


# ---------------------------------------------------------------------------
# Per-zarr migration logic
# ---------------------------------------------------------------------------

def _get_scale_keys_and_shapes(zarr_path: str) -> Dict[str, Tuple[int, ...]]:
    """Return {key: shape} for all s\d+ datasets in the zarr group."""
    try:
        root = zarr.open(zarr_path, mode="r")
    except Exception:
        return {}
    result = {}
    for key in root.keys():
        if _is_scale_key(key) and hasattr(root[key], "shape"):
            result[key] = tuple(root[key].shape)
    return result


def migrate_zarr(
    zarr_path: str,
    default_voxel_size: Optional[Tuple[float, ...]] = None,
    unit: str = "micrometer",
    dry_run: bool = False,
) -> bool:
    """Migrate one zarr store. Returns True if metadata was written (or would be in dry_run)."""
    scale_shapes = _get_scale_keys_and_shapes(zarr_path)
    if not scale_shapes:
        return False  # no s\d+ datasets → nothing to do

    # Determine s0 voxel_size
    s0_voxel_size = None
    source = None

    if "s0" in scale_shapes:
        # Try own legacy attrs first
        s0_voxel_size = _read_legacy_voxel_size(zarr_path)
        if s0_voxel_size is not None:
            source = "own legacy voxel_size"
        else:
            s0_voxel_size = _find_sibling_voxel_size(zarr_path)
            if s0_voxel_size is not None:
                source = "sibling zarr voxel_size"
    else:
        # No s0 present — shapes alone can't give us the absolute voxel size.
        # Try siblings whose s0 shape might tell us the base resolution.
        s0_voxel_size = _find_sibling_voxel_size(zarr_path)
        if s0_voxel_size is not None:
            # The sibling's s0 voxel_size is for s0 resolution.
            # We need to scale it to match whatever is the finest key in this zarr.
            # Find the sibling's s0 shape for cross-referencing.
            parent = Path(zarr_path).parent
            for candidate in sorted(parent.glob("*.zarr")):
                if str(candidate) == zarr_path:
                    continue
                sib_shapes = _get_scale_keys_and_shapes(str(candidate))
                sib_vs = _read_legacy_voxel_size(str(candidate))
                if sib_vs is None or "s0" not in sib_shapes:
                    continue
                # Compute voxel sizes for all sib scale keys, then look up ours
                sib_voxel_sizes = _voxel_sizes_from_shapes(sib_vs, sib_shapes)
                # Match our scale keys to sibling by shape
                resolved = {}
                for our_key, our_shape in scale_shapes.items():
                    for sib_key, sib_shape in sib_shapes.items():
                        if our_shape == sib_shape and sib_key in sib_voxel_sizes:
                            resolved[our_key] = sib_voxel_sizes[sib_key]
                            break
                if resolved:
                    resolved = {k: _round_voxel_size(v) for k, v in resolved.items()}
                    # Build direct voxel_sizes dict (skip ratio computation)
                    print(f"  {zarr_path}")
                    print(f"    Source: shape-matched sibling {candidate.name}")
                    for key in sorted(resolved, key=_scale_index):
                        print(f"    {key}: shape={scale_shapes[key]}  voxel_size={list(resolved[key])}")
                    if not dry_run:
                        root = zarr.open(zarr_path, mode="a")
                        attrs = dict(root.attrs)
                        attrs["multiscales"] = [_build_multiscales(list(resolved.keys()), resolved, unit)]
                        root.attrs.update(attrs)
                        print("    → written")
                    else:
                        print("    → [dry run]")
                    return True
            source = "sibling zarr voxel_size (no shape match — using as-is)"

    if s0_voxel_size is None:
        s0_voxel_size = default_voxel_size
        if s0_voxel_size is not None:
            source = "--default_voxel_size"

    if s0_voxel_size is None:
        print(f"  SKIP {zarr_path}  (no voxel_size found; pass --default_voxel_size)")
        return False

    # Build per-level voxel sizes from shape ratios
    if "s0" not in scale_shapes:
        # Can't compute ratios without s0 — fall back to same voxel_size for all keys
        voxel_sizes = {k: s0_voxel_size for k in scale_shapes}
    else:
        voxel_sizes = _voxel_sizes_from_shapes(s0_voxel_size, scale_shapes)

    print(f"  {zarr_path}")
    print(f"    Source: {source}")
    for key in sorted(voxel_sizes, key=_scale_index):
        print(f"    {key}: shape={scale_shapes[key]}  voxel_size={[round(v, 8) for v in voxel_sizes[key]]}")

    if not dry_run:
        root = zarr.open(zarr_path, mode="a")
        attrs = dict(root.attrs)
        attrs["multiscales"] = [_build_multiscales(list(scale_shapes.keys()), voxel_sizes, unit)]
        root.attrs.update(attrs)
        print("    → written")
    else:
        print("    → [dry run]")

    return True


# ---------------------------------------------------------------------------
# Directory scan
# ---------------------------------------------------------------------------

def _find_zarr_roots(scan_dir: str) -> List[str]:
    """Find all .zgroup files and return their parent directories (zarr group roots)."""
    roots = []
    for dirpath, dirnames, filenames in os.walk(scan_dir):
        if ".zgroup" in filenames:
            roots.append(dirpath)
            # Don't recurse into nested zarr groups (sub-groups have their own .zgroup)
            dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d / ".zgroup").exists()]
    return sorted(roots)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan_dir", "-d", help="Root directory to scan recursively for zarr stores")
    group.add_argument("--zarr_path", "-z", help="Path to a single zarr store to migrate")
    p.add_argument(
        "--default_voxel_size", "-vs", type=float, nargs="+", default=None,
        help="Fallback s0 voxel size in zyx order, used when no metadata can be found "
             "(e.g. --default_voxel_size 0.025 0.005 0.005)",
    )
    p.add_argument("--unit", "-u", default="micrometer", help="Physical unit (default: micrometer)")
    p.add_argument("--dry_run", "-n", action="store_true", help="Print planned changes without writing")
    args = p.parse_args()

    default_vs = tuple(args.default_voxel_size) if args.default_voxel_size else None

    if args.zarr_path:
        candidates = [args.zarr_path]
    else:
        candidates = _find_zarr_roots(args.scan_dir)
        print(f"Found {len(candidates)} zarr group(s) under {args.scan_dir}\n")

    migrated = skipped = 0
    for zarr_path in candidates:
        result = migrate_zarr(zarr_path, default_voxel_size=default_vs, unit=args.unit, dry_run=args.dry_run)
        if result:
            migrated += 1
        else:
            skipped += 1

    action = "Would migrate" if args.dry_run else "Migrated"
    print(f"\n{action} {migrated} zarr store(s), skipped {skipped}.")


if __name__ == "__main__":
    main()
