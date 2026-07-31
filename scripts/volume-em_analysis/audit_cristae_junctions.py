#!/usr/bin/env python
"""Audit cristae -> mito-membrane contact points (crista junctions) in the cristae training data.

Junctions/contact points are NOT annotated. Every cristae training file stores:
  - ``labels/cristae``        : the crista segmentation (mostly binary, sometimes instance-labelled)
  - ``raw_mitos_combined``    : (2, Z, Y, X); channel 0 = raw EM, channel 1 = semantic mito mask
                                {0=bg, 1=mito with cristae annotations, 2=mito without}

The mitochondrial membrane is not segmented; it is *approximated* from the mito mask exactly the way
the synapse-net cristae widget does (``approximate_membrane`` = mito minus its erosion, an N-nm shell).
A "contact point"/junction is a connected component of ``crista & membrane`` (``detect_contact_sites``).

This script reuses those two functions verbatim, so the numbers match the widget. It is read-only
w.r.t. the data: it only writes a per-file CSV.
"""
import argparse
import os
import re
import sys
import time
from glob import glob

import types

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label

# synapse_net.cristae_analysis imports geodesic/mesh symbols from bioimage_cpp at module load time
# (used only by the geodesic-distance metrics). This env's bioimage_cpp lacks them, so stub the
# missing names/submodule before import. approximate_membrane / detect_contact_sites never touch
# these, so the functions we use are the real thing, unchanged.
import bioimage_cpp.distance as _bd  # noqa: E402
if not hasattr(_bd, "geodesic_distances_mesh"):
    _bd.geodesic_distances_mesh = None
for _name in ("filters", "mesh"):
    try:
        __import__(f"bioimage_cpp.{_name}")
    except ModuleNotFoundError:
        sys.modules[f"bioimage_cpp.{_name}"] = types.ModuleType(f"bioimage_cpp.{_name}")
for _mod, _sym in (("filters", "structure_tensor_eigenvalues"), ("mesh", "marching_cubes")):
    _m = sys.modules[f"bioimage_cpp.{_mod}"]
    if not hasattr(_m, _sym):
        setattr(_m, _sym, None)

from synapse_net.cristae_analysis import approximate_membrane, detect_contact_sites  # noqa: E402

# Discovery roots + filters, replicated verbatim from training/cristae/train_cristae.py:353-368
ROOTS = [
    "/scratch-grete/projects/nim00007/data/mitochondria/cooper/raw_mito_combined_s2",
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann",
]
SUBSTRING = "_combined.h5"
EXCLUDE_STRINGS = [
    "Otof_AVCN03_429C_WT_M.Stim_G3_1_model_combined",  # raw data strange
    "WT20_eb8_AZ1_model_combined",                     # poor cristae annotations
    "WT22_eb8_model_combined",                         # poor cristae annotations
]

CRISTA_KEY = "labels/cristae"
RAW_KEY = "raw_mitos_combined"
MITO_CHANNEL = 1          # channel of raw_mitos_combined holding the semantic mito mask
ANNOTATED_STATE = 1       # cristae are only annotated inside state-1 mitochondria
CC_STRUCT = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity, matches detect_contact_sites


def discover_files():
    paths = []
    for root in ROOTS:
        paths.extend(glob(os.path.join(root, "**", "*.h5"), recursive=True))
    paths = [p for p in paths if SUBSTRING in p]
    paths = [p for p in paths if not any(e in p for e in EXCLUDE_STRINGS)]
    return sorted(set(paths))


def source_of(path):
    if "cristae_data/wichmann" in path:
        return "wichmann"
    return "cooper"


def genotype_of(path):
    """Heuristic WT / KO / DKO / unspecified from the filename (see CRISTAE_TRAINING_DATA.md)."""
    name = os.path.basename(path)
    if "Otof" in name:
        if "_WT_" in name:
            return "WT"
        if "_KO_" in name:
            return "KO"
        return "unknown"
    if name.startswith("WT"):
        return "WT"
    if name.startswith("KO"):
        return "KO"
    if "37371_O4" in name:   # cooper M13DKO
        return "DKO"
    if "37371_O5" in name:   # cooper CTRL (WT)
        return "WT"
    if "36194" in name or "36859" in name:  # cooper WT
        return "WT"
    if re.match(r"M\d+_(eb|syn)", name):     # wichmann M# lines, genotype not specified in doc
        return "unspecified"
    return "unknown"


def read_voxel_size(f, default_nm):
    """Return (voxel_size_for_analysis, source_str). Data is ~isotropic; store a z/y/x dict."""
    for holder in (f, f.get(RAW_KEY), f.get(CRISTA_KEY)):
        if holder is None:
            continue
        if "voxel_size" in holder.attrs:
            vs = np.atleast_1d(np.asarray(holder.attrs["voxel_size"], dtype=float)).ravel()
            if vs.size == 1:
                v = float(vs[0])
                return {"z": v, "y": v, "x": v}, "attr"
            if vs.size >= 3:  # stored z, y, x
                return {"z": float(vs[0]), "y": float(vs[1]), "x": float(vs[2])}, "attr"
    v = float(default_nm)
    return {"z": v, "y": v, "x": v}, "default"


def analyse_file(path, thicknesses, n_jobs, default_nm):
    """Return a list of per-(file, thickness) result dicts."""
    with h5py.File(path, "r") as f:
        crista = np.asarray(f[CRISTA_KEY][:]) > 0
        mito_state = np.asarray(f[RAW_KEY][MITO_CHANNEL])
        voxel_size, vs_source = read_voxel_size(f, default_nm)
        shape = tuple(int(s) for s in crista.shape)

    mito_annotated = mito_state == ANNOTATED_STATE

    crista_voxels = int(crista.sum())
    mito_voxels = int(mito_annotated.sum())

    # thickness-independent instance counts
    crista_cc, n_crista_inst = cc_label(crista, structure=CC_STRUCT)
    mito_cc, n_mito_inst = cc_label(mito_annotated, structure=CC_STRUCT)

    base = dict(
        path=path,
        file=os.path.basename(path),
        source=source_of(path),
        genotype=genotype_of(path),
        voxel_size_source=vs_source,
        voxel_size_nm=voxel_size["x"],
        shape=str(shape),
        n_voxels=int(np.prod(shape)),
        crista_voxels=crista_voxels,
        mito_state1_voxels=mito_voxels,
        n_cristae_instances=int(n_crista_inst),
        n_mito_instances=int(n_mito_inst),
        no_cristae=crista_voxels == 0,
        no_state1_mito=mito_voxels == 0,
    )

    rows = []
    for thickness in thicknesses:
        row = dict(base, thickness_nm=float(thickness))
        if crista_voxels == 0 or mito_voxels == 0:
            row.update(
                membrane_voxels=0, contact_voxel_count=0, crista_junction_count=0,
                contact_volume_nm3=0.0, n_cristae_touching=0, n_mito_with_junction=0,
                frac_crista_voxels_in_contact=np.nan, frac_cristae_touching=np.nan,
            )
            rows.append(row)
            continue

        membrane = approximate_membrane(
            mito_annotated, voxel_size, membrane_thickness_nm=float(thickness),
            n_jobs=n_jobs, membrane_mode="slice_2d", return_lumen=False,
        )
        contact_labels, summary = detect_contact_sites(crista, membrane, voxel_size)

        touching_ids = np.unique(crista_cc[membrane])
        n_cristae_touching = int((touching_ids > 0).sum())
        mito_hit_ids = np.unique(mito_cc[contact_labels > 0])
        n_mito_with_junction = int((mito_hit_ids > 0).sum())

        row.update(
            membrane_voxels=int(membrane.sum()),
            contact_voxel_count=int(summary["contact_voxel_count"]),
            crista_junction_count=int(summary["crista_junction_count"]),
            contact_volume_nm3=float(summary["contact_volume_nm3"]),
            n_cristae_touching=n_cristae_touching,
            n_mito_with_junction=n_mito_with_junction,
            frac_crista_voxels_in_contact=summary["contact_voxel_count"] / crista_voxels,
            frac_cristae_touching=(n_cristae_touching / n_crista_inst) if n_crista_inst else np.nan,
        )
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thicknesses", type=float, nargs="+", default=[8.0, 12.0])
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--default_voxel_nm", type=float, default=1.74)
    ap.add_argument("--limit", type=int, default=None, help="Process only the N smallest files.")
    ap.add_argument("--files", nargs="+", default=None, help="Explicit files (overrides discovery).")
    ap.add_argument(
        "--out_csv",
        default="/mnt/lustre-grete/usr/u12103/cristae/junction_audit/cristae_junctions_per_file.csv",
    )
    args = ap.parse_args()

    if args.files:
        files = list(args.files)
    else:
        files = discover_files()
        print(f"[discover] {len(files)} files after '_combined.h5' filter + {len(EXCLUDE_STRINGS)} exclusions")
        files = sorted(files, key=os.path.getsize)  # smallest first
        if args.limit:
            files = files[: args.limit]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    all_rows = []
    t_start = time.time()
    for i, path in enumerate(files, 1):
        t0 = time.time()
        try:
            rows = analyse_file(path, args.thicknesses, args.n_jobs, args.default_voxel_nm)
        except Exception as exc:  # keep going; record the failure
            print(f"[{i}/{len(files)}] ERROR {os.path.basename(path)}: {exc}", flush=True)
            all_rows.append(dict(path=path, file=os.path.basename(path), error=str(exc)))
            continue
        all_rows.extend(rows)
        # write incrementally so partial progress survives an interruption
        pd.DataFrame(all_rows).to_csv(args.out_csv, index=False)
        r8 = next((r for r in rows if r.get("thickness_nm") == args.thicknesses[0]), rows[0])
        print(
            f"[{i}/{len(files)}] {os.path.basename(path)}  "
            f"shape={r8.get('shape')}  cristae_inst={r8.get('n_cristae_instances')}  "
            f"mito_inst={r8.get('n_mito_instances')}  "
            f"junctions@{args.thicknesses[0]}nm={r8.get('crista_junction_count')}  "
            f"touching={r8.get('n_cristae_touching')}/{r8.get('n_cristae_instances')}  "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )

    pd.DataFrame(all_rows).to_csv(args.out_csv, index=False)
    print(f"[done] {len(files)} files in {(time.time() - t_start) / 60:.1f} min -> {args.out_csv}")


if __name__ == "__main__":
    sys.exit(main())
