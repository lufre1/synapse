#!/usr/bin/env python
"""Compare cristae-junction (contact-point) counts between predicted segmentations and GT.

Uses the same membrane/contact definition as evaluation/cristae/audit_cristae_junctions.py, but the
"cristae" mask is the model's *predicted* segmentation instead of the ground-truth label. Reads test
segmentation H5s written by inference/segment_cristae.py, which already carry `seg` (prediction),
`labels/cristae` (GT), and `labels/mitochondria` (mito state 0/1/2) for every test file.

Usage:
    python evaluation/cristae/compare_predicted_junctions.py \
        --model_dirs baseline=<dir_a> promising=<dir_b> \
        --band_nm 8 12 -o <out_dir>
"""
import argparse
import os
from glob import glob

import h5py
import numpy as np
import pandas as pd
from scipy import ndimage

from synapse.cristae.membrane import DEFAULT_VOXEL_SIZE, membrane_distance


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dirs", nargs="+", required=True,
                         help="name=dir pairs; each dir holds one H5 per test file with 'seg', "
                              "'labels/cristae', 'labels/mitochondria'.")
    parser.add_argument("--band_nm", type=float, nargs="+", default=[8.0, 12.0])
    parser.add_argument("--voxel_size", type=float, nargs=3, default=None,
                         help="(z y x) nm, overriding each file's 'voxel_size' attribute.")
    parser.add_argument("-o", "--output_path", required=True)
    return parser


def _voxel_size(f, override):
    if override is not None:
        return tuple(float(v) for v in override)
    for obj in (f.get("raw"), f):
        if obj is not None and "voxel_size" in obj.attrs:
            vs = np.atleast_1d(obj.attrs["voxel_size"]).astype(float)
            return tuple(vs) if vs.size == 3 else (float(vs[0]),) * 3
    return DEFAULT_VOXEL_SIZE


def junction_counts(mask, s1, dist, mito_inst, n_mito, band_nm):
    """One row per thickness: contact points / mask-instances touching / mito touching."""
    mask_inst, n_mask = ndimage.label(mask)
    rows = []
    for t in band_nm:
        band = s1 & (dist <= t)
        contact = band & mask
        _, n_contact = ndimage.label(contact)
        touching = np.unique(mask_inst[contact])
        mito_hit = np.unique(mito_inst[contact])
        rows.append(dict(
            thickness_nm=float(t), contact_points=int(n_contact),
            instances=int(n_mask), instances_touching=int((touching > 0).sum()),
            mito_instances=int(n_mito), mito_with_junction=int((mito_hit > 0).sum()),
        ))
    return rows


def audit_file(path, band_nm, voxel_size_override):
    with h5py.File(path, "r") as f:
        state = f["labels/mitochondria"][:]
        gt = f["labels/cristae"][()] > 0
        pred = f["seg"][()] > 0
        vs = _voxel_size(f, voxel_size_override)
    s1 = state == 1
    if not s1.any():
        return []
    dist = membrane_distance(s1, vs)
    mito_inst, n_mito = ndimage.label(s1)
    rows = []
    for source, mask in (("gt", gt), ("pred", pred)):
        for row in junction_counts(mask, s1, dist, mito_inst, n_mito, band_nm):
            rows.append(dict(file=os.path.basename(path), source=source, **row))
    return rows


def main():
    args = build_parser().parse_args()
    model_dirs = dict(kv.split("=", 1) for kv in args.model_dirs)

    all_rows = []
    for model_name, d in model_dirs.items():
        paths = sorted(glob(os.path.join(d, "*_combined.h5")))
        print(f"[{model_name}] {len(paths)} files in {d}", flush=True)
        for i, p in enumerate(paths, 1):
            rows = audit_file(p, args.band_nm, args.voxel_size)
            for r in rows:
                r["model"] = model_name
            all_rows.extend(rows)
            print(f"  [{i}/{len(paths)}] {os.path.basename(p)}", flush=True)

    df = pd.DataFrame(all_rows)
    os.makedirs(args.output_path, exist_ok=True)
    out_csv = os.path.join(args.output_path, "predicted_junction_counts.csv")
    df.to_csv(out_csv, index=False)

    print("\n=== Junction counts: predicted (per model) vs GT, pooled over test files ===")
    agg = df.groupby(["thickness_nm", "model", "source"]).agg(
        files=("file", "nunique"), contact_points=("contact_points", "sum"),
        instances=("instances", "sum"), instances_touching=("instances_touching", "sum"),
        mito_instances=("mito_instances", "sum"), mito_with_junction=("mito_with_junction", "sum"),
    )
    agg["pct_instances_touching"] = (100 * agg["instances_touching"] / agg["instances"]).round(1)
    agg["pct_mito_with_junction"] = (100 * agg["mito_with_junction"] / agg["mito_instances"]).round(1)
    print(agg.to_string())
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
