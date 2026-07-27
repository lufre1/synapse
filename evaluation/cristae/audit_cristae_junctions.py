"""Audit the cristae ground truth at the mitochondrial membrane (the cristae-junction region).

Answers two questions about the *annotations*, independent of any model:

  A. **Is there a GT ceiling?** For every crista instance, the minimum distance from the instance to
     the mito membrane (`gap_nm`). Biologically nearly every crista connects to the inner membrane,
     so a systematically large gap means the annotations stop short of the membrane — and no loss
     reweighting or architecture change can beat that ceiling. Written to `cristae_gt_gap.csv`.

  B. **Junction counts** — the table in `RESULTS_cristae_junction_audit.md` (contact points, cristae
     touching the membrane, mito with junction), regenerated from committed code. Written to
     `cristae_junction_counts.csv`.

The membrane is approximated as the surface of the annotated-mito mask (`raw_mitos_combined[1] == 1`);
`synapse.cristae.membrane` turns that into an nm-accurate inner shell via a distance transform.
NOTE: the original (uncommitted) audit used a per-slice XY erosion, this uses a 3D EDT, so contact
counts may differ slightly; instance counts are unaffected.

Usage:
    python evaluation/cristae/audit_cristae_junctions.py -c <audit_config.yaml>
    python evaluation/cristae/audit_cristae_junctions.py --data_dirs <dir> -o <out_dir>
"""
import argparse
import os
from glob import glob

import h5py
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

from synapse.cristae.membrane import DEFAULT_VOXEL_SIZE, membrane_distance

GAP_THRESHOLDS_NM = (2.0, 4.0, 8.0, 12.0)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=None, help="YAML config; CLI flags override it.")
    parser.add_argument("--h5_paths", nargs="+", default=None,
                        help="Explicit list of combined H5 files. Takes precedence over --data_dirs.")
    parser.add_argument("--data_dirs", nargs="+", default=None,
                        help="Directories searched recursively for *.h5 (the training discovery set).")
    parser.add_argument("--path_filter", default="_combined.h5",
                        help="Only keep discovered paths containing this substring.")
    parser.add_argument("--exclude_substrings", nargs="*", default=[],
                        help="Drop discovered paths containing any of these (the known-bad files).")
    parser.add_argument("-o", "--output_path", default=None, help="Directory for the CSVs.")
    parser.add_argument("--state_key", default="raw_mitos_combined",
                        help="H5 key holding the raw/state stack; channel --state_channel is the mito state.")
    parser.add_argument("--state_channel", type=int, default=1)
    parser.add_argument("-k", "--key", default="labels/cristae", help="H5 key for the cristae labels.")
    parser.add_argument("--band_nm", type=float, nargs="+", default=[8.0, 12.0],
                        help="Membrane-shell thicknesses in nm for the junction counts.")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=None,
                        help="(z y x) nm, overriding each file's 'voxel_size' attribute.")
    return parser


def parse_args():
    parser = build_parser()
    cfg_args, _ = parser.parse_known_args()
    if cfg_args.config is not None:
        with open(cfg_args.config, "r") as f:
            parser.set_defaults(**(yaml.safe_load(f) or {}))
    args = parser.parse_args()
    if not args.output_path:
        parser.error("provide --output_path (or `output_path` in the config)")
    if not args.h5_paths and not args.data_dirs:
        parser.error("provide --h5_paths or --data_dirs (or the equivalent config keys)")
    return args


def discover(args):
    """Mirror the training-set discovery of training/cristae/train_cristae.py (glob + filters)."""
    if args.h5_paths:
        paths = list(args.h5_paths)
    else:
        paths = []
        for d in args.data_dirs:
            paths.extend(sorted(glob(os.path.join(d, "**", "*.h5"), recursive=True)))
        paths = [p for p in paths if args.path_filter in p]
        for s in (args.exclude_substrings or []):
            paths = [p for p in paths if s not in p]
    return list(dict.fromkeys(paths))  # dedupe, keep order


def _voxel_size(path, override):
    """(z, y, x) nm: explicit override > file attribute > corpus default."""
    if override is not None:
        return tuple(float(v) for v in override), "config"
    with h5py.File(path, "r") as f:
        for obj in (f.get("raw"), f):
            if obj is not None and "voxel_size" in obj.attrs:
                vs = np.atleast_1d(obj.attrs["voxel_size"]).astype(float)
                if vs.size == 3:
                    return tuple(vs), "file-attr"
                if vs.size == 1:
                    return (float(vs[0]),) * 3, "file-attr"
    return DEFAULT_VOXEL_SIZE, "default"


def _instances(labels):
    """Instance ids for a label volume: use existing ids if present, else connected components."""
    labels = np.asarray(labels)
    if labels.max() > 1:
        return labels, int(labels.max())
    inst, n = ndimage.label(labels > 0)
    return inst, n


def _source(path):
    for s in ("cooper", "wichmann"):
        if s in path:
            return s
    return "other"


def _genotype(path):
    name = os.path.basename(path)
    for g in ("DKO", "KO", "WT"):  # DKO first — it contains "KO"
        if g in name:
            return g
    return "unspecified"


def audit_file(path, args):
    """Return (gap_rows, count_rows) for one combined H5."""
    with h5py.File(path, "r") as f:
        state = f[args.state_key][args.state_channel]
        cristae = f[args.key][()]
    s1 = np.asarray(state) == 1
    vs, vs_source = _voxel_size(path, args.voxel_size)
    meta = {"file": os.path.basename(path), "source": _source(path), "genotype": _genotype(path),
            "voxel_size_z": vs[0], "voxel_size_source": vs_source}

    if not s1.any() or not (cristae > 0).any():
        print(f"  SKIP {os.path.basename(path)}: no cristae or no state-1 mito", flush=True)
        return [], []

    dist = membrane_distance(s1, vs)                       # nm to the membrane, 0 outside the mito
    crista_inst, n_crista = _instances(cristae)
    mito_inst, n_mito = ndimage.label(s1)
    idx = np.arange(1, n_crista + 1)

    # --- A. per-instance gap to the membrane -----------------------------------------------------
    # Voxels outside the annotated mito cannot define the gap -> +inf so they never win the minimum.
    dist_in_mito = np.where(s1, dist, np.inf)
    gaps = np.atleast_1d(ndimage.minimum(dist_in_mito, crista_inst, idx)).astype(float)
    sizes = np.bincount(crista_inst.ravel(), minlength=n_crista + 1)[1:]
    gap_rows = [dict(meta, instance_id=int(i), n_voxels=int(sz),
                     gap_nm=(float(g) if np.isfinite(g) else np.nan),
                     inside_state1=bool(np.isfinite(g)))
                for i, sz, g in zip(idx, sizes, gaps)
                if sz > 0]  # pre-existing label volumes may have gaps in their id range

    # --- B. junction counts per membrane thickness -----------------------------------------------
    count_rows = []
    for t in args.band_nm:
        band = s1 & (dist <= t)
        contact = band & (cristae > 0)
        _, n_contact_points = ndimage.label(contact)
        touching = np.unique(crista_inst[contact])
        mito_hit = np.unique(mito_inst[contact])
        count_rows.append(dict(
            meta, thickness_nm=float(t),
            contact_points=int(n_contact_points),
            contact_voxels=int(contact.sum()),
            cristae_instances=int(n_crista),
            cristae_touching=int((touching > 0).sum()),
            mito_instances=int(n_mito),
            mito_with_junction=int((mito_hit > 0).sum()),
            cristae_voxels=int((cristae > 0).sum()),
            band_voxels=int(band.sum()),
            state1_voxels=int(s1.sum()),
        ))
    return gap_rows, count_rows


def report(gap_df, count_df):
    print("\n=== A. GT gap: distance from each crista instance to the mito membrane ===")
    g = gap_df.loc[gap_df["inside_state1"], "gap_nm"]
    if len(g):
        print(f"  n_instances={len(g):,}  median={g.median():.2f} nm  "
              f"IQR=[{g.quantile(.25):.2f}, {g.quantile(.75):.2f}]  P90={g.quantile(.90):.2f} nm")
        for t in GAP_THRESHOLDS_NM:
            print(f"    gap <= {t:>5.1f} nm : {100 * (g <= t).mean():5.1f}%  ({int((g <= t).sum()):,} instances)")
        outside = int((~gap_df["inside_state1"]).sum())
        if outside:
            print(f"  {outside:,} crista instances lie entirely outside state==1 mito (gap undefined)")
        print("  -> median gap near one voxel = GT reaches the membrane (band AP is a real target);"
              "\n     median gap of several nm = GT under-annotates junctions (ceiling, fix labels first).")

    print("\n=== B. Junction counts (compare with RESULTS_cristae_junction_audit.md) ===")
    agg = count_df.groupby("thickness_nm").agg(
        files=("file", "nunique"),
        files_with_junction=("contact_points", lambda s: int((s > 0).sum())),
        contact_points=("contact_points", "sum"),
        cristae_instances=("cristae_instances", "sum"),
        cristae_touching=("cristae_touching", "sum"),
        mito_instances=("mito_instances", "sum"),
        mito_with_junction=("mito_with_junction", "sum"),
        contact_voxels=("contact_voxels", "sum"),
    )
    agg["pct_cristae_touching"] = (100 * agg["cristae_touching"] / agg["cristae_instances"]).round(1)
    agg["pct_mito_with_junction"] = (100 * agg["mito_with_junction"] / agg["mito_instances"]).round(1)
    print(agg.to_string())

    for key in ("source", "genotype"):
        print(f"\n--- by {key} ---")
        print(count_df.groupby(["thickness_nm", key]).agg(
            files=("file", "nunique"), contact_points=("contact_points", "sum"),
            cristae_instances=("cristae_instances", "sum"), cristae_touching=("cristae_touching", "sum"),
            mito_instances=("mito_instances", "sum"), mito_with_junction=("mito_with_junction", "sum"),
        ).to_string())


def main():
    args = parse_args()
    paths = discover(args)
    print(f"Auditing {len(paths)} files at thicknesses {args.band_nm} nm", flush=True)

    gap_rows, count_rows = [], []
    for i, p in enumerate(paths, 1):
        gr, cr = audit_file(p, args)
        gap_rows.extend(gr)
        count_rows.extend(cr)
        if cr:
            c0 = cr[0]
            print(f"  [{i}/{len(paths)}] {c0['file']}: {c0['cristae_instances']} cristae, "
                  f"{c0['contact_points']} contact points @ {c0['thickness_nm']:g} nm", flush=True)

    if not count_rows:
        raise SystemExit("No analyzable files (every file lacked cristae or state-1 mito).")

    os.makedirs(args.output_path, exist_ok=True)
    gap_df, count_df = pd.DataFrame(gap_rows), pd.DataFrame(count_rows)
    gap_df.to_csv(os.path.join(args.output_path, "cristae_gt_gap.csv"), index=False)
    count_df.to_csv(os.path.join(args.output_path, "cristae_junction_counts.csv"), index=False)
    report(gap_df, count_df)
    print(f"\nWrote cristae_gt_gap.csv and cristae_junction_counts.csv to {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
