#!/usr/bin/env python
"""Aggregate the per-file cristae-junction audit into corpus numbers + a markdown summary.

Reads the CSV produced by audit_cristae_junctions.py and answers: do we have enough cristae labels
that show cristae junctions (contact points) to the approximated mito membrane? Reports totals,
per-source and per-genotype breakdowns, distributions, and the 8 nm vs 12 nm comparison.
"""
import argparse
import os

import numpy as np
import pandas as pd


def q(s):
    s = s.dropna()
    if s.empty:
        return "n/a"
    return f"median={s.median():.0f} IQR=[{s.quantile(.25):.0f},{s.quantile(.75):.0f}] max={s.max():.0f}"


def corpus_table(df):
    """One row per thickness with the headline corpus totals (analyzable files only)."""
    rows = []
    for t, g in df.groupby("thickness_nm"):
        ana = g[~g["no_cristae"] & ~g["no_state1_mito"]]
        rows.append(dict(
            thickness_nm=t,
            files_analyzable=len(ana),
            files_with_junction=int((ana["crista_junction_count"] > 0).sum()),
            total_contact_points=int(ana["crista_junction_count"].sum()),
            total_cristae_instances=int(ana["n_cristae_instances"].sum()),
            cristae_touching=int(ana["n_cristae_touching"].sum()),
            pct_cristae_touching=100 * ana["n_cristae_touching"].sum() / max(1, ana["n_cristae_instances"].sum()),
            total_mito_instances=int(ana["n_mito_instances"].sum()),
            mito_with_junction=int(ana["n_mito_with_junction"].sum()),
            pct_mito_with_junction=100 * ana["n_mito_with_junction"].sum() / max(1, ana["n_mito_instances"].sum()),
            total_contact_voxels=int(ana["contact_voxel_count"].sum()),
        ))
    return pd.DataFrame(rows)


def group_table(df, by):
    rows = []
    for (t, key), g in df.groupby(["thickness_nm", by]):
        ana = g[~g["no_cristae"] & ~g["no_state1_mito"]]
        rows.append(dict(
            thickness_nm=t, **{by: key},
            files=len(g), files_with_junction=int((ana["crista_junction_count"] > 0).sum()),
            contact_points=int(ana["crista_junction_count"].sum()),
            cristae_inst=int(ana["n_cristae_instances"].sum()),
            cristae_touching=int(ana["n_cristae_touching"].sum()),
            mito_inst=int(ana["n_mito_instances"].sum()),
            mito_with_junction=int(ana["n_mito_with_junction"].sum()),
        ))
    return pd.DataFrame(rows).sort_values(["thickness_nm", by])


def md_table(df):
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, float):
            return "n/a" if np.isnan(v) else f"{v:.1f}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join(
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    )
    return "\n".join([header, sep, body])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/mnt/lustre-grete/usr/u12103/cristae/junction_audit/cristae_junctions_per_file.csv")
    ap.add_argument("--out_md", default="evaluation/cristae/RESULTS_cristae_junction_audit.md")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    errors = df[df.get("error").notna()] if "error" in df.columns else df.iloc[0:0]
    df = df[df["thickness_nm"].notna()].copy()
    thicknesses = sorted(df["thickness_nm"].unique())

    corpus = corpus_table(df)
    by_source = group_table(df, "source")
    by_geno = group_table(df, "genotype")

    # distributions at each thickness (analyzable files)
    dist_lines = []
    for t in thicknesses:
        ana = df[(df["thickness_nm"] == t) & ~df["no_cristae"] & ~df["no_state1_mito"]]
        dist_lines.append(f"- **{t:.0f} nm** — contact points per file: {q(ana['crista_junction_count'])}; "
                          f"per mito-instance: {q(ana['n_mito_with_junction'])}; "
                          f"crista voxels in contact: median {100*ana['frac_crista_voxels_in_contact'].median():.1f}%")

    # zero-junction files at the smallest thickness, and how many recover at the largest
    t0, t1 = thicknesses[0], thicknesses[-1]
    a0 = df[(df["thickness_nm"] == t0) & ~df["no_cristae"] & ~df["no_state1_mito"]].set_index("path")
    a1 = df[(df["thickness_nm"] == t1) & ~df["no_cristae"] & ~df["no_state1_mito"]].set_index("path")
    zero0 = a0[a0["crista_junction_count"] == 0].index
    recovered = [f for f in zero0 if f in a1.index and a1.loc[f, "crista_junction_count"] > 0]

    skipped = df[df["no_cristae"] | df["no_state1_mito"]]["path"].nunique()

    lines = []
    lines.append("# Cristae → mito-membrane contact points: data audit\n")
    lines.append("**Question:** do we have enough cristae labels that show cristae junctions "
                 "(contact points) to the mitochondrial membrane? Junctions are *not* annotated — "
                 "they are computed with the synapse-net cristae-widget recipe: the membrane is "
                 "approximated as the mito mask minus its erosion (an N-nm shell) from "
                 "`raw_mitos_combined[1]==1` (state-1, cristae-annotated mitochondria), and a contact "
                 "point is a connected component of `cristae & membrane`.\n")
    lines.append(f"Files audited: **{df['path'].nunique()}** (147-file training discovery set). "
                 f"Membrane thickness swept at **{', '.join(f'{t:.0f}' for t in thicknesses)} nm**. "
                 f"Files skipped (no cristae or no state-1 mito): {skipped}.\n")

    lines.append("## Corpus totals\n")
    lines.append(md_table(corpus) + "\n")
    lines.append("## Distributions\n")
    lines += dist_lines
    lines.append("")
    lines.append("## By data source\n")
    lines.append(md_table(by_source) + "\n")
    lines.append("## By genotype (heuristic from filename)\n")
    lines.append(md_table(by_geno) + "\n")

    lines.append("## Thickness sensitivity\n")
    lines.append(f"- At {t0:.0f} nm, {len(zero0)} analyzable files have **zero** contact points; "
                 f"of those, {len(recovered)} gain ≥1 at {t1:.0f} nm. "
                 f"Large 8→12 nm swings mean those cristae sit just inside the eroded shell — the "
                 f"contact count is sensitive to the assumed membrane thickness for those files.\n")
    if len(errors):
        lines.append(f"## Errors\n{len(errors)} files failed to process: "
                     + ", ".join(errors["file"].tolist()) + "\n")

    lines.append("## Caveats\n")
    lines.append("- Membrane is an **approximation** (erosion shell), not a real IMM/OMM segmentation; "
                 "contact counts scale with the assumed thickness.\n"
                 "- A single crista can touch the membrane in several disconnected patches, so "
                 "`contact_points` (connected components) can exceed the number of cristae instances.\n"
                 "- Voxel size is read from the file attr when present, else assumed **1.74 nm isotropic** "
                 "(all wichmann files); the nm→voxel conversion of the shell thickness depends on it.\n"
                 "- Genotype is parsed heuristically from filenames (WT / KO / DKO / unspecified).\n")

    md = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w") as fh:
        fh.write(md)
    print(md)
    print(f"\n[written] {args.out_md}")


if __name__ == "__main__":
    main()
