#!/usr/bin/env python3
"""
Build cell-level mitochondrial summary tables for datasets 4007 and 4009.

Key principles:
- Only include mitochondria assigned to annotated cells
- Normalize counts/totals by cell volume
- Do NOT directly normalize shape features (sphericity, elongation, flatness) by cell volume
- Summarize mitochondrion-level morphology per cell
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent

DATASETS = {
    "4007": {
        "cell_path": DATA_DIR / "4007_analysis" / "cell_summary_3d.csv",
        "mito_path": DATA_DIR / "4007_analysis" / "mito_morphometrics_3d.csv",
        "out_dir": DATA_DIR / "4007_analysis",
    },
    "4009": {
        "cell_path": DATA_DIR / "4009_analysis" / "cell_summary_3d.csv",
        "mito_path": DATA_DIR / "4009_analysis" / "mito_morphometrics_3d.csv",
        "out_dir": DATA_DIR / "4009_analysis",
    },
}


def load_and_merge(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cell + mito tables and merge mito rows with parent cell volume."""
    cells = pd.read_csv(DATASETS[dataset]["cell_path"])
    mitos = pd.read_csv(DATASETS[dataset]["mito_path"])

    # Keep only cells with at least one mitochondrion
    cells = cells[cells["mito_count"] > 0].copy()

    # Keep only mitochondria assigned to a cell
    mitos = mitos[mitos["cell_id"].notna()].copy()

    # Merge parent cell volume onto each mitochondrion row
    merged = mitos.merge(
        cells[["label_id", "volume_um3"]].rename(columns={"label_id": "cell_id", "volume_um3": "cell_volume_um3"}),
        on="cell_id",
        how="inner"
    )

    # Rename mito volume if needed
    if "volume_um3" in merged.columns:
        merged = merged.rename(columns={"volume_um3": "mito_volume_um3"})

    # Optional mito-level density-like measure:
    # interpretable as "fraction of host cell volume occupied by this mito"
    if "mito_volume_um3" in merged.columns and "cell_volume_um3" in merged.columns:
        merged["mito_volume_fraction_of_cell"] = merged["mito_volume_um3"] / merged["cell_volume_um3"]

    return merged, cells


def compute_cell_level_summary(merged: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cell mitochondrial summaries."""
    # Start from cell-level data
    keep_cols = ["label_id", "volume_um3", "mito_count"]
    optional_cols = ["mito_volume_um3", "mito_volume_fraction", "mito_surface_um2"]

    existing_optional = [c for c in optional_cols if c in cells.columns]
    cell_metrics = cells[keep_cols + existing_optional].copy()

    cell_metrics = cell_metrics.rename(columns={
        "label_id": "cell_id",
        "volume_um3": "cell_volume_um3",
    })

    # Compute normalized cell-level density/content metrics
    cell_metrics["mito_count_density"] = cell_metrics["mito_count"] / cell_metrics["cell_volume_um3"]

    if "mito_volume_um3" in cell_metrics.columns:
        cell_metrics["mito_volume_density"] = cell_metrics["mito_volume_um3"] / cell_metrics["cell_volume_um3"]

    if "mito_surface_um2" in cell_metrics.columns:
        cell_metrics["mito_surface_density"] = cell_metrics["mito_surface_um2"] / cell_metrics["cell_volume_um3"]

    # Aggregate mitochondrion-level morphology per cell
    agg_dict = {}

    if "label_id" in merged.columns:
        agg_dict["label_id"] = ("label_id", "count")

    if "mito_volume_um3" in merged.columns:
        agg_dict["mean_mito_volume_um3"] = ("mito_volume_um3", "mean")
        agg_dict["median_mito_volume_um3"] = ("mito_volume_um3", "median")
        agg_dict["std_mito_volume_um3"] = ("mito_volume_um3", "std")

    if "mito_volume_fraction_of_cell" in merged.columns:
        agg_dict["mean_single_mito_fraction_of_cell"] = ("mito_volume_fraction_of_cell", "mean")

    if "sphericity" in merged.columns:
        agg_dict["mean_mito_sphericity"] = ("sphericity", "mean")
        agg_dict["median_mito_sphericity"] = ("sphericity", "median")

    if "sv_ratio" in merged.columns:
        agg_dict["mean_mito_sv_ratio"] = ("sv_ratio", "mean")
        agg_dict["median_mito_sv_ratio"] = ("sv_ratio", "median")

    if "elongation_a_over_c" in merged.columns:
        agg_dict["mean_mito_elongation"] = ("elongation_a_over_c", "mean")
        agg_dict["median_mito_elongation"] = ("elongation_a_over_c", "median")

    if "flatness_b_over_c" in merged.columns:
        agg_dict["mean_mito_flatness"] = ("flatness_b_over_c", "mean")
        agg_dict["median_mito_flatness"] = ("flatness_b_over_c", "median")

    mito_level = merged.groupby("cell_id").agg(**agg_dict).reset_index()

    if "label_id" in mito_level.columns:
        mito_level = mito_level.rename(columns={"label_id": "num_mitos_annotated"})

    full = cell_metrics.merge(mito_level, on="cell_id", how="inner")

    # Useful log-transformed columns for skewed size metrics
    for col in ["cell_volume_um3", "mito_count", "mito_count_density", "mito_volume_um3",
                "mito_volume_density", "mean_mito_volume_um3", "median_mito_volume_um3"]:
        if col in full.columns:
            full[f"log10_{col}"] = np.log10(full[col] + 1e-9)

    return full


def main():
    for dataset, cfg in DATASETS.items():
        print(f"\nProcessing dataset {dataset}...")
        merged, cells = load_and_merge(dataset)

        print(f"  Cells with mitochondria: {len(cells)}")
        print(f"  Mitochondria assigned to cells: {len(merged)}")

        cell_summary = compute_cell_level_summary(merged, cells)

        out_dir = cfg["out_dir"]
        out_dir.mkdir(exist_ok=True, parents=True)

        merged_out = out_dir / f"{dataset}_mito_assigned_to_cells.csv"
        summary_out = out_dir / f"{dataset}_cell_level_mito_summary.csv"

        merged.to_csv(merged_out, index=False)
        cell_summary.to_csv(summary_out, index=False)

        print(f"  Saved mito-level merged data: {merged_out}")
        print(f"  Saved cell-level summary:    {summary_out}")


if __name__ == "__main__":
    main()