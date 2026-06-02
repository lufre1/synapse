#!/usr/bin/env python3
"""
Build cell-level mitochondrial summary tables for datasets 4007 and 4009.

Key principles:
- Only include mitochondria assigned to annotated cells
- Apply QC filters before summarising (border objects, zero-intensity, merge artifacts)
- Normalize counts/totals by cell volume
- Do NOT directly normalize shape features (sphericity, elongation, flatness) by cell volume
- Summarize mitochondrion-level morphology per cell
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# QC thresholds — adjust if your data differs
# ---------------------------------------------------------------------------
QC_EXCLUDE_BORDER = True        # remove mitos touching the volume boundary (truncated shape)
QC_MIN_INTENSITY = 1.0          # remove mitos with mean_intensity == 0 (outside FOV)
QC_MAX_ELONGATION = 20.0        # remove merge-artifact mitos (a/c > 20 is not a real mito)
QC_MAX_CELL_MITO_VOL_FRAC = 1.0 # remove cells where total mito volume exceeds cell volume

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent.parent

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


def qc_filter(mitos: pd.DataFrame, cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove artifact mitos and cells with impossible mito fractions.

    After filtering mitos, per-cell counts and volumes are recomputed from
    the clean mito table so that the cell summary reflects only valid objects.
    """
    n_mito_before = len(mitos)

    if QC_EXCLUDE_BORDER and "touches_border" in mitos.columns:
        mitos = mitos[~mitos["touches_border"]].copy()

    if "mean_intensity" in mitos.columns:
        mitos = mitos[mitos["mean_intensity"] >= QC_MIN_INTENSITY].copy()

    if "elongation_a_over_c" in mitos.columns:
        mitos = mitos[mitos["elongation_a_over_c"] < QC_MAX_ELONGATION].copy()

    n_removed = n_mito_before - len(mitos)
    print(f"  QC mitos: {n_mito_before} → {len(mitos)} ({n_removed} removed: "
          f"border / zero-intensity / elongation ≥ {QC_MAX_ELONGATION})")

    # Recompute per-cell mito count and volume from the filtered mito table.
    # The original cell_summary_3d.csv aggregates include all mitos; we replace
    # those columns so downstream density metrics reflect only clean mitos.
    assigned = mitos[mitos["cell_id"].notna()].copy()
    cell_agg = (
        assigned.groupby("cell_id")
        .agg(mito_count=("label_id", "count"),
             mito_volume_um3=("volume_um3", "sum"))
        .reset_index()
        .rename(columns={"cell_id": "label_id"})
    )

    stale = [c for c in ["mito_count", "mito_volume_um3", "mito_volume_fraction",
                          "mito_mean_volume_um3", "mito_density_per_um3"]
             if c in cells.columns]
    cells = cells.drop(columns=stale).merge(cell_agg, on="label_id", how="left")
    cells["mito_count"] = cells["mito_count"].fillna(0).astype(int)
    cells["mito_volume_um3"] = cells["mito_volume_um3"].fillna(0.0)
    cells["mito_volume_fraction"] = cells["mito_volume_um3"] / cells["volume_um3"]

    n_cell_before = len(cells)
    cells = cells[
        (cells["mito_count"] > 0) &
        (cells["mito_volume_fraction"] <= QC_MAX_CELL_MITO_VOL_FRAC)
    ].copy()
    print(f"  QC cells:  {n_cell_before} → {len(cells)} "
          f"({n_cell_before - len(cells)} removed: no mitos after filter "
          f"or mito_vol_frac > {QC_MAX_CELL_MITO_VOL_FRAC})")

    return mitos, cells


def load_and_merge(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cell + mito tables, apply QC, and merge mito rows with parent cell volume."""
    cells = pd.read_csv(DATASETS[dataset]["cell_path"])
    mitos = pd.read_csv(DATASETS[dataset]["mito_path"])

    mitos, cells = qc_filter(mitos, cells)

    # Keep only mitochondria assigned to a surviving cell
    valid_cell_ids = set(cells["label_id"].values)
    mitos = mitos[mitos["cell_id"].isin(valid_cell_ids)].copy()

    # Merge parent cell volume onto each mitochondrion row
    merged = mitos.merge(
        cells[["label_id", "volume_um3"]].rename(
            columns={"label_id": "cell_id", "volume_um3": "cell_volume_um3"}
        ),
        on="cell_id",
        how="inner",
    )

    if "volume_um3" in merged.columns:
        merged = merged.rename(columns={"volume_um3": "mito_volume_um3"})

    if "mito_volume_um3" in merged.columns and "cell_volume_um3" in merged.columns:
        merged["mito_volume_fraction_of_cell"] = (
            merged["mito_volume_um3"] / merged["cell_volume_um3"]
        )

    return merged, cells


def compute_cell_level_summary(merged: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cell mitochondrial summaries."""
    keep_cols = ["label_id", "volume_um3", "mito_count"]
    optional_cols = ["mito_volume_um3", "mito_volume_fraction", "mito_surface_um2"]

    existing_optional = [c for c in optional_cols if c in cells.columns]
    cell_metrics = cells[keep_cols + existing_optional].copy()
    cell_metrics = cell_metrics.rename(columns={
        "label_id": "cell_id",
        "volume_um3": "cell_volume_um3",
    })

    cell_metrics["mito_count_density"] = (
        cell_metrics["mito_count"] / cell_metrics["cell_volume_um3"]
    )
    if "mito_volume_um3" in cell_metrics.columns:
        cell_metrics["mito_volume_density"] = (
            cell_metrics["mito_volume_um3"] / cell_metrics["cell_volume_um3"]
        )
    if "mito_surface_um2" in cell_metrics.columns:
        cell_metrics["mito_surface_density"] = (
            cell_metrics["mito_surface_um2"] / cell_metrics["cell_volume_um3"]
        )

    agg_dict = {}

    if "mito_volume_um3" in merged.columns:
        agg_dict["mean_mito_volume_um3"]   = ("mito_volume_um3", "mean")
        agg_dict["median_mito_volume_um3"] = ("mito_volume_um3", "median")
        agg_dict["std_mito_volume_um3"]    = ("mito_volume_um3", "std")

    if "mito_volume_fraction_of_cell" in merged.columns:
        agg_dict["mean_single_mito_fraction_of_cell"] = (
            "mito_volume_fraction_of_cell", "mean"
        )

    if "sphericity" in merged.columns:
        agg_dict["mean_mito_sphericity"]   = ("sphericity", "mean")
        agg_dict["median_mito_sphericity"] = ("sphericity", "median")

    if "sv_ratio" in merged.columns:
        agg_dict["mean_mito_sv_ratio"]   = ("sv_ratio", "mean")
        agg_dict["median_mito_sv_ratio"] = ("sv_ratio", "median")

    if "elongation_a_over_c" in merged.columns:
        agg_dict["mean_mito_elongation"]   = ("elongation_a_over_c", "mean")
        agg_dict["median_mito_elongation"] = ("elongation_a_over_c", "median")

    if "flatness_b_over_c" in merged.columns:
        agg_dict["mean_mito_flatness"]   = ("flatness_b_over_c", "mean")
        agg_dict["median_mito_flatness"] = ("flatness_b_over_c", "median")

    mito_level = merged.groupby("cell_id").agg(**agg_dict).reset_index()

    full = cell_metrics.merge(mito_level, on="cell_id", how="inner")

    for col in ["cell_volume_um3", "mito_count", "mito_count_density",
                "mito_volume_um3", "mito_volume_density",
                "mean_mito_volume_um3", "median_mito_volume_um3"]:
        if col in full.columns:
            full[f"log10_{col}"] = np.log10(full[col] + 1e-9)

    return full


def main():
    for dataset, cfg in DATASETS.items():
        print(f"\nProcessing dataset {dataset}...")
        merged, cells = load_and_merge(dataset)

        print(f"  Cells surviving QC: {len(cells)}")
        print(f"  Mitos assigned to surviving cells: {len(merged)}")

        cell_summary = compute_cell_level_summary(merged, cells)

        out_dir = cfg["out_dir"]
        out_dir.mkdir(exist_ok=True, parents=True)

        merged_out  = out_dir / f"{dataset}_mito_assigned_to_cells.csv"
        summary_out = out_dir / f"{dataset}_cell_level_mito_summary.csv"

        merged.to_csv(merged_out, index=False)
        cell_summary.to_csv(summary_out, index=False)

        print(f"  Saved mito-level merged data: {merged_out}")
        print(f"  Saved cell-level summary:    {summary_out}")


if __name__ == "__main__":
    main()
