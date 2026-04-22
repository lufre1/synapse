#!/usr/bin/env python3
"""
Compare mitochondria metrics inside axons (cells) between datasets 4007 and 4009.
Normalizes mito-level metrics by dividing by cell volume for fair comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent
OUTPUT_DIR = DATA_DIR

DATASETS = {
    "4007": {
        "cell_path": DATA_DIR / "4007_analysis" / "cell_summary_3d.csv",
        "mito_path": DATA_DIR / "4007_analysis" / "mito_morphometrics_3d.csv",
    },
    "4009": {
        "cell_path": DATA_DIR / "4009_analysis" / "cell_summary_3d.csv",
        "mito_path": DATA_DIR / "4009_analysis" / "mito_morphometrics_3d.csv",
    },
}


def load_and_merge(dataset: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load cell and mito data, filter to cells with mitos, return merged data + cell volumes."""
    cells = pd.read_csv(DATASETS[dataset]["cell_path"])
    mitos = pd.read_csv(DATASETS[dataset]["mito_path"])
    
    # Keep only cells that have at least one mito
    cell_with_mitos = cells[cells["mito_count"] > 0].copy()
    
    # Filter mitos to only those inside cells (non-null cell_id)
    mitos_with_cells = mitos[mitos["cell_id"].notna()].copy()
    
    # Merge mito data with cell volumes by matching cell_id to label_id
    merged = mitos_with_cells.merge(
        cell_with_mitos[["label_id", "volume_um3"]].rename(columns={"label_id": "cell_id"}),
        on="cell_id", 
        how="left"
    )
    
    # After merge: volume_um3_x (mito volume), volume_um3_y (cell volume)
    merged = merged.rename(columns={"volume_um3_x": "mito_volume_um3", "volume_um3_y": "cell_volume_um3"})
    
    # Drop rows where merge failed (no matching cell)
    merged = merged.dropna(subset=["cell_volume_um3"])
    
    # Add normalized metrics at mito level
    merged["mito_size_norm"] = merged["mito_volume_um3"] / merged["cell_volume_um3"]
    merged["mito_count_per_cell"] = merged.groupby("cell_id")["mito_volume_um3"].transform("sum") / merged["cell_volume_um3"]
    
    return merged, cell_with_mitos


def compute_cell_level_summary(
    dataset: str, 
    merged: pd.DataFrame, 
    cell_summary: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute cell-level aggregated metrics.
    Mito metrics are normalized by cell volume at the mito row level.
    """
    # Cell-level metrics from cell summary (not per-mito)
    cell_metrics = cell_summary[cell_summary["mito_count"] > 0][
        ["label_id", "volume_um3", "mito_count", "mito_volume_um3", "mito_volume_fraction"]
    ].rename(columns={"label_id": "cell_id", "volume_um3": "cell_volume_um3"})
    
    # Aggregate per-mito metrics per cell (means of normalized values)
    mito_level = merged.groupby("cell_id").agg(
        num_mitos_annotated=("label_id", "count"),
        mean_mito_volume_um3=("mito_volume_um3", "mean"),
        median_mito_volume_um3=("mito_volume_um3", "median"),
        std_mito_volume_um3=("mito_volume_um3", "std"),
        mean_mito_size_norm=("mito_size_norm", "mean"),
        mean_mito_sphericity=("sphericity", "mean"),
        mean_mito_sv_ratio=("sv_ratio", "mean"),
        mean_mito_elongation=("elongation_a_over_c", "mean"),
        mean_mito_flatness=("flatness_b_over_c", "mean"),
    ).reset_index()
    
    # Join cell-level with aggregated mito-level
    full_agg = cell_metrics.merge(mito_level, on="cell_id", how="inner")
    
    return full_agg


def compare_datasets_stats(c407: pd.DataFrame, c409: pd.DataFrame) -> dict:
    """Compute comparison statistics between datasets."""
    metrics = [
        "cell_volume_um3", "num_mitos_annotated",
        "mean_mito_volume_um3", "median_mito_volume_um3",
        "mean_mito_size_norm", "mean_mito_sphericity", "mean_mito_sv_ratio",
        "mean_mito_elongation", "mean_mito_flatness",
    ]
    
    results = {}
    for m in metrics:
        if m not in c407.columns or m not in c409.columns:
            continue
        
        v407 = c407[m].dropna()
        v409 = c409[m].dropna()
        
        if len(v407) == 0 or len(v409) == 0:
            continue
        
        # Means and CV (coefficient of variation = std/mean)
        results[m] = {
            "4007_mean": v407.mean(),
            "4007_cv": v407.std() / v407.mean() if v407.mean() != 0 else np.nan,
            "4009_mean": v409.mean(),
            "4009_cv": v409.std() / v409.mean() if v409.mean() != 0 else np.nan,
            "ratio_4009_4007": v409.mean() / v407.mean() if v407.mean() != 0 else np.nan,
        }
    
    return results


def main():
    print("Loading 4007 data...")
    merged_4007, cells_4007 = load_and_merge("4007")
    print(f"  Cells with mitos: {len(cells_4007)}")
    print(f"  Total mito instances: {len(merged_4007)}")
    
    print("\nLoading 4009 data...")
    merged_4009, cells_4009 = load_and_merge("4009")
    print(f"  Cells with mitos: {len(cells_4009)}")
    print(f"  Total mito instances: {len(merged_4009)}")
    
    # Compute cell-level summaries
    print("\nComputing cell-level summaries...")
    summary_4007 = compute_cell_level_summary("4007", merged_4007, cells_4007)
    summary_4009 = compute_cell_level_summary("4009", merged_4009, cells_4009)
    
    # Save summaries
    summary_4007.to_csv(OUTPUT_DIR / "4007_cell_summary_normalized.csv", index=False)
    summary_4009.to_csv(OUTPUT_DIR / "4009_cell_summary_normalized.csv", index=False)
    print("Saved: 4007_cell_summary_normalized.csv")
    print("Saved: 4009_cell_summary_normalized.csv")
    
    # Save mito-level normalized data
    merged_4007.to_csv(OUTPUT_DIR / "4007_mito_normalized.csv", index=False)
    merged_4009.to_csv(OUTPUT_DIR / "4009_mito_normalized.csv", index=False)
    print("\nSaved mito-level data:")
    print("  4007_mito_normalized.csv")
    print("  4009_mito_normalized.csv")
    
    # Compute and print comparison statistics
    print("\n" + "="*60)
    print("COMPARISON STATISTICS (4009 / 4007 ratio)")
    print("="*60)
    
    stats = compare_datasets_stats(summary_4007, summary_4009)
    
    for metric, vals in stats.items():
        print(f"\n{metric}:")
        print(f"  4007: {vals['4007_mean']:.4f} (CV={vals['4007_cv']:.3f})")
        print(f"  4009: {vals['4009_mean']:.4f} (CV={vals['4009_cv']:.3f})")
        print(f"  Ratio (4009/4007): {vals['ratio_4009_4007']:.3f}")
    
    print("\n" + "="*60)
    print("Cell-level mito volume fraction comparison")
    print("="*60)
    
    # Specific comparison of mito_volume_fraction (already normalized by cell volume)
    vf_4007 = summary_4007["mito_volume_fraction"].dropna()
    vf_4009 = summary_4009["mito_volume_fraction"].dropna()
    
    print(f"\n  4007: mean={vf_4007.mean():.4f}, median={vf_4007.median():.4f}, n={len(vf_4007)}")
    print(f"  4009: mean={vf_4009.mean():.4f}, median={vf_4009.median():.4f}, n={len(vf_4009)}")
    if vf_4007.mean() != 0:
        print(f"  Ratio 4009/4007: {vf_4009.mean()/vf_4007.mean():.3f}")


if __name__ == "__main__":
    main()
