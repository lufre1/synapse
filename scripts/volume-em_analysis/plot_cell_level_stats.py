#!/usr/bin/env python3
"""
Publication-style plots for cell-level mitochondrial summaries.

Uses the plot style from PLOT_STYLE.md:
- Color palette: WT (green) / KO (red-orange)
- Violin + boxplot + scatter overlay
- Clean spines and tick styling
- High-DPI export
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


DATA_DIR = Path(__file__).parent.parent.parent

DATASETS = {
    "4007": DATA_DIR / "4007_analysis" / "4007_cell_level_mito_summary.csv",
    "4009": DATA_DIR / "4009_analysis" / "4009_cell_level_mito_summary.csv",
}

OUTPUT_DIR = DATA_DIR / "comparison_analysis" / "plots"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

COLOR_4007 = "#1b9e77"
COLOR_4009 = "#fc8d62"

GROUP_LABELS = ["4007", "4009"]

METRICS_TO_PLOT = [
    "cell_volume_um3",
    "mito_count",
    "mito_count_density",
    "mito_volume_um3",
    "mito_volume_fraction",
    "mito_volume_density",
    "mean_mito_volume_um3",
    "median_mito_volume_um3",
    "mean_mito_sphericity",
    "median_mito_sphericity",
    "mean_mito_sv_ratio",
    "median_mito_sv_ratio",
    "mean_mito_elongation",
    "median_mito_elongation",
    "mean_mito_flatness",
    "median_mito_flatness",
]

Y_LABEL_MAP = {
    "cell_volume_um3": r"Cell volume (µm$^3$)",
    "mito_count": "Mitochondria count per cell",
    "mito_count_density": r"Mitochondria density (count/µm$^3$)",
    "mito_volume_um3": r"Total mito volume (µm$^3$)",
    "mito_volume_fraction": "Mito volume fraction",
    "mito_volume_density": r"Mito volume density (µm$^3$/µm$^3$)",
    "mean_mito_volume_um3": r"Mean mito volume (µm$^3$)",
    "median_mito_volume_um3": r"Median mito volume (µm$^3$)",
    "mean_mito_sphericity": "Mean mito sphericity",
    "median_mito_sphericity": "Median mito sphericity",
    "mean_mito_sv_ratio": "Mean SV ratio",
    "median_mito_sv_ratio": "Median SV ratio",
    "mean_mito_elongation": "Mean mito elongation",
    "median_mito_elongation": "Median mito elongation",
    "mean_mito_flatness": "Mean mito flatness",
    "median_mito_flatness": "Median mito flatness",
}


def load_data():
    d4007 = pd.read_csv(DATASETS["4007"])
    d4009 = pd.read_csv(DATASETS["4009"])
    return d4007, d4009


def plot_metric_violin_box_scatter(df1, df2, metric, outdir: Path):
    values_4007 = df1[metric].dropna().values
    values_4009 = df2[metric].dropna().values
    data = [values_4007, values_4009]
    colors = [COLOR_4007, COLOR_4009]

    fig, ax = plt.subplots(figsize=(5.0, 5.6))

    parts = ax.violinplot(
        data,
        positions=[0, 1],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.6,
        bw_method=0.3,
        points=200
    )

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.35)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.0)

    box = ax.boxplot(
        data,
        positions=[0, 1],
        widths=0.18,
        patch_artist=True,
        showfliers=False
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.2)

    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    for whisker in box["whiskers"]:
        whisker.set_linewidth(1.2)

    for cap in box["caps"]:
        cap.set_linewidth(1.2)

    rng = np.random.default_rng(0)
    jitter_width = 0.08

    ax.scatter(
        rng.normal(0, jitter_width, len(values_4007)),
        values_4007,
        s=10, alpha=0.35, color=COLOR_4007, zorder=2
    )

    ax.scatter(
        rng.normal(1, jitter_width, len(values_4009)),
        values_4009,
        s=10, alpha=0.35, color=COLOR_4009, zorder=2
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.2, length=5)

    y_label = Y_LABEL_MAP.get(metric, metric.replace("_", " ").title())
    ax.set_ylabel(y_label)
    ax.set_xlabel("Dataset")
    ax.set_xticklabels(GROUP_LABELS)

    plt.tight_layout()
    plt.savefig(outdir / f"{metric}.png", dpi=600, bbox_inches="tight")
    plt.savefig(outdir / f"{metric}.svg", bbox_inches="tight")
    plt.close()


def plot_effect_sizes_bar(results_df: pd.DataFrame, outpath: Path):
    if len(results_df) == 0:
        return

    df = results_df.copy()
    df = df.sort_values("cohens_d_4009_vs_4007")

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(df))))

    colors = [COLOR_4009 if x > 0 else COLOR_4007 for x in df["cohens_d_4009_vs_4007"]]
    ax.barh(df["metric"], df["cohens_d_4009_vs_4007"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", linewidth=1)

    for i, (_, row) in enumerate(df.iterrows()):
        if row.get("significant_fdr_0_05", False):
            ax.text(row["cohens_d_4009_vs_4007"], i, " *", va="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=1.2, length=5)

    ax.set_xlabel("Cohen's d (positive = 4009 > 4007)")
    ax.set_ylabel("Metric")
    ax.set_title("Effect sizes for cell-level mitochondrial metrics")
    plt.tight_layout()
    plt.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.savefig(Path(str(outpath).replace(".png", ".svg")), bbox_inches="tight")
    plt.close()


def main():
    print("Loading data...")
    df4007, df4009 = load_data()

    print(f"4007 cells: {len(df4007)}")
    print(f"4009 cells: {len(df4009)}")

    metrics = [m for m in METRICS_TO_PLOT if m in df4007.columns and m in df4009.columns]
    print(f"\nPlotting {len(metrics)} metrics...")

    for metric in metrics:
        print(f"  {metric}")
        plot_metric_violin_box_scatter(df4007, df4009, metric, OUTPUT_DIR)

    print(f"\nSaved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
