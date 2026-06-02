#!/usr/bin/env python3
"""
Statistical comparison of cell-level mitochondrial summaries between datasets 4007 and 4009.

Uses:
- Welch t-test
- Mann-Whitney U test
- FDR correction
- effect sizes
- publication-style plots
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests



DATA_DIR = Path(__file__).parent.parent.parent

DATASETS = {
    "4007": DATA_DIR / "4007_analysis" / "4007_cell_level_mito_summary.csv",
    "4009": DATA_DIR / "4009_analysis" / "4009_cell_level_mito_summary.csv",
}

OUTPUT_DIR = DATA_DIR / "comparison_analysis_with_tests"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CELL_LEVEL_METRICS = [
    "cell_volume_um3",
    # count / density (prefer density for cross-dataset comparison)
    "mito_count",
    "mito_count_density",
    # volume / density
    "mito_volume_um3",
    "mito_volume_fraction",
    "mito_volume_density",
    # per-mito size
    "mean_mito_volume_um3",
    "median_mito_volume_um3",
    # shape metrics (intrinsically normalised — no cell-volume normalisation needed)
    "mean_mito_sphericity",
    "median_mito_sphericity",
    "mean_mito_sv_ratio",
    "median_mito_sv_ratio",
    "mean_mito_elongation",
    "median_mito_elongation",
    "mean_mito_flatness",
    "median_mito_flatness",
    # num_mitos_annotated omitted: identical to mito_count after QC filtering
]

LOG_METRICS_IF_AVAILABLE = [
    "log10_cell_volume_um3",
    "log10_mito_count",
    "log10_mito_count_density",
    "log10_mito_volume_um3",
    "log10_mito_volume_density",
    "log10_mean_mito_volume_um3",
    "log10_median_mito_volume_um3",
]


def load_data():
    d4007 = pd.read_csv(DATASETS["4007"])
    d4009 = pd.read_csv(DATASETS["4009"])
    return d4007, d4009


def cohen_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * sx + (ny - 1) * sy) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(y) - np.mean(x)) / pooled  # positive means 4009 > 4007


def rank_biserial_from_u(u_stat, n1, n2):
    # positive means tendency for second group to be larger if U corresponds to group1
    return 1 - (2 * u_stat) / (n1 * n2)


def run_tests(df1: pd.DataFrame, df2: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    results = []

    for metric in metrics:
        if metric not in df1.columns or metric not in df2.columns:
            continue

        x = df1[metric].dropna().values
        y = df2[metric].dropna().values

        if len(x) < 2 or len(y) < 2:
            continue

        # Welch t-test
        t_stat, t_p = stats.ttest_ind(x, y, equal_var=False)

        # Mann-Whitney U
        u_stat, u_p = stats.mannwhitneyu(x, y, alternative="two-sided")

        results.append({
            "metric": metric,
            "n_4007": len(x),
            "mean_4007": np.mean(x),
            "median_4007": np.median(x),
            "std_4007": np.std(x, ddof=1),
            "n_4009": len(y),
            "mean_4009": np.mean(y),
            "median_4009": np.median(y),
            "std_4009": np.std(y, ddof=1),
            "ratio_mean_4009_4007": np.mean(y) / np.mean(x) if np.mean(x) != 0 else np.nan,
            "welch_t_stat": t_stat,
            "welch_t_p": t_p,
            "mw_u_stat": u_stat,
            "mw_u_p": u_p,
            "cohens_d_4009_vs_4007": cohen_d(x, y),
            "rank_biserial": rank_biserial_from_u(u_stat, len(x), len(y)),
        })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        reject, p_adj, _, _ = multipletests(results_df["mw_u_p"], method="fdr_bh")
        results_df["mw_u_p_fdr_bh"] = p_adj
        results_df["significant_fdr_0_05"] = reject

    return results_df.sort_values("mw_u_p")


def make_long_df(df1, df2, metric):
    a = pd.DataFrame({"dataset": "4007", "value": df1[metric].dropna().values})
    b = pd.DataFrame({"dataset": "4009", "value": df2[metric].dropna().values})
    return pd.concat([a, b], ignore_index=True)


def plot_metric(df1, df2, metric, outdir: Path):
    long_df = make_long_df(df1, df2, metric)

    plt.figure(figsize=(7, 5))
    sns.violinplot(data=long_df, x="dataset", y="value", inner=None, cut=0)
    sns.boxplot(data=long_df, x="dataset", y="value", width=0.2, showcaps=True,
                boxprops={"facecolor": "none", "zorder": 3},
                showfliers=False, whiskerprops={"linewidth": 1}, zorder=3)
    sns.stripplot(data=long_df, x="dataset", y="value", color="black", alpha=0.4, jitter=0.18, size=4)
    plt.title(metric.replace("_", " "))
    plt.tight_layout()
    plt.savefig(outdir / f"{metric}.png", dpi=200)
    plt.savefig(outdir / f"{metric}.svg")
    plt.close()


def plot_effect_sizes(results_df: pd.DataFrame, outpath: Path):
    if len(results_df) == 0:
        return

    df = results_df.copy()
    df = df.sort_values("cohens_d_4009_vs_4007")

    plt.figure(figsize=(10, max(5, 0.35 * len(df))))
    colors = ["crimson" if x > 0 else "royalblue" for x in df["cohens_d_4009_vs_4007"]]
    plt.barh(df["metric"], df["cohens_d_4009_vs_4007"], color=colors, alpha=0.8)
    plt.axvline(0, color="black", linewidth=1)

    for i, (_, row) in enumerate(df.iterrows()):
        if row.get("significant_fdr_0_05", False):
            plt.text(row["cohens_d_4009_vs_4007"], i, " *", va="center")

    plt.xlabel("Cohen's d (positive = 4009 > 4007)")
    plt.ylabel("Metric")
    plt.title("Effect sizes for cell-level mitochondrial metrics")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.savefig(Path(str(outpath).replace(".png", ".svg")))
    plt.close()


def main():
    print("Loading data...")
    df4007, df4009 = load_data()

    print(f"4007 cells: {len(df4007)}")
    print(f"4009 cells: {len(df4009)}")

    plots_dir = OUTPUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Main tests on raw cell-level metrics
    metrics = [m for m in CELL_LEVEL_METRICS if m in df4007.columns and m in df4009.columns]
    results_raw = run_tests(df4007, df4009, metrics)
    results_raw.to_csv(OUTPUT_DIR / "cell_level_stats_raw.csv", index=False)

    # Optional tests on log-transformed size variables
    log_metrics = [m for m in LOG_METRICS_IF_AVAILABLE if m in df4007.columns and m in df4009.columns]
    results_log = run_tests(df4007, df4009, log_metrics)
    results_log.to_csv(OUTPUT_DIR / "cell_level_stats_log_metrics.csv", index=False)

    # Plots
    for metric in metrics:
        plot_metric(df4007, df4009, metric, plots_dir)

    plot_effect_sizes(results_raw, OUTPUT_DIR / "effect_sizes_raw.png")

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'cell_level_stats_raw.csv'}")
    print(f"  {OUTPUT_DIR / 'cell_level_stats_log_metrics.csv'}")
    print(f"  {OUTPUT_DIR / 'effect_sizes_raw.png'}")
    print(f"  {OUTPUT_DIR / 'effect_sizes_raw.svg'}")
    print(f"  plots in {plots_dir} (png + svg)")

    print("\nTop results:")
    if len(results_raw) > 0:
        print(results_raw[[
            "metric", "mean_4007", "mean_4009", "ratio_mean_4009_4007",
            "mw_u_p", "mw_u_p_fdr_bh", "significant_fdr_0_05", "cohens_d_4009_vs_4007"
        ]].head(20).to_string(index=False))
    else:
        print("No comparable metrics found.")


if __name__ == "__main__":
    main()