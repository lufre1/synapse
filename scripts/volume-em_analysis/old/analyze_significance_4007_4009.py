#!/usr/bin/env python3
"""
Statistical comparison of mitochondrial characteristics between datasets 4007 and 4009.
Uses both parametric (t-test) and non-parametric (Mann-Whitney U) tests.
Generates comprehensive visualizations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os

# Paths
DATA_DIR = Path(__file__).parent.parent
BASE_DIR = DATA_DIR / "scripts"

DATASETS = {
    "4007": {
        "cell_summary_path": DATA_DIR / "4007_analysis" / "4007_cell_summary_normalized.csv",
        "mito_normalized_path": DATA_DIR / "4007_analysis" / "4007_mito_normalized.csv",
        "output_dir": DATA_DIR / "4007_analysis",
    },
    "4009": {
        "cell_summary_path": DATA_DIR / "4009_analysis" / "4009_cell_summary_normalized.csv",
        "mito_normalized_path": DATA_DIR / "4009_analysis" / "4009_mito_normalized.csv",
        "output_dir": DATA_DIR / "4009_analysis",
    },
}

# Metrics to compare (cell-level)
CELL_LEVEL_METRICS = [
    "cell_volume_um3",
    "num_mitos_annotated", 
    "mean_mito_volume_um3",
    "median_mito_volume_um3",
    "mean_mito_size_norm",
    "mean_mito_sphericity",
    "mean_mito_sv_ratio",
    "mean_mito_elongation",
    "mean_mito_flatness",
    "mito_volume_fraction",
]

# Metrics to compare (mito-level)
MITO_LEVEL_METRICS = [
    "mito_volume_um3",
    "sphericity",
    "sv_ratio",
    "elongation_a_over_c",
    "flatness_b_over_c",
    "mito_size_norm",
]


def load_data(dataset: str):
    """Load cell summary and mito data."""
    path_config = DATASETS[dataset]
    
    cell_summary = pd.read_csv(path_config["cell_summary_path"])
    mito_data = pd.read_csv(path_config["mito_normalized_path"])
    
    return cell_summary, mito_data


def run_statistical_tests(cell_4007: pd.DataFrame, cell_4009: pd.DataFrame) -> dict:
    """Run both t-test and Mann-Whitney U test for each metric."""
    results = {}
    
    for metric in CELL_LEVEL_METRICS:
        if metric not in cell_4007.columns or metric not in cell_4009.columns:
            continue
        
        v407 = cell_4007[metric].dropna()
        v409 = cell_4009[metric].dropna()
        
        # Check both datasets have enough data
        if len(v407) < 2 or len(v409) < 2:
            continue
        
        # T-test (parametric)
        t_stat, t_pvalue = stats.ttest_ind(v407, v409, equal_var=False)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(v407, v409, alternative="two-sided")
        
        # Effect size (Cohen's d for t-test)
        mean_pooled = np.sqrt((v407.var() + v409.var()) / 2)
        cohens_d = (v407.mean() - v409.mean()) / mean_pooled if mean_pooled != 0 else 0
        
        # Rank-biserial correlation for U-test (non-parametric effect size)
        n1, n2 = len(v407), len(v409)
        r = 1 - (2 * u_stat) / (n1 * n2)
        
        results[metric] = {
            "4007_mean": v407.mean(),
            "4007_std": v407.std(),
            "4007_n": len(v407),
            "4009_mean": v409.mean(),
            "4009_std": v409.std(),
            "4009_n": len(v409),
            "t_statistic": t_stat,
            "t_pvalue": t_pvalue,
            "t_significant": t_pvalue < 0.05,
            "u_statistic": u_stat,
            "u_pvalue": u_pvalue,
            "u_significant": u_pvalue < 0.05,
            "cohens_d": cohens_d,
            "rank_biserial_r": r,
            "4009_4007_ratio": v409.mean() / v407.mean() if v407.mean() != 0 else np.nan,
        }
    
    # Collect metrics and p-values
    metrics_list = list(results.keys())
    all_pvalues = [results[m]["u_pvalue"] for m in metrics_list]
    rejected, corrected_pvalues, _, _ = multipletests(all_pvalues, method="fdr_bh")
    
    # Assign corrected p-values
    for metric, reject, corr_p in zip(metrics_list, rejected, corrected_pvalues):
        results[metric]["corrected_pvalue"] = corr_p
        results[metric]["significant_after_correction"] = reject
    
    return results


def create_violin_plot(data_list, labels, title, output_path):
    """Create violin plot showing distribution comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))
    vp = ax.violinplot(data_list, showmeans=True, showmedians=True, showextrema=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_boxplot(data_list, labels, title, output_path):
    """Create box plot showing box plot."""
    plt.figure(figsize=(8, 6))
    plt.boxplot(data_list, patch_artist=True, showmeans=True)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("Value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_stripplot(data_list, labels, title, output_path):
    """Create strip plot with individual data points."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Find minimum length and truncate if needed
    min_len = min(len(d) for d in data_list)
    truncated = [d[:min_len] for d in data_list]
    
    # Create DataFrame
    df = pd.DataFrame({
        labels[0]: truncated[0],
        labels[1]: truncated[1]
    })
    
    # Melt to long format
    df_melted = df.melt(var_name="Dataset", value_name="Value")
    
    sns.stripplot(data=df_melted, x="Dataset", y="Value", jitter=0.15, alpha=0.5, s=5, ax=ax)
    sns.boxplot(data=df_melted, x="Dataset", y="Value", width=0.2, color='none', linecolor='black', linewidth=1, ax=ax)
    
    ax.set_ylabel("Value")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def create_all_plots_for_metric(metric, cell_4007: pd.DataFrame, cell_4009: pd.DataFrame, output_dir: Path):
    """Generate all plot types for a single metric."""
    v407 = cell_4007[metric].dropna().values
    v409 = cell_4009[metric].dropna().values
    
    # Skip if either dataset has too few points
    if len(v407) < 10 or len(v409) < 10:
        return
    
    labels = ["4007", "4009"]
    plots_dir = output_dir / "plots"
    
    # Create plots
    create_violin_plot([v407, v409], labels, f"{metric.replace('_', ' ').title()}", 
                       plots_dir / f"violin_{metric}.png")
    
    create_boxplot([v407, v409], labels, f"{metric.replace('_', ' ').title()}", 
                   plots_dir / f"boxplot_{metric}.png")
    
    create_stripplot([v407, v409], labels, f"{metric.replace('_', ' ').title()}", 
                     plots_dir / f"strip_{metric}.png")
    
    print(f"  Created plots for: {metric}")


def create_combined_comparison_plot(results: dict, output_path: Path):
    """Create a combined comparison visualization."""
    fig = plt.figure(figsize=(10, 8))
    
    metrics = list(results.keys())
    effect_sizes = np.array([results[m]["cohens_d"] for m in metrics])
    
    colors = ['red' if e > 0 else 'blue' if e < 0 else 'gray' for e in effect_sizes]
    sig = np.array([results[m]["significant_after_correction"] for m in metrics])
    
    # Effect sizes bar plot
    ax = fig.add_subplot(111)
    bars = ax.bar(range(len(metrics)), effect_sizes, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel("Cohen's d (Effect Size)")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha='right')
    ax.set_title("Effect Size Comparison (4009 vs 4007)\nRed: 4009>4007, Blue: 4007>4009, Gray: No effect, *: sig")
    ax.set_ylim(-1.5, 1.5)
    
    # Mark significant results with stars
    for i, (bar, is_sig) in enumerate(zip(bars, sig)):
        if is_sig:
            height = bar.get_height()
            offset = 0.05 if abs(height) < 0.5 else 0.1
            ax.text(bar.get_x() + bar.get_width()/2., 
                    height + offset if height > 0 else height - offset,
                    "*", ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=12, color='green')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def print_summary_table(results: dict):
    """Print a formatted summary table."""
    print("\n" + "="*110)
    print("STATISTICAL COMPARISON SUMMARY (FDR-Benjamini-Hochberg corrected)")
    print("="*110)
    
    print(f"\n{'Metric':<30} {'4007 Mean':>12} {'4009 Mean':>12} {'Ratio':>8} {'u-pval':>10} {'adj-p':>10} {'Sig':>6} {'d':>8}")
    print("-"*110)
    
    for metric, res in sorted(results.items()):
        sig_marker = "*" if res["significant_after_correction"] else ""
        print(f"{metric:<30} {res['4007_mean']:>12.4f} {res['4009_mean']:>12.4f} "
              f"{res['4009_4007_ratio']:>8.3f} {res['u_pvalue']:>10.4f} {res['corrected_pvalue']:>10.4f} "
              f"{sig_marker:>6} {res['cohens_d']:>8.3f}")


def main():
    print("Loading data...")
    cell_4007, mito_4007 = load_data("4007")
    cell_4009, mito_4009 = load_data("4009")
    
    print(f"  4007 cells with mitos: {len(cell_4007)}")
    print(f"  4009 cells with mitos: {len(cell_4009)}")
    
    # Run statistical tests
    print("\nRunning statistical tests...")
    results = run_statistical_tests(cell_4007, cell_4009)
    
    # Create output directories
    for dataset, config in DATASETS.items():
        plots_dir = config["output_dir"] / "plots"
        plots_dir.mkdir(exist_ok=True)
        print(f"\nOutput directory: {plots_dir}")
    
    # Generate plots for each metric
    print("\nGenerating visualizations...")
    for dataset_key, dataset_config in DATASETS.items():
        plots_dir = dataset_config["output_dir"] / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        for metric in CELL_LEVEL_METRICS:
            if metric in results and metric in cell_4007.columns and metric in cell_4009.columns:
                v407 = cell_4007[metric].dropna().values
                v409 = cell_4009[metric].dropna().values
                if len(v407) > 0 and len(v409) > 0:
                    create_all_plots_for_metric(metric, cell_4007, cell_4009, dataset_config["output_dir"])
                    print(f"  ✓ {metric}: saved to {plots_dir}")
        
        # Create combined comparison plot
        print(f"\nCombined plot: {plots_dir / 'comparison_summary.png'}")
        create_combined_comparison_plot(results, plots_dir / "comparison_summary.png")
        
        # Save results to CSV
        results_df = pd.DataFrame(results).T
        results_df.to_csv(plots_dir / "statistical_results.csv", index=True)
        print(f"  ✓ Statistical results: {plots_dir / 'statistical_results.csv'}")


if __name__ == "__main__":
    main()
