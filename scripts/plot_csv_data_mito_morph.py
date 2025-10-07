# --------------------------------------------------------------
# 1️⃣  IMPORTS & GLOBAL SETTINGS
# --------------------------------------------------------------
import pathlib
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Optional interactive 3‑D visualisation
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:               # plotly not installed – the script will still run
    PLOTLY_AVAILABLE = False

# Make plots look nice (you can tweak the style later)
sns.set(style="whitegrid", font_scale=1.2, rc={"figure.figsize": (10, 6)})
plt.rcParams["savefig.bbox"] = "tight"

# Folder for output figures
FIG_DIR = pathlib.Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------
# 2️⃣  LOAD THE CSV
# --------------------------------------------------------------
csv_path = pathlib.Path("/home/freckmann15/data/mitochondria/volume-em/embl/paper/cutout_1_ground_truth_phenotypes.csv")   # <-- change if needed
df = pd.read_csv(csv_path)

# Quick sanity check
print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")
print(df.head())
print("\nData types:")
print(df.dtypes)

# --------------------------------------------------------------
# 3️⃣  BASIC CLEAN‑UP / DERIVED COLUMNS
# --------------------------------------------------------------

# Convert the “touches_border” column to a proper Boolean (it may be 0/1 or string)
if df["touches_border"].dtype != bool:
    df["touches_border"] = df["touches_border"].astype(bool)

# -----------------------------------------------------------------
# OPTIONAL: drop objects that touch the image border (they are truncated)
# -----------------------------------------------------------------
# Uncomment the next line if you want to exclude them from the plots
# df = df[~df["touches_border"]].reset_index(drop=True)

# -----------------------------------------------------------------
# Create a simple “phenotype” label from shape descriptors.
# This is only an illustration – you can replace the thresholds with
# whatever you used in your manuscript.
# -----------------------------------------------------------------
def phenotype_from_shape(row):
    # a ≥ b ≥ c are already sorted in the analysis function
    a, b, c = row["a_um"], row["b_um"], row["c_um"]
    elong = row["elongation_a_over_c"]
    flat  = row["flatness_b_over_c"]
    euler = row["euler_char"]

    # 1) round/compact
    if elong < 1.6 and flat < 1.2 and euler == 1:
        return "Round"
    # 2) tubular / rod‑like
    if elong >= 3.5 and flat < 1.3 and euler == 1:
        return "Tubular"
    # 3) donut / toroidal (single hole)
    if euler == 0:
        return "Donut"
    # 4) branched / complex (multiple branches in skeleton)
    if row["skeleton_branches"] > 0:
        return "Branched"
    # fallback
    return "Other"

df["phenotype"] = df.apply(phenotype_from_shape, axis=1)

# Show phenotype distribution
print("\nPhenotype counts:")
print(df["phenotype"].value_counts())

# --------------------------------------------------------------
# 4️⃣  PLOT 1 – HISTOGRAM / VIOLIN OF KEY METRICS
# --------------------------------------------------------------
metrics = [
    ("volume_um3", "Volume (µm³)"),
    ("surface_um2", "Surface area (µm²)"),
    ("sphericity", "Sphericity"),
    ("elongation_a_over_c", "Elongation (a/c)"),
    ("flatness_b_over_c", "Flatness (b/c)"),
    ("euler_char", "Euler characteristic"),
    ("nearest_neighbor_um", "Nearest‑neighbor distance (µm)"),
]

for col, label in metrics:
    plt.figure()
    # Violin + swarm (shows distribution + individual points)
    sns.violinplot(
        data=df,
        x="phenotype",
        y=col,
        inner=None,
        palette="muted",
        cut=0,
    )
    sns.swarmplot(
        data=df,
        x="phenotype",
        y=col,
        color=".25",
        size=3,
        alpha=0.7,
    )
    plt.title(f"{label} by phenotype")
    plt.ylabel(label)
    plt.xlabel("Phenotype")
    plt.yscale("log" if df[col].max() / df[col].min() > 100 else "linear")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{col}_by_phenotype.png")
    plt.show()

# --------------------------------------------------------------
# 5️⃣  PLOT 2 – PAIR‑GRID (scatter matrix) OF SELECTED SHAPE METRICS
# --------------------------------------------------------------
pair_vars = [
    "volume_um3",
    "elongation_a_over_c",
    "flatness_b_over_c",
    "sphericity",
    "nearest_neighbor_um",
]

# Use a hue to colour by phenotype
g = sns.pairplot(
    df,
    vars=pair_vars,
    hue="phenotype",
    palette="Set2",
    plot_kws={"alpha": 0.6, "s": 30},
    diag_kind="kde",
    corner=True,          # only lower‑triangle
)
g.fig.suptitle("Pairwise relationships of mitochondrial shape/size", y=1.02)
g.savefig(FIG_DIR / "pairgrid_shape_metrics.png")
plt.show()

# --------------------------------------------------------------
# 6️⃣  PLOT 3 – 3‑D SCATTER OF CENTROIDS (interactive if plotly is installed)
# --------------------------------------------------------------
if PLOTLY_AVAILABLE:
    # Scale point size by volume (log‑scaled for visibility)
    size = np.log10(df["volume_um3"] + 1e-12) * 5 + 2

    fig = px.scatter_3d(
        df,
        x="centroid_um_x",
        y="centroid_um_y",
        z="centroid_um_z",
        color="phenotype",
        size=size,
        hover_data=[
            "label",
            "volume_um3",
            "elongation_a_over_c",
            "sphericity",
            "euler_char",
        ],
        title="3‑D distribution of mitochondria centroids",
        width=900,
        height=800,
    )
    fig.update_traces(marker=dict(opacity=0.8))
    fig.write_html(FIG_DIR / "centroids_3d.html")
    fig.show()
else:
    print("\n⚠️ Plotly not installed – skipping interactive 3‑D scatter.\n"
          "You can install it with `pip install plotly` and re‑run the script.")

# --------------------------------------------------------------
# 7️⃣  PLOT 4 – BAR CHART OF PHENOTYPE FRACTIONS (with 95 % CI)
# --------------------------------------------------------------
from scipy import stats

# Compute proportion and Wilson confidence interval for each phenotype
prop_df = (
    df.groupby("phenotype")
    .size()
    .reset_index(name="count")
    .assign(total=len(df))
)
prop_df["prop"] = prop_df["count"] / prop_df["total"]

# Wilson interval (more accurate for small N)
def wilson_ci(k, n, alpha=0.05):
    """Return lower, upper bound of Wilson confidence interval."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    rad = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    lower = (centre - rad) / denom
    upper = (centre + rad) / denom
    return lower, upper


prop_df[["ci_low", "ci_high"]] = prop_df.apply(
    lambda r: pd.Series(wilson_ci(r["count"], r["total"])), axis=1
)

plt.figure()
sns.barplot(
    data=prop_df,
    x="phenotype",
    y="prop",
    palette="pastel",
    order=prop_df.sort_values("prop", ascending=False)["phenotype"],
)
# Add error bars (95 % CI)
plt.errorbar(
    x=np.arange(len(prop_df)),
    y=prop_df["prop"],
    yerr=[prop_df["prop"] - prop_df["ci_low"], prop_df["ci_high"] - prop_df["prop"]],
    fmt="none",
    c="k",
    capsize=5,
)
plt.ylim(0, 1)
plt.ylabel("Fraction of mitochondria")
plt.title("Phenotype composition of the Moebius dataset")
plt.tight_layout()
plt.savefig(FIG_DIR / "phenotype_fractions.png")
plt.show()

# --------------------------------------------------------------
# 8️⃣  OPTIONAL – CORRELATION MATRIX (heatmap)
# --------------------------------------------------------------
corr_vars = [
    "volume_um3",
    "surface_um2",
    "sv_ratio",
    "sphericity",
    "elongation_a_over_c",
    "flatness_b_over_c",
    "isotropy_c_over_a",
    "euler_char",
    "skeleton_length_um",
    "nearest_neighbor_um",
    "intensity_mean",
]

corr = df[corr_vars].corr(method="pearson")

plt.figure(figsize=(12, 9))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "Pearson r"},
)
plt.title("Correlation matrix of mitochondrial metrics")
plt.tight_layout()
plt.savefig(FIG_DIR / "correlation_matrix.png")
plt.show()

# --------------------------------------------------------------
# 9️⃣  QUICK SUMMARY PRINT‑OUT (for the manuscript)
# --------------------------------------------------------------
metric_names = [col for col, _ in metrics]

def summarize_numeric(col):
    return {
        "mean":   df[col].mean(),
        "median": df[col].median(),
        "std":    df[col].std(),
        "min":    df[col].min(),
        "max":    df[col].max(),
    }

summary = {c: summarize_numeric(c) for c in metric_names}

print("\n=== Numeric summary (rounded) ===")
for col, stats_dict in summary.items():
    print(
        f"{col:>20}:  mean={stats_dict['mean']:.3g},  median={stats_dict['median']:.3g}, "
        f"sd={stats_dict['std']:.3g},  min={stats_dict['min']:.3g}, max={stats_dict['max']:.3g}"
    )

print("\nPhenotype fractions (with 95 % CI):")
print(prop_df[["phenotype", "prop", "ci_low", "ci_high"]].to_string(index=False))

# --------------------------------------------------------------
# END OF SCRIPT
# --------------------------------------------------------------
