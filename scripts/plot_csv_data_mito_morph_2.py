# --------------------------------------------------------------
# 1️⃣  IMPORTS & SETTINGS
# --------------------------------------------------------------
import pathlib, numpy as np, pandas as pd
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy import stats
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
sns.set(style="whitegrid", font_scale=1.2, rc={"figure.figsize": (10, 6)})
plt.rcParams["savefig.bbox"] = "tight"
FIG_DIR = pathlib.Path("figures")
FIG_DIR.mkdir(exist_ok=True)

try:                                    # pragma: no cover
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception as exc:                # pragma: no cover
    PLOTLY_AVAILABLE = False
    px = None                           # name exists, avoids NameError
    import warnings
    warnings.warn(
        f"Plotly could not be imported ({exc!r}). 3‑D scatter will be skipped.",
        UserWarning,
    )


import pathlib, matplotlib.pyplot as plt, seaborn as sns
import pandas as pd
from typing import List, Tuple, Optional

# --------------------------------------------------------------
# 1️⃣  Box‑plot + optional jittered strip (good for small‑N groups)
# --------------------------------------------------------------
def plot_boxstrip(
    df: pd.DataFrame,
    metric: str,
    label: Optional[str] = None,
    out_dir: pathlib.Path = FIG_DIR,
    jitter: bool = True,
    palette: str = "pastel",
) -> pathlib.Path:
    """
    Box‑plot of `metric` by phenotype, optionally overlaid with a
    jittered strip (shows every observation).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="phenotype",
        y=metric,
        palette=palette,
        showcaps=True,
        boxprops=dict(alpha=0.7),
        whiskerprops=dict(lw=1.5),
        medianprops=dict(color="black", lw=2),
    )
    if jitter:
        sns.stripplot(
            data=df,
            x="phenotype",
            y=metric,
            color="black",
            size=3,
            jitter=0.25,
            alpha=0.6,
        )
    plt.title(f"{label or metric} by phenotype")
    plt.ylabel(label or metric)
    plt.xlabel("Phenotype")
    plt.tight_layout()
    out_path = out_dir / f"boxstrip_{metric}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_ridge(
    df: pd.DataFrame,
    metric: str,
    out_dir: pathlib.Path = FIG_DIR,
    cmap: str = "viridis",
) -> pathlib.Path:
    """
    Stacked KDE (ridge/joy plot) of `metric` for each phenotype.
    Requires the `joypy` package (pip install joypy).
    """
    import joypy  # local import – optional dependency

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    joypy.joyplot(
        df,
        by="phenotype",
        column=metric,
        colormap=plt.cm.get_cmap(cmap),
        linewidth=1,
        overlap=1,
        fade=True,
    )
    plt.title(f"Ridge plot of {metric} by phenotype")
    plt.xlabel(metric)
    plt.tight_layout()
    out_path = out_dir / f"ridge_{metric}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


# --------------------------------------------------------------
# 3️⃣  ECDF – easy comparison of cumulative distributions
# --------------------------------------------------------------
def plot_ecdf(
    df: pd.DataFrame,
    metric: str,
    out_dir: pathlib.Path = FIG_DIR,
    palette: str = "Set2",
) -> pathlib.Path:
    """
    Empirical CDF of `metric` coloured by phenotype.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.ecdfplot(
        data=df,
        x=metric,
        hue="phenotype",
        palette=palette,
        linewidth=2,
    )
    plt.title(f"ECDF of {metric} by phenotype")
    plt.xlabel(metric)
    plt.ylabel("Cumulative proportion")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f"ecdf_{metric}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def auto_plot_metric(
    df: pd.DataFrame,
    col: str,
    label: str,
    out_dir: pathlib.Path = FIG_DIR,
    min_points_for_violin: int = 30,
) -> pathlib.Path:
    """
    Choose a sensible plot for ``col`` based on the data type,
    number of points per phenotype and the number of phenotypes.

    Parameters
    ----------
    df        : pd.DataFrame – must contain a categorical column called
                ``phenotype`` (the output of ``cluster`` or
                ``define_phenotype_rules``).
    col       : str – column name to visualise.
    label     : str – human‑readable axis label (e.g. “Volume (µm³)”).
    out_dir   : pathlib.Path – where the PNG will be saved.
    min_points_for_violin : int – phenotype to show a
                violin/KDE. Below this we fall back to a strip/box plot.

    Returns
    -------
    pathlib.Path – path of the saved figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1️⃣  Basic stats that drive the decision
    # ------------------------------------------------------------------
    n_per_group = df.groupby("phenotype").size().min()          # smallest group size
    n_phenotypes = df["phenotype"].nunique()
    uniq_vals    = df[col].nunique()
    is_integer   = pd.api.types.is_integer_dtype(df[col]) or uniq_vals < 15

    # ------------------------------------------------------------------
    # 2️⃣  Discrete / count‑type columns (e.g. skeleton_branches,
    #     skeleton_endpoints, touches_border)
    # ------------------------------------------------------------------
    if is_integer and uniq_vals <= 15:
        # Bar‑plot of the *counts* of each integer value, coloured by phenotype
        plt.figure(figsize=(9, 5))
        sns.countplot(
            data=df,
            x=col,
            hue="phenotype",
            palette="Set2",
            edgecolor="black",
        )
        plt.title(f"{label} (counts) by phenotype")
        plt.xlabel(label)
        plt.ylabel("Count")
        plt.legend(title="Phenotype", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        out_path = out_dir / f"count_{col}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    # ------------------------------------------------------------------
    # 3️⃣  Very few points per phenotype → strip + box (shows every point)
    # ------------------------------------------------------------------
    if n_per_group <= 10:
        # Box‑plot with a jittered strip on top
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=df,
            x="phenotype",
            y=col,
            palette="pastel",
            showcaps=True,
            boxprops=dict(alpha=0.7),
            whiskerprops=dict(lw=1.5),
            medianprops=dict(color="black", lw=2),
        )
        sns.stripplot(
            data=df,
            x="phenotype",
            y=col,
            color="black",
            size=3,
            jitter=0.25,
            alpha=0.7,
        )
        plt.title(f"{label} by phenotype (few points)")
        plt.ylabel(label)
        plt.xlabel("Phenotype")
        plt.tight_layout()
        out_path = out_dir / f"boxstrip_{col}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    # ------------------------------------------------------------------
    # 4️⃣  Many phenotypes → ridge/joy plot (stacked KDEs)
    # ------------------------------------------------------------------
    if n_phenotypes > 5:
        try:
            import joypy  # optional dependency
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "joypy is required for ridge plots. Install with `pip install joypy`."
            ) from exc

        plt.figure(figsize=(10, 6))
        joypy.joyplot(
            df,
            by="phenotype",
            column=col,
            colormap=plt.cm.viridis,
            linewidth=1,
            overlap=1,
            fade=True,
        )
        plt.title(f"{label} – ridge plot by phenotype")
        plt.xlabel(label)
        plt.tight_layout()
        out_path = out_dir / f"ridge_{col}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    # ------------------------------------------------------------------
    # 5️⃣  Enough points for a KDE → violin (with inner box)
    # ------------------------------------------------------------------
    if n_per_group >= min_points_for_violin:
        plt.figure(figsize=(8, 6))
        sns.violinplot(
            data=df,
            x="phenotype",
            y=col,
            inner="box",
            palette="muted",
            cut=0,
        )
        plt.title(f"{label} by phenotype")
        plt.ylabel(label)
        plt.xlabel("Phenotype")
        # If the range spans >100×, switch to log‑scale for readability
        if df[col].max() / max(df[col].min(), 1e-12) > 100:
            plt.yscale("log")
        plt.tight_layout()
        out_path = out_dir / f"violin_{col}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    # ------------------------------------------------------------------
    # 6️⃣  Fallback – simple box‑plot (should rarely be hit)
    # ------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="phenotype", y=col, palette="pastel")
    plt.title(f"{label} by phenotype")
    plt.ylabel(label)
    plt.xlabel("Phenotype")
    plt.tight_layout()
    out_path = out_dir / f"box_{col}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def load_data(csv_path: pathlib.Path = pathlib.Path("/home/freckmann15/data/mitochondria/volume-em/embl/paper/cutout_1_ground_truth_phenotypes.csv")) -> pd.DataFrame:
    """
    Load the CSV containing the mitochondrial measurements.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with the raw data.
    """
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


def define_phenotype_rules(df):
    vol_5  = np.percentile(df["volume_um3"], 5)
    vol_95 = np.percentile(df["volume_um3"], 95)

    def classify(row):
        el, fl, sp = row["elongation_a_over_c"], row["flatness_b_over_c"], row["sphericity"]
        ec, br, vol = row["euler_char"], row["skeleton_branches"], row["volume_um3"]
        # 1) Donut
        if ec == 0:
            return "Donut"
        # 2) Branched
        if br > 0:
            return "Branched"
        # 3) Swollen
        if vol >= vol_95 and sp <= 0.5:
            return "Swollen"
        # 4) Round
        if el <= 1.6 and sp >= 0.8:
            return "Round"
        # 5) Tubular
        if el >= 3.0 and fl <= 1.2:
            return "Tubular"
        # 6) Catch‑all
        return "Other"

    df["phenotype"] = df.apply(classify, axis=1)
    return df


def cluster(df, k=None, k_range=range(3, 7), random_state=0, standardize=True):
    """
    Parameters
    - df: input DataFrame (can contain non-numeric columns; they are ignored).
    - k: number of clusters to use. If None, choose k from k_range by silhouette score.
    - k_range: range of k values to try if k is None (default 3..6).
    - random_state: random seed for KMeans.
    - standardize: if True, z-score features before clustering.

    Returns
    - DataFrame with an added categorical column 'phenotype' (Cluster 1, Cluster 2, ...).

    Notes
    - Non-numeric columns are ignored.
    - Columns with zero variance are dropped.
    - NaN/inf values are imputed per column with the median.
    """
    df_out = df.copy()

    # Use all numeric columns
    X = df_out.select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError("No numeric columns found to cluster on.")

    # Clean: replace inf, impute NaN with column median
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Drop constant (zero-variance) columns
    var = X.var(axis=0)
    keep_cols = var.index[var > 0]
    X = X[keep_cols]
    if X.shape[1] == 0:
        raise ValueError("All numeric columns have zero variance after cleaning.")

    # Standardize
    if standardize:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = X.to_numpy(dtype=float)

    n = Xs.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 rows to cluster.")

    # Choose k if not provided
    if k is None:
        best_k, best_score, best_model = None, -np.inf, None
        # Cap k to at most n-1
        k_candidates = [kk for kk in k_range if 2 <= kk <= max(2, n - 1)]
        if not k_candidates:
            k_candidates = [2] if n >= 2 else []
        for kk in k_candidates:
            km = KMeans(n_clusters=kk, n_init=50, random_state=random_state)
            labels = km.fit_predict(Xs)
            # Silhouette requires at least 2 clusters and no empty clusters
            if len(np.unique(labels)) < 2:
                continue
            # Avoid singletons if possible
            counts = np.bincount(labels, minlength=kk)
            if (counts < 2).any() and n > kk:
                continue
            score = silhouette_score(Xs, labels)
            if score > best_score:
                best_k, best_score, best_model = kk, score, km
        if best_model is None:
            # Fallback
            best_k = 2 if n >= 2 else 1
            best_model = KMeans(n_clusters=best_k, n_init=50, random_state=random_state).fit(Xs)
        model = best_model
        k_final = best_k
    else:
        if k < 1 or k > n:
            raise ValueError(f"Invalid k={k} for n={n}.")
        model = KMeans(n_clusters=k, n_init=50, random_state=random_state).fit(Xs)
        k_final = k

    labels = model.labels_.astype(int)
    # Human-readable names: Cluster 1..k
    df_out["phenotype"] = pd.Categorical([f"Cluster {i+1}" for i in labels],
                                        categories=[f"Cluster {i+1}" for i in range(k_final)],
                                        ordered=True)
    return df_out


def plot_violin_swarm(
    df: pd.DataFrame,
    metric: str,
    label: Optional[str] = None,
    out_dir: pathlib.Path = FIG_DIR,
    log_scale_threshold: float = 100.0,
) -> pathlib.Path:
    """
    Create a violin plot with an over‑laid swarm for a single metric
    split by the ``phenotype`` column.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing the metric and a categorical ``phenotype`` column.
    metric : str
        Column name to plot.
    label : str | None
        Y‑axis label. If ``None`` the column name is used.
    out_dir : pathlib.Path
        Directory where the PNG will be saved.
    log_scale_threshold : float
        If ``max/min`` of the data exceeds this value the y‑axis is set to log.

    Returns
    -------
    pathlib.Path
        Path to the saved figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    y_label = label or metric

    plt.figure()
    sns.violinplot(
        data=df,
        x="phenotype",
        y=metric,
        inner=None,
        palette="muted",
        cut=0,
    )
    sns.swarmplot(
        data=df,
        x="phenotype",
        y=metric,
        color=".25",
        size=3,
        alpha=0.7,
    )
    plt.title(f"{y_label} by phenotype")
    plt.ylabel(y_label)
    plt.xlabel("Phenotype")

    # Automatic log‑scale if the range is huge
    if df[metric].max() / max(df[metric].min(), 1e-12) > log_scale_threshold:
        plt.yscale("log")

    plt.tight_layout()
    out_path = out_dir / f"{metric}_by_phenotype.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_all_violins(
    df: pd.DataFrame,
    metrics: List[Tuple[str, str]],
    out_dir: pathlib.Path = FIG_DIR,
) -> List[pathlib.Path]:
    """
    Convenience wrapper that loops over ``metrics`` (list of (col, label))
    and calls :func:`plot_violin_swarm` for each.

    Returns
    -------
    List[pathlib.Path] – paths of the generated PNG files.
    """
    paths = []
    for col, label in metrics:
        p = plot_violin_swarm(df, col, label, out_dir)
        paths.append(p)
    return paths


def plot_pairgrid(
    df: pd.DataFrame,
    vars_: List[str],
    hue: str = "phenotype",
    palette: str = "Set2",
    out_dir: pathlib.Path = FIG_DIR,
) -> pathlib.Path:
    """
    Seaborn ``pairplot`` showing pairwise relationships of the supplied
    variables, coloured by phenotype.

    Returns the path to the saved PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    g = sns.pairplot(
        df,
        vars=vars_,
        hue=hue,
        palette=palette,
        plot_kws={"alpha": 0.6, "s": 30},
        diag_kind="kde",
        corner=True,
    )
    g.fig.suptitle("Pairwise relationships of mitochondrial metrics", y=1.02)
    out_path = out_dir / "pairgrid_shape_metrics.png"
    g.savefig(out_path)
    plt.close(g.fig)
    return out_path


def plot_3d_scatter(
    df: pd.DataFrame,
    size_metric: str = "volume_um3",
    out_dir: pathlib.Path = FIG_DIR,
    width: int = 900,
    height: int = 800,
    *,
    min_marker: float = 5.0,
    max_marker: float = 30.0,
) -> Optional[pathlib.Path]:
    """
    Plotly 3‑D scatter of the centroid coordinates, coloured by phenotype.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns ``centroid_um_x``, ``centroid_um_y``,
        ``centroid_um_z`` and ``phenotype``.
    size_metric : str, default ``"volume_um3"``
        Column whose values are used (after a log‑scale) to set the marker size.
    out_dir : pathlib.Path, default ``FIG_DIR``
        Directory where the interactive HTML file will be written.
    width, height : int, default 900 / 800
        Figure size in pixels.
    min_marker, max_marker : float, default 5 / 30
        Desired range for the marker size after rescaling.  Values are in
        screen‑pixel units (the same units Plotly expects for ``size``).

    Returns
    -------
    pathlib.Path | None
        Path to the saved HTML file, or ``None`` when Plotly is not available.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Guard – Plotly must be installed
    # ------------------------------------------------------------------
    if not PLOTLY_AVAILABLE:          # pragma: no cover
        print("\n⚠️ Plotly not installed – 3‑D scatter skipped.\n")
        return None

    # ------------------------------------------------------------------
    # 2️⃣  Prepare output folder
    # ------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3️⃣  Build a *positive* marker‑size vector
    # ------------------------------------------------------------------
    # 3a – make sure we are dealing with a float array
    raw = df[size_metric].astype(float).to_numpy(copy=True)

    # 3b – replace non‑positive values (0 or negative) with a tiny positive
    #      number so that log10 does not explode to -inf.
    raw[raw <= 0] = np.nan
    # If the whole column is NaN we fall back to a constant size.
    if np.isnan(raw).all():
        size_scaled = np.full_like(raw, (min_marker + max_marker) / 2.0)
    else:
        # 3c – log‑scale (base‑10).  Adding a small epsilon avoids log10(0)
        log_vals = np.log10(np.where(np.isnan(raw), 1e-12, raw))

        # 3d – normalise to [0, 1]
        lo, hi = np.nanmin(log_vals), np.nanmax(log_vals)
        # Guard against constant column (lo == hi)
        if np.isclose(lo, hi):
            norm = np.zeros_like(log_vals)
        else:
            norm = (log_vals - lo) / (hi - lo)

        # 3e – stretch to the user‑defined marker range
        size_scaled = norm * (max_marker - min_marker) + min_marker

        # Replace any remaining NaNs (e.g. original zeros) with the median size
        median_sz = np.median(size_scaled[~np.isnan(size_scaled)])
        size_scaled = np.where(np.isnan(size_scaled), median_sz, size_scaled)

    # ------------------------------------------------------------------
    # 4️⃣  Build the Plotly figure
    # ------------------------------------------------------------------
    fig = px.scatter_3d(
        df,
        x="centroid_um_x",
        y="centroid_um_y",
        z="centroid_um_z",
        color="phenotype",
        size=size_scaled,                     # already a 1‑D array of positives
        hover_data=[
            "label",
            "volume_um3",
            "elongation_a_over_c",
            "sphericity",
            "euler_char",
        ],
        title="3‑D distributionria centroids",
        width=width,
        height=height,
    )
    # Slight transparency makes dense regions easier to read
    fig.update_traces(marker=dict(opacity=0.8))

    # ------------------------------------------------------------------
    # 5️⃣  Save the interactive HTML file
    # ------------------------------------------------------------------
    out_path = out_dir / "centroids_3d.html"
    fig.write_html(out_path)

    return out_path


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Return Wilson confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - alpha / 2)
    phat = k / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    rad = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (centre - rad) / denom, (centre + rad) / denom


# def plot_phenotype_fractions(
#     df: pd.DataFrame,
#     out_dir: pathlib.Path = FIG_DIR,
#     palette: str = {
#         "Cluster 1": "#6596DF",  # pastel blue",
#         "Cluster 2": "#FF6961",  # pastel red
#     },  #"pastel",
# ) -> pathlib.Path:
#     """
#     Bar plot showing the proportion of each phenotype together with a
#     95 % Wilson confidence interval.

#     Returns the path to the saved PNG.
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)

#     prop_df = (
#         df.groupby("phenotype")
#         .size()
#         .reset_index(name="count")
#         .assign(total=len(df))
#     )
#     prop_df["prop"] = prop_df["count"] / prop_df["total"]
#     prop_df[["ci_low", "ci_high"]] = prop_df.apply(
#         lambda r: pd.Series(wilson_ci(r["count"], r["total"])), axis=1
#     )

#     # Order bars by decreasing proportion (makes the plot easier to read)
#     order = prop_df.sort_values("prop", ascending=False)["phenotype"]

#     plt.figure()
#     sns.barplot(
#         data=prop_df,
#         x="phenotype",
#         y="prop",
#         palette=palette,
#         order=order,
#         edgecolor="black",
#         linewidth=0.8,
#     )
#     plt.errorbar(
#         x=np.arange(len(prop_df)),
#         y=prop_df["prop"],
#         yerr=[
#             prop_df["prop"] - prop_df["ci_low"],
#             prop_df["ci_high"] - prop_df["prop"],
#         ],
#         fmt="none",
#         c="k",
#         capsize=5,
#     )
#     plt.ylim(0, 1)
#     plt.ylabel("Fraction of mitochondria")
#     plt.title("Phenotype composition")
#     plt.tight_layout()

#     out_path = out_dir / "phenotype_fractions.png"
#     plt.savefig(out_path)
#     plt.close()
#     return out_path
def plot_phenotype_fractions(df, out_dir=FIG_DIR, palette=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1️⃣  Compute counts, proportions and Wilson CIs -----------------
    prop_df = (
        df.groupby("phenotype")
          .size()
          .reset_index(name="count")
          .assign(total=len(df))
    )
    prop_df["prop"] = prop_df["count"] / prop_df["total"]
    prop_df[["ci_low", "ci_high"]] = prop_df.apply(
        lambda r: pd.Series(wilson_ci(r["count"], r["total"])), axis=1
    )

    # ---- 2️⃣  Sort by decreasing proportion (the order we want to show) ----
    prop_df = prop_df.sort_values("prop", ascending=False).reset_index(drop=True)

    # ---- 3️⃣  Plot the bars ------------------------------------------------
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(
        data=prop_df,
        x="phenotype",
        y="prop",
        palette=palette,
        edgecolor="black",
        linewidth=0.8,
    )

    # ---- 4️⃣  Add the confidence‑interval error bars -----------------------
    ax.errorbar(
        x=np.arange(len(prop_df)),          # now matches the sorted order
        y=prop_df["prop"],
        yerr=[
            prop_df["prop"] - prop_df["ci_low"],
            prop_df["ci_high"] - prop_df["prop"],
        ],
        fmt="none",
        c="k",
        capsize=5,
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of mitochondria")
    ax.set_title("Phenotype composition")
    plt.tight_layout()

    out_path = out_dir / "phenotype_fractions.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def plot_correlation_matrix(
    df: pd.DataFrame,
    vars_: List[str],
    cmap: str = "coolwarm",
    out_dir: pathlib.Path = FIG_DIR,
) -> pathlib.Path:
    """
    Pearson correlation heat‑map for the supplied variables.

    Returns the path to the saved PNG.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    corr = df[vars_].corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
    )
    plt.title("Correlation matrix of mitochondrial metrics")
    plt.tight_layout()

    out_path = out_dir / "correlation_matrix.png"
    plt.savefig(out_path)
    plt.close()
    return out_path


def numeric_summary(df: pd.DataFrame, cols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Return a dictionary ``{col: {mean, median, std, min, max}}`` for the
    requested columns.
    """
    summary = {}
    for c in cols:
        s = df[c]
        summary[c] = {
            "mean": s.mean(),
            "median": s.median(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
        }
    return summary


def print_numeric_summary(summary: Dict[str, Dict[str, float]]) -> None:
    """Pretty‑print the dictionary returned by :func:`numeric_summary`. """
    print("\n=== Numeric summary ===")
    for col, stats_ in summary.items():
        print(
            f"{col:>20}: mean={stats_['mean']:.3g}, median={stats_['median']:.3g}, "
            f"sd={stats_['std']:.3g}, min={stats_['min']:.3g}, max={stats_['max']:.3g}"
        )


# --------------------------------------------------------------
#  MAIN PIPELINE ------------------------------------------------
# --------------------------------------------------------------
def main(
    csv_path: pathlib.Path,
    *,
    use_rule_based: bool = False,
    k_clusters: Optional[int] = None,
    out_dir: pathlib.Path = FIG_DIR,
    random_state: int = 0,
) -> None:
    """
    Run the complete mitochondrial‑phenotype analysis.

    Parameters
    ----------
    csv_path : pathlib.Path
        Path to the CSV that contains the raw measurements.
    use_rule_based : bool, default=False
        If ``True`` use the deterministic rule‑based phenotype definition.
        If ``False`` use K‑means clustering.
    k_clusters : int | None, default=None
        When clustering, force the algorithm to use this number of clusters.
        If ``None`` the optimal *k* is chosen automatically via silhouette score.
    out_dir : pathlib.Path, default=FIG_DIR
        Directory where all figures will be written.
    random_state : int, default=0
        Seed for reproducible clustering (only used when ``use_rule_based=False``).

    Returns
    -------
    None
        The function prints a few tables to stdout and writes all figures to
        ``out_dir``.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Load the data
    # ------------------------------------------------------------------
    df = load_data(csv_path)

    # ------------------------------------------------------------------
    # 2️⃣  Phenotype assignment
    # ------------------------------------------------------------------
    if use_rule_based:
        df = define_phenotype_rules(df)
        print("\n✅ Rule‑based phenotypes assigned.")
    else:
        df = cluster(df, k=k_clusters, random_state=random_state)
        print("\n✅ K‑means clustering completed.")
        if k_clusters is None:
            print(f"   → Optimal number of clusters chosen automatically.")
        else:
            print(f"   → Fixed number of clusters: {k_clusters}")

    # ------------------------------------------------------------------
    # 3️⃣  Quick overview of phenotype frequencies
    # ------------------------------------------------------------------
    print("\n🧬 Phenotype distribution")
    print(df["phenotype"].value_counts())

    # ------------------------------------------------------------------
    # 4️⃣  Plotting ----------------------------------------------------
    # ------------------------------------------------------------------
    # 4a – Violin + swarm for each metric
    metrics = [
        # ("centroid_px_z", "Centroid z (px)"),
        # ("centroid_px_y", "Centroid y (px)"),
        # ("centroid_px_x", "Centroid x (px)"),
        # ("centroid_um_z", "Centroid z (µm)"),
        # ("centroid_um_y", "Centroid y (µm)"),
        # ("centroid_um_x", "Centroid x (µm)"),
        ("volume_um3", "Volume (µm³)"),
        ("surface_um2", "Surface area (µm²)"),
        ("sv_ratio", "Surface/volume ratio"),
        ("sphericity", "Sphericity"),
        # ("a_um", "Principal semi-axes a (µm)"),
        # ("b_um", "Principal semi-axes b (µm)"),
        # ("c_um", "Principal semi-axes c (µm)"),
        # ("elongation_a_over_c", "Elongation (a/c)"),
        # ("flatness_b_over_c", "Flatness (b/c)"),
        # ("isotropy_c_over_a", "Isotropy (c/a)"),
        # ("euler_char", "Euler characteristic"),
        ("skeleton_length_um", "Skeleton length (µm)"),
        # ("skeleton_branches", "Skeleton branches"),
        # ("skeleton_endpoints", "Skeleton endpoints"),
        ("touches_border", "Touches border"),
        ("intensity_max", "Intensity max (AU)"),
        ("intensity_mean", "Intensity mean (AU)"),
        ("intensity_min", "Intensity min (AU)"),
        ("intensity_std", "Intensity std (AU)"),
        ("nearest_neighbor_um", "Nearest-neighbor distance (µm)"),
    ]
    print("\n📊 Creating violin‑swarm plots …")
    # plot_all_violins(df, metrics, out_dir=out_dir)
    for col, label in metrics:
        auto_plot_metric(df, col, label, out_dir=out_dir)

    # 4b – Pair‑grid of shape metrics
    # pair_vars = [
    #     "volume_um3",
    #     "surface_um2",
    #     "sphericity",
    #     "nearest_neighbor_um",
    # ]
    pair_vars = [
        "volume_um3",          # size anchor
        "sv_ratio",            # membrane complexity
        "sphericity",          # compactness
        "elongation_a_over_c",  # shape anisotropy
        "skeleton_length_um",  # network length
        "nearest_neighbor_um",  # spatial packing
    ]
    print("📊 Creating pair‑grid …")
    plot_pairgrid(df, pair_vars, out_dir=out_dir)

    # 4c – 3‑D scatter (Plotly) – optional
    if PLOTLY_AVAILABLE:
        print("📊 Creating 3‑D scatter (Plotly) …")
        plot_3d_scatter(df, out_dir=out_dir)
    else:
        print("⚠️ Plotly not installed – skipping 3‑D scatter.")

    # 4d – Bar chart of phenotype fractions (with Wilson CI)
    print("📊 Creating phenotype‑fraction bar chart …")
    plot_phenotype_fractions(df, out_dir=out_dir)

    # 4e – Correlation matrix heat‑map
    corr_vars = [
        "volume_um3",
        "surface_um2",
        "sv_ratio",
        "sphericity",
        # "elongation_a_over_c",
        # "flatness_b_over_c",
        # "isotropy_c_over_a",
        # "euler_char",
        "skeleton_length_um",
        "nearest_neighbor_um",
        "intensity_mean",
    ]
    print("📊 Creating correlation‑matrix heat‑map …")
    plot_correlation_matrix(df, corr_vars, out_dir=out_dir)

    # ------------------------------------------------------------------
    # 5️⃣  Numeric summary (for the manuscript)
    # ------------------------------------------------------------------
    metric_names = [c for c, _ in metrics]   # just the column identifiers
    summary = numeric_summary(df, metric_names)
    print_numeric_summary(summary)

    # ------------------------------------------------------------------
    # 6️⃣  Phenotype fractions table (with 95 % CI)
    # ------------------------------------------------------------------
    prop_df = (
        df.groupby("phenotype")
        .size()
        .reset_index(name="count")
        .assign(total=len(df))
    )
    prop_df["prop"] = prop_df["count"] / prop_df["total"]
    prop_df[["ci_low", "ci_high"]] = prop_df.apply(
        lambda r: pd.Series(wilson_ci(r["count"], r["total"])), axis=1
    )
    print("\n📈 Phenotype fractions (95 % Wilson CI):")
    print(prop_df[["phenotype", "prop", "ci_low", "ci_high"]].to_string(index=False))

    print("\n✅ All figures saved to:", out_dir.resolve())

    print("\nDone.")


if __name__ == "__main__":
    # Path to the original CSV (adjust to your environment)
    CSV_PATH = pathlib.Path(
        "/home/freckmann15/data/mitochondria/volume-em/embl/paper/cutout_1_ground_truth_phenotypes.csv"
    )

    # Example 1 – let the script pick the best K‑means solution automatically
    main(CSV_PATH, use_rule_based=False, k_clusters=None)

    # Example 2 – use the deterministic rule‑based phenotypes instead
    # main(CSV_PATH, use_rule_based=True)

    # Example 3 – force K‑means to use 4 clusters
    # main(CSV_PATH, use_rule_based=False, k_clusters=4)

# metrics = [
#     ("volume_um3", "Volume (µm³)"),
#     ("surface_um2", "Surface area (µm²)"),
#     ("sphericity", "Sphericity"),
#     ("elongation_a_over_c", "Elongation (a/c)"),
#     ("flatness_b_over_c", "Flatness (b/c)"),
#     ("euler_char", "Euler characteristic"),
#     ("nearest_neighbor_um", "Nearest‑neighbor distance (µm)"),
# ]

# for col, label in metrics:
#     plt.figure()
#     sns.violinplot(data=df, x="phenotype", y=col, inner=None, palette="muted", cut=0)
#     sns.swarmplot(data=df, x="phenotype", y=col, color=".25", size=3, alpha=0.7)
#     plt.title(f"{label} by phenotype")
#     plt.ylabel(label)
#     plt.xlabel("Phenotype")
#     if df[col].max() / max(df[col].min(), 1e-12) > 100:
#         plt.yscale("log")
#     plt.tight_layout()
#     plt.savefig(FIG_DIR / f"{col}_by_phenotype.png")
#     plt.show()


# pair_vars = [
#     "volume_um3",
#     "elongation_a_over_c",
#     "flatness_b_over_c",
#     "sphericity",
#     "nearest_neighbor_um",
# ]

# g = sns.pairplot(df, vars=pair_vars, hue="phenotype", palette="Set2",
#                 plot_kws={"alpha": 0.6, "s": 30}, diag_kind="kde", corner=True)
# g.fig.suptitle("Pairwise relationships of mitochondrial metrics", y=1.02)
# g.savefig(FIG_DIR / "pairgrid_shape_metrics.png")
# plt.show()


# if PLOTLY_AVAILABLE:
#     size = np.log10(df["volume_um3"] + 1e-12) * 5 + 2
#     fig = px.scatter_3d(df,
#                         x="centroid_um_x", y="centroid_um_y", z="centroid_um_z",
#                         color="phenotype", size=size,
#                         hover_data=["label", "volume_um3", "elongation_a_over_c",
#                                     "sphericity", "euler_char"],
#                         title="3‑D distribution of mitochondria centroids",
#                         width=900, height=800)
#     fig.update_traces(marker=dict(opacity=0.8))
#     fig.write_html(FIG_DIR / "centroids_3d.html")
#     fig.show()
# else:
#     print("\n⚠️ Plotly not available – 3‑D scatter skipped.\n")

# from scipy import stats
# prop_df = (df.groupby("phenotype")
#              .size()
#              .reset_index(name="count")
#              .assign(total=len(df)))
# prop_df["prop"] = prop_df["count"] / prop_df["total"]

# def wilson_ci(k, n, alpha=0.05):
#     if n == 0:
#         return 0.0, 0.0
#     z = stats.norm.ppf(1 - alpha / 2)
#     phat = k / n
#     denom = 1 + z**2 / n
#     centre = phat + z**2 / (2 * n)
#     rad = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
#     return (centre - rad) / denom, (centre + rad) / denom

# prop_df[["ci_low", "ci_high"]] = prop_df.apply(
#     lambda r: pd.Series(wilson_ci(r["count"], r["total"])), axis=1)

# plt.figure()
# sns.barplot(data=prop_df, x="phenotype", y="prop", palette="pastel",
#             order=prop_df.sort_values("prop", ascending=False)["phenotype"],
#             edgecolor="black", linewidth=0.8)
# plt.errorbar(x=np.arange(len(prop_df)),
#              y=prop_df["prop"],
#              yerr=[prop_df["prop"] - prop_df["ci_low"],
#                    prop_df["ci_high"] - prop_df["prop"]],
#              fmt="none", c="k", capsize=5)
# plt.ylim(0, 1)
# plt.ylabel("Fraction of mitochondria")
# plt.title("Phenotype composition")
# plt.tight_layout()
# plt.savefig(FIG_DIR / "phenotype_fractions.png")
# plt.show()


# corr_vars = [
#     "volume_um3", "surface_um2", "sv_ratio", "sphericity",
#     "elongation_a_over_c", "flatness_b_over_c", "isotropy_c_over_a",
#     "euler_char", "skeleton_length_um", "nearest_neighbor_um",
#     "intensity_mean"
# ]
# corr = df[corr_vars].corr()
# plt.figure(figsize=(12, 9))
# sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
#             vmin=-1, vmax=1, linewidths=0.5,
#             cbar_kws={"label": "Pearson r"})
# plt.title("Correlation matrix of mitochondrial metrics")
# plt.tight_layout()
# plt.savefig(FIG_DIR / "correlation_matrix.png")
# plt.show()


# metric_names = [c for c, _ in metrics]   # only the column names
# def summarize(col):
#     return {"mean": df[col].mean(),
#             "median": df[col].median(),
#             "std": df[col].std(),
#             "min": df[col].min(),
#             "max": df[col].max()}

# summary = {c: summarize(c) for c in metric_names}
# print("\n=== Numeric summary ===")
# for c, s in summary.items():
#     print(f"{c:>20}: mean={s['mean']:.3g}, median={s['median']:.3g}, "
#           f"sd={s['std']:.3g}, min={s['min']:.3g}, max={s['max']:.3g}")

# print("\nPhenotype fractions (95 % CI):")
# print(prop_df[["phenotype", "prop", "ci_low", "ci_high"]].to_string(index=False))
