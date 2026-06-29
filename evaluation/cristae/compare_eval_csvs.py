"""Print a side-by-side comparison of two cristae_eval_results.csv files (masked Dice eval).

Computes per-file means over the metric columns (excluding any pre-aggregated 'averaged' row)
so the summary is robust regardless of how each CSV was written.
"""
import argparse

import pandas as pd


METRICS = ["dice", "precision", "recall", "hd95"]


def summarize(csv_path):
    df = pd.read_csv(csv_path)
    # drop any pre-aggregated/average rows so we mean only per-file rows
    if "dataset" in df.columns:
        df = df[~df["dataset"].astype(str).str.contains("average|all-files", case=False, na=False)]
    cols = [c for c in METRICS if c in df.columns]
    return len(df), df[cols].mean(numeric_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", "-c", action="append", nargs=2, metavar=("LABEL", "PATH"),
                    required=True, help="Repeatable: a label and a CSV path. e.g. -c cristae3 a.csv -c cristae4 b.csv")
    args = ap.parse_args()

    print("\n================ cristae segmentation comparison (masked to mito state==1) ================")
    header = f"{'model':<18}{'n':>4}  " + "".join(f"{m:>12}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for label, path in args.csv:
        n, means = summarize(path)
        row = f"{label:<18}{n:>4}  "
        for m in METRICS:
            v = means.get(m, float("nan"))
            row += f"{v:>12.4f}" if pd.notna(v) else f"{'nan':>12}"
        print(row)
    print("=" * len(header))
    print("Dice/precision/recall higher = better; HD95 lower = better.\n")


if __name__ == "__main__":
    main()
