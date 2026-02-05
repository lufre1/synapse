import argparse
import os
from tqdm import tqdm
import pandas as pd
import synapse.io.util as io
import numpy as np


# def compute_statistics(files):
#     file_stats = []
#     for path in tqdm(files):
#         data = io.load_data_from_file(path)
#         file_stats.append({
#             "file": path,
#             "total_mito": len(np.unique(data["labels/mitochondria"])) - 1,  # Exclude background
#         })


#     # Convert to DataFrame for better visualization
#     df_stats = pd.DataFrame(file_stats)
#     print("\nPer-file statistics:\n", df_stats)

#     return df_stats

def compute_statistics(files, dataset_name):
    file_stats = []
    for path in tqdm(files):
        data = io.load_data_from_file(path)
        total_mito = len(np.unique(data[dataset_name])) - 1  # exclude background
        file_stats.append({"file": path, "total_mito": int(total_mito)})

    df = pd.DataFrame(file_stats)

    total_all = int(df["total_mito"].sum()) if len(df) else 0
    avg_per_file = float(df["total_mito"].mean()) if len(df) else 0.0

    summary_rows = pd.DataFrame([
        {"file": "__TOTAL_ALL_FILES__", "total_mito": total_all},
        {"file": "__MEAN_PER_FILE__",   "total_mito": avg_per_file},
    ])

    df = pd.concat([df, summary_rows], ignore_index=True)

    print("\nPer-file statistics (+ summary rows):\n", df)
    return df


def main(args):
    files = io.load_file_paths(args.path, args.ext)
    if files is None:
        print("Could not find any files")
        return
    if args.output_path is None:
        output_path = args.path
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, "mitos_stats.csv")

    df_stats = compute_statistics(files, args.dataset_name)


    # Save to CSV
    df_stats.to_csv(output_path, index=False)
    
    print(f"Statistics exported to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, required=True)
    parser.add_argument("--ext", "-e", type=str, default=None)
    parser.add_argument("--output_path", "-o", type=str, default=None)
    parser.add_argument("--dataset_name", "-dn", type=str, default=None)
    args = parser.parse_args()
    main(args)