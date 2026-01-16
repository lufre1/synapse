import argparse
import os

import pandas as pd
import synapse.io.util as io
import numpy as np
import elf.parallel as parallel


def compute_statistics(files):
    file_stats = []
    for path in files:
        data = io.load_data_from_file(path)
        file_stats.append({
            "file": path,
            "total_mito": len(np.unique(data["labels/mitochondria"])) - 1,  # Exclude background
        })
    

    # Convert to DataFrame for better visualization
    df_stats = pd.DataFrame(file_stats)
    print("\nPer-file statistics:\n", df_stats)

    return df_stats


def main(args):
    files = io.load_file_paths(args.path, args.ext)
    if files is None:
        print("Could not find any files")
        return
    if args.output_path is None:
        output_path = args.path
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, "stats.csv")
    
    df_stats = compute_statistics(files)


    # Save to CSV
    df_stats.to_csv(output_path, index=False)
    
    print(f"Statistics exported to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, required=True)
    parser.add_argument("--ext", "-e", type=str, default=None)
    parser.add_argument("--output_path", "-o", type=str, default=None)
    args = parser.parse_args()
    main(args)