import argparse
import os

import pandas as pd
import synapse.io.util as io
import numpy as np
import elf.parallel as parallel


def filter_mitochondria_with_cristae(mitochondria: np.ndarray, cristae: np.ndarray) -> np.ndarray:
    """
    Filters out mitochondria that do not contain any cristae labels.

    Args:
        mitochondria (np.ndarray): 3D labeled mitochondria array.
        cristae (np.ndarray): 3D labeled cristae array.

    Returns:
        np.ndarray: A labeled mitochondria array where only mitochondria containing cristae remain.
    """
    if mitochondria.shape != cristae.shape:
        raise ValueError("Mitochondria and cristae arrays must have the same shape.")
    
    # Label mitochondria if they are binary
    if mitochondria.dtype == bool or mitochondria.max() == 1:
        mitochondria = parallel.label(mitochondria, block_shape=(128, 256, 256), verbose=True)
    
    # Get unique mitochondria IDs (excluding background 0)
    mito_ids = np.unique(mitochondria)
    mito_ids = mito_ids[mito_ids != 0]

    # Keep track of mitochondria that contain cristae
    valid_mito_mask = np.zeros_like(mitochondria, dtype=bool)

    for mito_id in mito_ids:
        # Create a mask for the current mitochondrion
        mito_mask = mitochondria == mito_id
        # Check if any cristae labels overlap with this mitochondrion
        if np.any(cristae[mito_mask] > 0):
            valid_mito_mask |= mito_mask  # Retain mitochondria that contain cristae

    # Return a filtered mitochondria array, where non-overlapping mitochondria are removed
    return mitochondria * valid_mito_mask


def get_first_matching_key(d, substring):
    return next((key for key in d.keys() if substring in key), None)


def compute_statistics(files):
    all_mito_sizes = []
    all_cristae_sizes = []
    file_stats = []

    for path in files:
        data = io.load_data_from_file(path)
        
        mito_key = get_first_matching_key(data, "mito")
        seg_key = get_first_matching_key(data, "cristae_seg")
        labels_key = get_first_matching_key(data, "cristae")
        
        print(f"Processing file: {path}")
        print(f"Keys: {mito_key}, {seg_key}, {labels_key}")
        print(f"Shapes: mito={data[mito_key].shape}, cristae={data[labels_key].shape}")

        # Filter mitochondria that contain cristae
        filtered_mitos = filter_mitochondria_with_cristae(data[mito_key][1], data[labels_key])

        # Label cristae
        labeled_cristae = parallel.label(data[labels_key], block_shape=(128, 256, 256), verbose=True)
        orig_mitos = parallel.label(data[mito_key][1], block_shape=(128, 256, 256), verbose=True)
        # Get unique mitochondria (before & after filtering)
        mito_ids, mito_counts = np.unique(orig_mitos, return_counts=True)
        filtered_mito_ids, filtered_mito_counts = np.unique(filtered_mitos, return_counts=True)

        # Get unique cristae
        cristae_ids, cristae_counts = np.unique(labeled_cristae, return_counts=True)

        # Store all sizes
        all_mito_sizes.extend(filtered_mito_counts[1:])  # Exclude background (0)
        all_cristae_sizes.extend(cristae_counts[1:])  # Exclude background (0)

        # Store per-file stats
        file_stats.append({
            "file": path,
            "total_mito": len(mito_ids) - 1,  # Exclude background
            "mitos_with_cristae": len(filtered_mito_ids) - 1,
            "total_cristae": len(cristae_ids) - 1,
            "mean_mito_size": np.mean(filtered_mito_counts[1:]) if len(filtered_mito_counts) > 1 else 0,
            "mean_cristae_size": np.mean(cristae_counts[1:]) if len(cristae_counts) > 1 else 0,
        })

        print(f"Filtered mitochondria: {len(filtered_mito_ids)-1}, Total cristae: {len(cristae_ids)-1}")
        print("-" * 50)

    # Compute aggregate statistics
    total_mitos = sum(stat["filtered_mito"] for stat in file_stats)
    total_cristae = sum(stat["total_cristae"] for stat in file_stats)
    
    stats_summary = {
        "total_files": len(files),
        "total_mito": total_mitos,
        "total_cristae": total_cristae,
        "mean_mito_size": np.mean(all_mito_sizes) if all_mito_sizes else 0,
        "std_mito_size": np.std(all_mito_sizes) if all_mito_sizes else 0,
        "mean_cristae_size": np.mean(all_cristae_sizes) if all_cristae_sizes else 0,
        "std_cristae_size": np.std(all_cristae_sizes) if all_cristae_sizes else 0,
    }

    # Convert to DataFrame for better visualization
    df_stats = pd.DataFrame(file_stats)
    print("\nPer-file statistics:\n", df_stats)
    print("\nAggregated statistics:\n", stats_summary)

    return df_stats, stats_summary


def main(args):
    files = io.load_file_paths(args.path, args.ext)
    if files is None:
        print("Could not find any files")
        return
    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.join(args.output_path, "stats.csv")
    
    df_stats, stats_summary = compute_statistics(files)

    # Convert stats_summary (dict) to DataFrame with one row
    df_summary = pd.DataFrame([stats_summary])

    # Concatenate detailed stats with summary stats
    df_combined = pd.concat([df_stats, df_summary], ignore_index=True)

    # Save to CSV
    df_combined.to_csv(output_path, index=False)
    
    print(f"Statistics exported to: {output_path}")
    # for path in files:
    #     data = io.load_data_from_file(path)
    #     mito_key = get_first_matching_key(data, "mito")
    #     seg_key = get_first_matching_key(data, "cristae_seg")
    #     labels_key = get_first_matching_key(data, "cristae")
    #     print(mito_key, seg_key, labels_key)
    #     print("shape of data[mito_key], data[labels_key]", data[mito_key].shape, data[labels_key].shape)
    #     filtered_mitos = filter_mitochondria_with_cristae(data[mito_key][1], data[labels_key])
    #     labeled_cristae = parallel.label(data[labels_key], block_shape=(128, 256, 256), verbose=True)
    #     uniq, counts = np.unique(filtered_mitos, return_counts=True)
    #     print("filtered mitos: uniq, counts, len(uniq)", uniq, counts, len(uniq))
    #     uniq_cristae, cristae_counts = np.unique(labeled_cristae, return_counts=True)
    #     print("how many cristae", uniq_cristae, cristae_counts, len(uniq_cristae))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/cristae_test_segmentations/")
    parser.add_argument("--ext", "-e", type=str, default=None)
    parser.add_argument("--output_path", "-o", type=str, default="/user/freckmann15/u12103/synapse/stat_out")
    args = parser.parse_args()
    main(args)