import argparse
import numpy as np
from elf.io import open_file
from tifffile import imread
from tqdm import tqdm
import csv


def export_results_to_csv(results_dict, csv_path):
    """
    Export the morphology results dictionary to a CSV file.

    Parameters:
        results_dict (dict): Dictionary of mitochondria feature dicts (from analyze_mito_morphology).
        csv_path (str): File path to save the CSV.
    """
    if not results_dict:
        raise ValueError("Empty results dictionary.")

    # Collect all fieldnames (keys) from the first mitochondrial entry
    fieldnames = list(next(iter(results_dict.values())).keys())

    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for mitochondrion in results_dict.values():
            writer.writerow(mitochondrion)
    print(f"Finished writing CSV file to {csv_path}")


def analyze_mito_morphology(raw_img, seg_img, voxel_size):
    """
    Analyze morphology and intensity features for each mitochondrion in a segmentation, accounting for anisotropic voxel size.

    Parameters:
        raw_img (np.ndarray): The raw image (intensity values).
        seg_img (np.ndarray): The segmentation (integer labels; 0=background).
        voxel_size (dict): Physical voxel size scaling, e.g. {'z': float, 'y': float, 'x': float}, units in microns.

    Returns:
        dict: For each mitochondrion label, a sub-dictionary containing:
            - x, y, z (centroid, in pixels)
            - volume (in cubic microns)
            - intensity_max, intensity_mean, intensity_min, intensity_std
            - elongation (ratio of largest to smallest eigenvalue of covariance, scaled for anisotropy)
            - index (enumerated order)
    """
    labels = np.unique(seg_img)
    labels = labels[labels != 0]  # exclude background label 0
    results = {}
    scale = np.array([voxel_size['z'], voxel_size['y'], voxel_size['x']])
    voxel_volume = np.prod(scale)  # physical volume of one voxel
    for i, label in enumerate(tqdm(labels, desc="Analyzing morphology and intensity features")):
        mask = seg_img == label
        coords = np.argwhere(mask)

        # Scale coordinates by voxel size for physical distances
        scaled_coords = coords * scale

        centroid = coords.mean(axis=0)
        z, y, x = centroid
        voxel_count = mask.sum()
        pixel_values = raw_img[mask]

        cov = np.cov(scaled_coords, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)

        if np.any(eigenvalues <= 0):
            elongation = 1.0
        else:
            elongation = float(eigenvalues[-1] / eigenvalues[0])  # largest / smallest eigenvalue

        results[int(label)] = {
            'label': int(label),
            'x': int(np.round(x)),
            'y': int(np.round(y)),
            'z': int(np.round(z)),
            'volume (microns^3)': float(voxel_count * voxel_volume),
            'intensity_max': float(pixel_values.max()),
            'intensity_mean': float(pixel_values.mean()),
            'intensity_min': float(pixel_values.min()),
            'intensity_std': float(pixel_values.std()),
            'elongation': elongation,
            'index': i + 1
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default="/home/freckmann15/data/mitochondria/volume-em/embl/paper/cutout_1_out_mitonet_reordered.h5")
    parser.add_argument("--label_path", "-lp", type=str, default="/home/freckmann15/data/mitochondria/volume-em/embl/cutout_1_committed_objects_leonie_2025-08-07.tif")
    parser.add_argument("--output_path", "-o", type=str, default="/home/freckmann15/data/mitochondria/volume-em/embl/paper/cutout_1_ground_truth_mophology.csv")
    parser.add_argument("--voxel_size", "-v", type=dict, default={'z': 0.0025, 'y': 0.0005, 'x': 0.0005})
    
    args = parser.parse_args()
    
    raw_img, label = None, None
    with open_file(args.path, 'r') as f:
        for key in f.keys():
            if 'raw' in key:
                raw_img = f[key][:]
            elif 'label' in key:
                label = f[key][:]
    if label is None:
        label = imread(args.label_path)
    assert raw_img is not None and label is not None
    assert raw_img.shape == label.shape
    
    results = analyze_mito_morphology(raw_img, label, args.voxel_size)
    print(results)
    export_results_to_csv(results, args.output_path)


if __name__ == "__main__":
    main()