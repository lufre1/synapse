import argparse
import numpy as np
from elf.io import open_file
from tifffile import imread
from tqdm import tqdm
import csv
from scipy.spatial import cKDTree
from skimage.measure import euler_number
from skimage.measure import marching_cubes
from skimage.morphology import skeletonize


def _skeleton_metrics(obj, spacing_zyx):
    skel = skeletonize(obj)
    # Skeleton params; approximate length counting voxels weighted by spacing
    voxel_distances = np.linalg.norm(spacing_zyx)
    length = skel.sum() * voxel_distances

    # For branches and endpoints, approximate using endpoints of skeleton voxels
    # (This is simplified; more complex skeleton graph analysis may be needed)
    endpoints = 0
    branches = 0
    # Placeholder: detailed branch/endpoints require graph analysis
    # For now, return 0 (or implement graph analysis)
    return {
        'skeleton_length_um': length,
        'skeleton_branches': branches,
        'skeleton_endpoints': endpoints
    }


def _surface_area_um2(obj, spacing_zyx):
    verts, faces, _, _ = marching_cubes(obj, spacing=spacing_zyx)
    # Compute area via mesh triangles

    def triangle_area(v0, v1, v2):
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    surface_area = sum(triangle_area(verts[f[0]], verts[f[1]], verts[f[2]]) for f in faces)
    return surface_area


def _bbox_from_coords(coords, shape, margin=1):
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    starts = np.maximum(mins - margin, 0)
    ends = np.minimum(maxs + margin, shape)
    z_slice = slice(starts[0], ends[0])
    y_slice = slice(starts[1], ends[1])
    x_slice = slice(starts[2], ends[2])
    return z_slice, y_slice, x_slice


def _principal_axes_um(scaled_coords):
    cov = np.cov(scaled_coords, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]
    # Axis lengths proportional to sqrt of eigenvalues
    semi_axes = np.sqrt(np.maximum(sorted_eigenvalues, 0))
    a, b, c = semi_axes

    axis_metrics = {
        'elongation_a_over_c': a / c if c > 0 else np.nan,
        'flatness_b_over_c': b / c if c > 0 else np.nan,
        'isotropy_c_over_a': c / a if a > 0 else np.nan
    }
    return (a, b, c), axis_metrics


def _sphericity(volume_um3, surface_um2):
    if surface_um2 == 0 or np.isnan(surface_um2) or volume_um3 == 0 or np.isnan(volume_um3):
        return np.nan
    return (np.pi**(1/3)) * ((6 * volume_um3)**(2/3)) / surface_um2


def export_results_to_csv(results_dict, export_path_base):
    """
    Export mitochondrial morphology results and optional summary to CSV files.

    Parameters:
        results_dict (dict): Dictionary with mitochondria feature dicts and optional '_summary' key.
        export_path_base (str): Base file path without extension to save CSV files.
                                Writes '<export_path_base>.csv' and optionally '<export_path_base>_summary.csv'.
    """
    if not results_dict:
        raise ValueError("Empty results dictionary.")

    # Export mitochondria results excluding summary
    per_mito_data = {k: v for k, v in results_dict.items() if k != '_summary'}
    if not per_mito_data:
        raise ValueError("No mitochondria data to write.")

    mito_csv_path = export_path_base
    if not mito_csv_path.endswith('.csv'):
        mito_csv_path += ".csv"
    fieldnames = list(next(iter(per_mito_data.values())).keys())

    with open(mito_csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for mito_dict in per_mito_data.values():
            writer.writerow(mito_dict)
    print(f"Saved mitochondria data to {mito_csv_path}")

    # Export summary if present
    summary = results_dict.get('_summary')
    if summary:
        summary_csv_path = export_path_base + "_summary.csv"
        with open(summary_csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            for key, value in summary.items():
                writer.writerow([key, value])
        print(f"Saved summary data to {summary_csv_path}")


def analyze_mitochondria_phenotypes(
    raw_img, seg_img, voxel_size,
    compute_surface=True,
    compute_skeleton=True
):
    """
    Analyze the 3D morphology, intensity, and spatial organization of mitochondria from volumetric image data and segmentation.

    Parameters:
        raw_img (np.ndarray): 3D array of image intensities (same shape as seg_img).
        seg_img (np.ndarray): 3D array of integer labels (0 = background, positive integers = mitochondria).
        voxel_size (dict): Mapping with keys 'z', 'y', 'x' specifying the physical size of one voxel in microns.
        compute_surface (bool): If True, computes surface area, surface/volume ratio, and sphericity (requires marching cubes). Default: True.
        compute_skeleton (bool): If True, computes skeleton length, branch count, and endpoint count. Default: True.

    Returns:
        dict: Mapping from label ID to mitochondrion feature dictionary, including:
            - centroid_px (z, y, x), centroid_um (z, y, x)
            - volume_um3, (optional) surface_um2, sv_ratio_um_inv, sphericity
            - principal semi-axes (a_um, b_um, c_um), elongation a/c, flatness b/c, isotropy c/a
            - euler_char
            - (optional) skeleton_length_um, skeleton_branches, skeleton_endpoints
            - touches_border
            - intensity_max, intensity_mean, intensity_min, intensity_std (AU)
            - nearest_neighbor_um (filled after per-object loop)
            - index (1-based order)
        Includes a special key '_summary' for dataset-level metrics:
            - count, analysis_volume_um3, density_per_um3, nnd_mean_um, nnd_median_um, voxel_size_um
    """

    # Helper functions (should be defined/imported elsewhere in your codebase)
    # _principal_axes_um, _bbox_from_coords, euler_number, _surface_area_um2, _sphericity, _skeleton_metrics

    labels = np.unique(seg_img)
    labels = labels[labels != 0]
    results = {}

    vz, vy, vx = voxel_size['z'], voxel_size['y'], voxel_size['x']
    scale = np.array([vz, vy, vx], dtype=float)
    voxel_volume = float(vz * vy * vx)

    centroids_um = []
    label_list = []

    for i, label in enumerate(tqdm(labels, desc="Analyzing mitochondria")):
        mask = (seg_img == label)
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue

        scaled_coords = coords.astype(float) * scale

        centroid_px = coords.mean(axis=0)
        centroid_um = centroid_px * scale
        zc, yc, xc = centroid_px

        voxel_count = int(mask.sum())
        volume_um3 = float(voxel_count * voxel_volume)

        pv = raw_img[mask]
        intensity_max = float(pv.max())
        intensity_mean = float(pv.mean())
        intensity_min = float(pv.min())
        intensity_std = float(pv.std())

        semi_axes_um, axis_metrics = _principal_axes_um(scaled_coords)
        a_um, b_um, c_um = semi_axes_um if np.all(np.isfinite(semi_axes_um)) else (np.nan, np.nan, np.nan)

        touches_border = (
            (coords[:, 0].min() == 0) or (coords[:, 0].max() == seg_img.shape[0] - 1) or
            (coords[:, 1].min() == 0) or (coords[:, 1].max() == seg_img.shape[1] - 1) or
            (coords[:, 2].min() == 0) or (coords[:, 2].max() == seg_img.shape[2] - 1)
        )

        zsl, ysl, xsl = _bbox_from_coords(coords, seg_img.shape, margin=1)
        sub = seg_img[zsl, ysl, xsl]
        obj = (sub == label)

        # Euler characteristic (3D)
        try:
            euler_char = int(euler_number(obj, connectivity=3))
        except Exception:
            euler_char = 0

        surface_um2 = np.nan
        sv_ratio = np.nan
        sphericity = np.nan
        if compute_surface and obj.sum() > 0:
            try:
                surface_um2 = _surface_area_um2(obj, spacing_zyx=(vz, vy, vx))
                if surface_um2 > 0:
                    sv_ratio = float(surface_um2 / volume_um3)
                    sphericity = _sphericity(volume_um3, surface_um2)
            except Exception:
                pass

        skel_len = 0.0
        skel_br = 0
        skel_ep = 0
        if compute_skeleton:
            sk = _skeleton_metrics(obj, spacing_zyx=(vz, vy, vx))
            skel_len = sk['skeleton_length_um']
            skel_br = sk['skeleton_branches']
            skel_ep = sk['skeleton_endpoints']

        res = {
            'label': int(label),
            'index': i + 1,
            'centroid_px_z': float(zc), 'centroid_px_y': float(yc), 'centroid_px_x': float(xc),
            'centroid_um_z': float(centroid_um[0]), 'centroid_um_y': float(centroid_um[1]), 'centroid_um_x': float(centroid_um[2]),
            'volume_um3': volume_um3,
            'surface_um2': surface_um2,
            'sv_ratio': sv_ratio,
            'sphericity': sphericity,
            'a_um': float(a_um), 'b_um': float(b_um), 'c_um': float(c_um),
            'elongation_a_over_c': axis_metrics['elongation_a_over_c'],
            'flatness_b_over_c': axis_metrics['flatness_b_over_c'],
            'isotropy_c_over_a': axis_metrics['isotropy_c_over_a'],
            'euler_char': euler_char,
            'skeleton_length_um': float(skel_len),
            'skeleton_branches': int(skel_br),
            'skeleton_endpoints': int(skel_ep),
            'touches_border': bool(touches_border),
            'intensity_max': intensity_max,
            'intensity_mean': intensity_mean,
            'intensity_min': intensity_min,
            'intensity_std': intensity_std,
            'nearest_neighbor_um': np.nan
        }
        results[int(label)] = res

        centroids_um.append(centroid_um)
        label_list.append(int(label))

    # Spatial organization: nearest-neighbor distances (in microns)
    if len(centroids_um) >= 2:
        pts = np.vstack(centroids_um)
        tree = cKDTree(pts)
        dists, idxs = tree.query(pts, k=2)
        nn = dists[:, 1]
        for lab, d in zip(label_list, nn):
            results[lab]['nearest_neighbor_um'] = float(d)

    # Global density and summary
    analysis_volume_um3 = float(seg_img.shape[0] * vz * seg_img.shape[1] * vy * seg_img.shape[2] * vx)
    count = len(label_list)
    density = float(count / analysis_volume_um3) if analysis_volume_um3 > 0 else np.nan
    nn_vals = np.array([results[lab]['nearest_neighbor_um'] for lab in label_list if np.isfinite(results[lab]['nearest_neighbor_um'])])
    nnd_mean = float(np.mean(nn_vals)) if nn_vals.size > 0 else np.nan
    nnd_median = float(np.median(nn_vals)) if nn_vals.size > 0 else np.nan

    results['_summary'] = {
        'count': count,
        'analysis_volume_um3': analysis_volume_um3,
        'density_per_um3': density,
        'nnd_mean_um': nnd_mean,
        'nnd_median_um': nnd_median,
        'voxel_size_um': {'z': vz, 'y': vy, 'x': vx}
    }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=str, default="/home/freckmann15/data/mitochondria/volume-em/embl/paper/cutout_1_out_mitonet_reordered.h5")
    parser.add_argument("--label_path", "-lp", type=str, default="/home/freckmann15/data/mitochondria/volume-em/embl/cutout_1_committed_objects_leonie_2025-08-07.tif")
    parser.add_argument("--output_path", "-o", type=str, default="/home/freckmann15/data/mitochondria/volume-em/embl/paper/cutout_1_ground_truth_phenotypes.csv")
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
    
    # results = analyze_mito_morphology(raw_img, label, args.voxel_size)
    results = analyze_mitochondria_phenotypes(raw_img, label, args.voxel_size)
    print(results)
    export_results_to_csv(results, args.output_path)


if __name__ == "__main__":
    main()