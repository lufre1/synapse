
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import h5py
import numpy as np
from collections import defaultdict

from tqdm import tqdm


def get_uniques_from_file(h5_path, key="label_crop/all"):
    try:
        with h5py.File(h5_path, 'r') as f:
            labels = f[key]
            uniques = np.unique(labels)
        return uniques
    except Exception as e:
        print(f"Error in {h5_path}: {e}")
        return None


def get_scale_from_file(h5_path):
    try:
        with h5py.File(h5_path, 'r') as f:
            scale = f.attrs['scale']
        return scale
    except Exception as e:
        print(f"Error in {h5_path}: {e}")
        return None


def get_scale_stats(h5_paths):
    scales = []
    for path in h5_paths:
        scale = get_scale_from_file(path)
        if scale is not None:
            scales.append(scale)
            if np.any(scale == 64):
                print("path with scale of 64:", path, scale)
    if not scales:
        print("No scales found.")
        return None
    scales_arr = np.array(scales)
    stats = {}
    n_dim = scales_arr.shape[1] if scales_arr.ndim > 1 else 1
    for d in range(n_dim):
        dim_values = scales_arr[:, d] if n_dim > 1 else scales_arr
        stats[f'dim{d}_min'] = np.min(dim_values)
        stats[f'dim{d}_max'] = np.max(dim_values)
        stats[f'dim{d}_mean'] = float(np.mean(dim_values))
        stats[f'dim{d}_median'] = float(np.median(dim_values))
    # Unique scales
    unique, counts = np.unique(scales_arr, axis=0, return_counts=True)
    stats['unique_scales'] = [tuple(row) for row in unique]
    stats['unique_scale_counts'] = counts.tolist()
    stats['n_files'] = len(scales)
    return stats


def get_shape_from_file(h5_path):
    try:
        with h5py.File(h5_path, 'r') as f:
            shape = f['label_crop/all'].shape
        return shape
    except Exception as e:
        print(f"Error reading {h5_path}: {e}")
        return None


def get_labelcropall_stats(filepaths, n_workers=None):
    if n_workers is None:
        n_workers = max(1, os.cpu_count() or 1)
    shapes = []
    # Collect shapes in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_file = {executor.submit(get_shape_from_file, path): path for path in filepaths}
        for future in as_completed(future_to_file):
            result = future.result()
            if result:
                shapes.append(result)
    shapes_arr = np.array(shapes)
    stats = {}
    # Statistics per dimension
    if shapes_arr.size == 0:
        print("No valid shapes found.")
        return None
    for i in range(shapes_arr.shape[1]):
        stats[f"dim{i}_min"] = np.min(shapes_arr[:, i])
        stats[f"dim{i}_max"] = np.max(shapes_arr[:, i])
        stats[f"dim{i}_mean"] = np.mean(shapes_arr[:, i])
        stats[f"dim{i}_median"] = np.median(shapes_arr[:, i])
    # Unique shapes
    unique, counts = np.unique(shapes_arr, axis=0, return_counts=True)
    stats['unique_shapes'] = [tuple(row) for row in unique]
    stats['unique_shape_counts'] = counts.tolist()
    stats['n_files'] = len(shapes)
    return stats


def file_group_stats(path, ID_GROUPS, dataset="label_crop/all"):
    per_file = {}
    try:
        with h5py.File(path, "r") as f:
            if dataset not in f:
                return per_file
            data = f[dataset][:]
            total_voxels = np.prod(data.shape)
            for gi, group in enumerate(ID_GROUPS):
                mask = np.isin(data, group)
                n_voxels = np.count_nonzero(mask)
                found_ids = np.intersect1d(np.unique(data), group)
                n_instances = len(found_ids)
                if n_instances > 0:
                    per_file[gi] = (1, n_instances, n_voxels / total_voxels)
    except Exception as e:
        print(f"Error: {path}: {e}")
    return per_file


def parallel_group_stats_in_h5(data_paths, ID_GROUPS, dataset="label_crop/all", group_names=None, n_workers=None):
    n_files = len(data_paths)
    n_groups = len(ID_GROUPS)
    if n_workers is None:
        n_workers = max(1, os.cpu_count() or 1)
    if group_names is None:
        group_names = [f"group_{i}" for i in range(n_groups)]

    stats = defaultdict(lambda: {"n_crops_with_group": 0, "total_instance_count": 0, "total_proportion": 0.0})

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all file jobs at once
        futures = {executor.submit(file_group_stats, path, ID_GROUPS, dataset): path for path in data_paths}
        # Use tqdm to display progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            per_file = future.result()
            for gi, (present, n_instances, prop) in per_file.items():
                name = group_names[gi]
                stats[name]["n_crops_with_group"] += present
                stats[name]["total_instance_count"] += n_instances
                stats[name]["total_proportion"] += prop

    print("\n=== Per-group statistics ===")
    for i, group in enumerate(ID_GROUPS):
        name = group_names[i]
        cstat = stats[name]
        prevalence = cstat["n_crops_with_group"]
        avg_ninst = (cstat["total_instance_count"] / prevalence) if prevalence else 0
        avg_prop = (cstat["total_proportion"] / prevalence) if prevalence else 0
        print(f"{name}: Appears in {prevalence}/{n_files} crops"
              f" | Avg. #instances: {avg_ninst:.2f}"
              f" | Avg. proportion: {avg_prop:.4f}")

    return stats


def print_present_groups(data_paths, ID_GROUPS, dataset="label_crop/all", group_names=None):
    """
    For each HDF5 in data_paths, prints which ID_GROUPS are present in the dataset.
    Optionally, provide group_names (same length as ID_GROUPS) for readable output.
    """
    for path in data_paths:
        try:
            with h5py.File(path, "r") as f:
                if dataset not in f:
                    print(f"{path}: {dataset} not found.")
                    continue
                data = f[dataset][:]
                groups_present = []
                for i, group in enumerate(ID_GROUPS):
                    found_ids = np.intersect1d(np.unique(data), group)
                    if len(found_ids) > 0:
                        name = group_names[i] if group_names is not None else f"group_{i}"
                        groups_present.append((name, list(found_ids)))
                if not groups_present:
                    print(f"{path}: No group present")
                else:
                    found_str = "; ".join([f"{g}: {ids}" for g, ids in groups_present])
                    print(f"{path}: {found_str}", f"shape: {data.shape}")
        except Exception as e:
            print(f"{path}: Error - {e}")


def check_any_id_group(path, ID_GROUPS, dataset="label_crop/all"):
    try:
        with h5py.File(path, "r") as f:
            if dataset not in f:
                return None
            data = f[dataset][:]
            present = any(
                len(np.intersect1d(np.unique(data), group)) > 0
                for group in ID_GROUPS
            )
            if present:
                return path
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return None


def get_paths_with_any_id_group(data_paths, ID_GROUPS, dataset="label_crop/all", n_workers=None):
    if n_workers is None:
        n_workers = max(1, os.cpu_count() or 1)
    valid_paths = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(check_any_id_group, path, ID_GROUPS, dataset)
            for path in data_paths
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                valid_paths.append(result)
    return valid_paths


class WeightedGroupSampler:
    """
    With probability p_prioritized, sample for one rare group (weighted).
    With probability 1-p_prioritized, fallback to classic AtLeastNGroupsSampler (random).
    In either mode, if criteria are not met, accept with probability p_fallback.
    """
    def __init__(self, id_groups, group_weights, min_num_instances=1, min_size=None,
                 p_prioritized=0.75, min_num_groups=1, p_fallback=0.05):
        self.id_groups = id_groups
        self.group_weights = group_weights / np.sum(group_weights)
        self.min_num_instances = min_num_instances
        self.min_size = min_size
        self.p_prioritized = p_prioritized
        self.min_num_groups = min_num_groups
        self.p_fallback = p_fallback  # Probability to accept even if sample fails all criteria

    def _check_group(self, group, y):
        mask = np.isin(y, group)
        if self.min_size is not None:
            present_ids = [i for i in group if np.sum(y == i) >= self.min_size]
            n_found = len(present_ids)
        else:
            n_found = np.unique(y[mask]).size
        return n_found >= self.min_num_instances

    def __call__(self, x: np.ndarray, y: np.ndarray) -> bool:
        if np.random.rand() < self.p_prioritized:
            # Prioritized mode: select rare group (weighted)
            group_idx = np.random.choice(len(self.id_groups), p=self.group_weights)
            ok = self._check_group(self.id_groups[group_idx], y)
        else:
            # Classic AtLeastNGroupsSampler
            satisfied = 0
            for group in self.id_groups:
                if self._check_group(group, y):
                    satisfied += 1
            ok = (satisfied >= self.min_num_groups)
        # Final fallback logic
        if ok:
            return True
        else:
            return np.random.rand() < self.p_fallback  # Accept "bad" patch with low probability


class AtLeastNGroupsSampler:
    """
    Accepts patch if at least N distinct organelle groups have at least min_num_instances present.
    """
    def __init__(self, id_groups, min_num_groups=1, min_num_instances=1, p_reject=1.0, min_size=None):
        self.id_groups = id_groups
        self.min_num_groups = min_num_groups
        self.min_num_instances = min_num_instances
        self.p_reject = p_reject
        self.min_size = min_size

    def __call__(self, x: np.ndarray, y: np.ndarray) -> bool:
        satisfied = 0
        for group in self.id_groups:
            mask = np.isin(y, group)
            if self.min_size is not None:
                present_ids = [i for i in group if np.sum(y == i) >= self.min_size]
                n_found = len(present_ids)
            else:
                n_found = np.unique(y[mask]).size
            if n_found >= self.min_num_instances:
                satisfied += 1
        if satisfied < self.min_num_groups:
            # Not enough groups satisfied, reject
            return np.random.rand() > self.p_reject
        return True


class IDGroupsSampler:
    """
    Ensures each group of label IDs occurs at least min_num_instances times in the patch.
    """
    def __init__(self, id_groups, min_num_instances=1, p_reject=1.0, min_size=None):
        self.id_groups = id_groups
        self.min_num_instances = min_num_instances
        self.p_reject = p_reject
        self.min_size = min_size

    def __call__(self, x: np.ndarray, y: np.ndarray) -> bool:
        failed = False
        for group in self.id_groups:
            mask = np.isin(y, group)
            if self.min_size is not None:
                present_ids = [i for i in group if np.sum(y == i) >= self.min_size]
                n_found = len(present_ids)
            else:
                n_found = np.unique(y[mask]).size
            if n_found < self.min_num_instances:
                failed = True
        if failed:
            # One or more groups failed, reject with probability p_reject
            return np.random.rand() > self.p_reject
        return True


def get_resized_cellmap_paths(organelle_size="medium"):
    if organelle_size == "medium":
        return [
            '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_177.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_139.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_189.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_226.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_369.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_173.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_133.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_174.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_276.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_242.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_28.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_172.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_146.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_111.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_144.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_239.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_320.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_187.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_280.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_266.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_203.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_417.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_191.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_134.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_268.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_212.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_275.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_210.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_40.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_273.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_249.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_220.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_201.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_341.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_47.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_188.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_143.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_125.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_48.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_113.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_186.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_292.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_3.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_323.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_278.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_145.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_238.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_206.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_277.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_377.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_120.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_171.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_240.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_4.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_267.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_37.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_199.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_38.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_159.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_131.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_225.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_32.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_8.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_342.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_116.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_115.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_183.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_272.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_245.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_160.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_259.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_49.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_190.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_42.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_184.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_256.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_180.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_193.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_200.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_230.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_15.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_20.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_147.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_376.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_151.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_101.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_33.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_14.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_221.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_118.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_217.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_78.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_110.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_13.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_132.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_27.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_218.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_117.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_80.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_214.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_178.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_39.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_79.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_155.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_208.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_162.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_326.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_202.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_252.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_163.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_36.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_192.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_198.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_368.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_157.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_175.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_325.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_258.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_107.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_216.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_9.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_321.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_166.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_279.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_237.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_43.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_126.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_6.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_124.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_322.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_222.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_31.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_270.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_333.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_274.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_109.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_195.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_138.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_150.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_129.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_197.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_121.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_182.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_228.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_130.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_235.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_140.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_19.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_229.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_22.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_35.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_291.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_196.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_219.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_16.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_122.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_141.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_112.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_158.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_370.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_119.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_241.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_416.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_231.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_51.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_236.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_340.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_213.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_255.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_34.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_176.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_142.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_148.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_248.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_227.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_1.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_123.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_209.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_211.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_185.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_23.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_137.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_269.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_165.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_50.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_313.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_234.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_136.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_161.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_156.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_319.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_18.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_164.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_298.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_135.h5', '/scratch-grete/projects/nim00007/data/cellmap/resized_crops/crop_181.h5'
        ]


def get_cellmap_mito_paths():
    
    return [
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_1.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_100.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_101.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_107.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_109.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_110.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_111.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_112.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_113.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_115.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_116.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_117.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_118.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_119.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_120.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_121.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_122.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_123.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_124.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_125.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_126.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_129.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_13.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_130.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_131.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_132.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_133.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_134.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_135.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_136.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_137.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_138.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_139.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_14.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_140.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_141.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_142.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_143.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_144.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_145.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_146.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_147.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_148.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_149.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_15.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_150.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_151.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_155.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_156.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_157.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_158.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_159.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_16.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_160.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_161.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_162.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_163.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_164.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_165.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_166.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_171.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_172.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_173.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_174.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_175.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_176.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_177.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_178.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_18.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_180.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_181.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_182.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_183.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_184.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_185.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_186.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_187.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_188.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_189.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_19.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_190.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_191.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_192.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_193.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_195.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_196.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_197.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_198.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_199.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_20.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_200.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_201.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_202.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_203.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_206.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_208.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_209.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_210.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_211.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_212.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_213.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_214.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_216.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_217.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_218.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_219.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_22.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_220.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_221.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_222.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_224.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_225.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_226.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_227.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_228.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_229.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_23.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_230.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_231.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_234.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_235.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_236.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_237.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_238.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_239.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_240.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_241.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_242.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_243.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_245.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_248.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_249.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_25.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_252.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_255.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_256.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_258.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_259.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_26.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_266.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_267.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_268.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_269.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_27.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_270.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_272.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_273.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_274.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_275.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_276.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_277.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_278.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_279.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_28.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_280.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_291.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_292.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_298.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_3.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_31.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_313.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_319.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_32.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_320.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_321.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_322.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_323.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_325.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_326.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_33.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_333.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_34.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_340.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_341.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_342.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_345.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_346.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_35.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_355.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_356.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_36.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_368.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_369.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_37.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_370.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_376.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_377.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_38.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_380.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_381.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_39.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_4.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_40.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_416.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_417.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_42.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_43.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_47.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_48.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_49.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_50.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_51.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_54.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_55.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_56.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_57.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_58.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_59.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_6.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_60.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_61.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_62.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_63.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_64.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_65.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_66.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_67.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_68.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_69.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_7.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_70.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_71.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_72.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_73.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_74.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_75.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_76.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_77.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_78.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_79.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_8.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_80.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_81.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_82.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_83.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_84.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_85.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_86.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_87.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_88.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_89.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_9.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_90.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_91.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_92.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_93.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_94.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_95.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_96.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_97.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_98.h5",
        "/scratch-grete/projects/nim00007/data/cellmap/data_crops/crop_99.h5",

    ]
    