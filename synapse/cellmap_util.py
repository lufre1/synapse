
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import chain
import os
import h5py
import numpy as np
from collections import defaultdict
import multiprocessing as mp
from elf.parallel import label

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
        avail_workers = os.cpu_count()
        if avail_workers and avail_workers >= 32:
            n_workers = 32
        elif avail_workers >= 16:
            n_workers = 16
        elif avail_workers >= 8:
            n_workers = 8
        elif avail_workers >= 4:
            n_workers = 4
        else:
            n_workers = 1
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


def check_if_only_foreground(path):
    try:
        with h5py.File(path, "r") as f:
            if "label_crop/all" in f.keys():
                data = f["label_crop/all"][:]
                if np.all(data == 0) or np.all(data == 1) or len(np.unique(data)) <= 1:
                    return None
    except Exception as e:
        print(f"Error: {path}: {e}")
    return path


def filter_paths_for_only_foreground_parallel(data_paths, dataset="label_crop/all", n_workers=8):
    print("Using", n_workers, "workers for parallel processing")
    valid_paths = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_file = {executor.submit(check_if_only_foreground, path): path for path in data_paths}
        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc="Processing files"):
            try:
                result = future.result()
                if result is not None:
                    valid_paths.append(result)
            except Exception as e:
                print(f"Error processing file {future_to_file[future]}: {e}")
    return valid_paths


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


def check_any_id_group(path, ID_GROUPS, dataset="label_crop/all", min_pct_slices=0):
    try:
        with h5py.File(path, "r") as f:
            if dataset not in f:
                return None
            data = f[dataset][:]
            if min_pct_slices == 0:
                uniq = np.unique(data)
            # data.shape = (z, y, x) assumed
            for group in ID_GROUPS:
                # no percentages
                if min_pct_slices == 0:
                    if any(label in group for label in uniq):
                        return path
                else:
                    z_contain = []
                    for z in range(data.shape[0]):
                        # unique labels in this slice
                        labels_in_slice = np.unique(data[z])
                        if any(label in group for label in labels_in_slice):
                            z_contain.append(1)
                        else:
                            z_contain.append(0)
                    fraction = sum(z_contain) / data.shape[0]
                    if fraction > min_pct_slices:
                        return path
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return None


def get_paths_with_any_id_group(data_paths, ID_GROUPS, dataset="label_crop/all", n_workers=None, min_pct_slices=0):
    # breakpoint()
    if n_workers is None:
        avail_workers = os.cpu_count()
        if avail_workers and avail_workers >= 32:
            n_workers = 32
        elif avail_workers >= 16:
            n_workers = 16
        elif avail_workers >= 8:
            n_workers = 8
        elif avail_workers >= 4:
            n_workers = 4
        else:
            n_workers = 1
    n_threads = mp.cpu_count()
    print("mp.cpu_count()", n_threads)
    print("Using", n_workers, "workers for parallel processing")
    valid_paths = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                check_any_id_group, path, ID_GROUPS, dataset, min_pct_slices
            )
            for path in data_paths
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
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
    def __init__(self, id_groups, min_num_groups=1, p_reject=1.0, min_size=None):
        """
        Args:
            id_groups: A list of lists of label IDs.
            min_num_groups: The minimum number of groups that must be present in a patch.
            min_num_instances: The minimum number of instances of each group that must be present in a patch.
            p_reject: The probability of rejecting a patch that does not meet the criteria.
            min_size: The minimum size of an instance of a group.
        """
        self.id_groups = id_groups
        self.min_num_groups = min_num_groups
        self.p_reject = p_reject
        self.min_size = min_size

    def __call__(self, x: np.ndarray, y: np.ndarray) -> bool:
        # Identify labels that are considered background
        background_mask = (y == 0) | ~(np.isin(y, [label for group in self.id_groups for label in group]))
        # Check if there is at least one pixel labeled as background
        has_background = np.any(background_mask)
        if not has_background:
            # Reject batch if there is no background
            return np.random.rand() > self.p_reject
        segmentation_ids = np.unique(y)
        if len(segmentation_ids) == 1:
            # If there is only one label ID or it's just background, reject
            return np.random.rand() > self.p_reject
        satisfied_groups = 0

        for group in self.id_groups:
            mask = np.isin(y, group)
            masked_y = np.where(mask, y, 0)
            if self.min_size is not None:
                labels = label(masked_y, block_shape=(1, 256, 256))
                uniqs, counts = np.unique(labels, return_counts=True)
                filtered = [uniq for uniq, count in zip(uniqs, counts) if uniq != 0 and count >= self.min_size]
                n_found = len(filtered)
                # present_ids = [i for i in group if np.sum(y == i) >= self.min_size]
                # n_found = len(present_ids)
            else:
                labels = label(masked_y, block_shape=(1, 256, 256))
                n_found = np.unique(labels).size - 1
            if n_found > 0:
                satisfied_groups += 1
                if satisfied_groups >= self.min_num_groups:
                    return True
        if len(self.id_groups) == 1 and np.sum(y == 0) == 0:
            # If there is only one group and no background, reject
            return np.random.rand() > self.p_reject
        if satisfied_groups >= self.min_num_groups:
            return True
        return np.random.rand() > self.p_reject


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


def get_cellmap_paths_without_nuclei():
    return


def get_cellmap_paths_min_2_groups():
    return [
        '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_101.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_116.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_120.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_121.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_117.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_122.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_14.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_107.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_109.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_13.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_139.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_140.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_119.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_118.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_143.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_123.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_15.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_144.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_142.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_126.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_130.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_147.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_146.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_150.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_102.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_149.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_134.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_148.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_157.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_131.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_162.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_151.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_160.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_161.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_166.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_163.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_141.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_165.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_1.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_16.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_159.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_158.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_100.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_174.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_164.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_18.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_176.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_178.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_181.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_180.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_19.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_182.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_183.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_111.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_110.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_171.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_173.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_115.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_185.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_125.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_124.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_20.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_138.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_112.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_137.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_133.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_113.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_136.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_187.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_135.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_196.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_189.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_186.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_132.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_188.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_195.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_22.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_191.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_172.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_155.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_193.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_198.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_192.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_190.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_197.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_200.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_206.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_199.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_209.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_201.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_175.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_203.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_234.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_202.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_236.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_214.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_235.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_239.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_241.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_212.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_237.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_240.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_23.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_217.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_219.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_245.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_242.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_255.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_249.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_208.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_210.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_256.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_248.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_252.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_213.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_259.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_222.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_258.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_224.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_27.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_220.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_216.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_227.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_28.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_228.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_291.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_218.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_226.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_225.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_292.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_31.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_32.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_272.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_267.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_273.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_268.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_269.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_270.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_129.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_279.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_275.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_276.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_266.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_33.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_278.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_274.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_34.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_298.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_313.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_280.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_277.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_320.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_3.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_322.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_35.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_321.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_156.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_36.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_145.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_323.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_319.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_325.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_326.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_37.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_333.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_341.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_342.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_340.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_38.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_368.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_370.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_376.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_369.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_39.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_40.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_377.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_4.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_229.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_247.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_221.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_184.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_416.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_417.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_42.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_257.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_238.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_43.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_47.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_230.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_177.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_346.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_48.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_49.h5', 
        '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_51.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_50.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_231.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_6.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_7.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_356.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_8.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_80.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_79.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_211.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_345.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_78.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_9.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_410.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_386.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_411.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_355.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_387.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_412.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_413.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_472.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_380.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_381.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_421.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_473.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_423.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_407.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_408.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_362.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_347.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_329.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_348.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_324.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_366.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_379.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_378.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_354.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_353.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_351.h5'
    ]


def get_cellmap_paths_without_cell_and_nuclei():
    return [
        '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_34.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_182.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_208.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_118.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_14.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_124.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_80.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_122.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_270.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_138.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_143.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_7.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_156.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_214.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_28.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_144.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_151.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_198.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_269.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_75.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_157.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_166.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_50.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_107.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_266.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_195.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_225.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_56.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_237.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_278.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_188.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_218.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_189.h5',
        # '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_26.h5',
        '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_258.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_410.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_35.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_1.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_321.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_48.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_183.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_119.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_386.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_51.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_248.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_177.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_342.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_380.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_6.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_62.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_227.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_19.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_313.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_423.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_184.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_192.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_190.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_43.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_228.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_416.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_280.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_110.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_136.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_272.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_217.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_407.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_60.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_32.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_112.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_61.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_81.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_37.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_176.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_161.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_292.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_298.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_111.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_49.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_259.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_77.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_191.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_13.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_147.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_413.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_230.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_148.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_347.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_142.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_132.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_141.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_216.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_57.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_235.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_67.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_348.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_252.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_64.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_59.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_220.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_178.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_100.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_231.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_120.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_123.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_33.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_268.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_134.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_180.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_145.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_71.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_219.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_135.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_238.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_73.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_54.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_68.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_42.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_411.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_267.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_255.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_15.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_126.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_345.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_27.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_173.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_193.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_109.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_346.h5',
        # '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_243.h5',
        '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_412.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_47.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_206.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_362.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_115.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_140.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_213.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_202.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_234.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_368.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_322.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_70.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_323.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_158.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_210.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_275.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_4.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_16.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_370.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_9.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_171.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_247.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_200.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_38.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_325.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_55.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_172.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_199.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_36.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_279.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_209.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_242.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_341.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_116.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_353.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_421.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_381.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_149.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_162.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_175.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_185.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_249.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_408.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_65.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_102.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_212.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_133.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_276.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_222.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_329.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_79.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_130.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_257.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_196.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_376.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_473.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_131.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_159.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_40.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_229.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_417.h5',
        # '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_25.h5',
        '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_125.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_164.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_378.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_273.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_187.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_277.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_78.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_58.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_355.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_224.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_31.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_113.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_63.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_117.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_324.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_84.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_20.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_23.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_39.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_236.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_239.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_160.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_256.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_129.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_137.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_76.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_150.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_320.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_121.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_186.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_221.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_181.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_226.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_139.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_101.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_66.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_74.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_18.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_377.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_69.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_356.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_22.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_241.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_379.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_319.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_165.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_326.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_163.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_240.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_291.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_211.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_8.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_333.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_245.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_146.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_369.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_83.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_387.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_72.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_174.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_3.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_274.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_203.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_472.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_155.h5',
        # '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_82.h5',
        '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_197.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_340.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_201.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_354.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_366.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_351.h5'

    ]


def get_cellmaps_paths_fully_annotated(data_paths=None):
    if data_paths is None:
        print("Returning resized crop paths")
        return ['/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_34.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_182.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_124.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_118.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_14.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_80.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_208.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_122.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_138.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_143.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_270.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_7.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_156.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_144.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_28.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_151.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_198.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_269.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_75.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_157.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_166.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_50.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_107.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_266.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_195.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_225.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_56.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_237.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_278.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_188.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_218.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_86.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_189.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_26.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_258.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_35.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_1.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_321.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_48.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_183.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_119.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_51.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_248.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_342.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_88.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_6.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_87.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_62.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_227.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_19.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_313.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_192.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_190.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_43.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_228.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_416.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_280.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_110.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_136.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_272.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_217.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_112.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_60.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_32.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_61.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_81.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_37.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_298.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_176.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_161.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_292.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_111.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_49.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_90.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_91.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_259.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_77.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_13.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_147.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_148.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_95.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_142.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_132.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_85.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_141.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_216.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_57.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_235.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_67.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_92.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_252.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_64.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_59.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_220.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_178.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_120.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_123.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_33.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_268.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_134.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_180.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_145.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_71.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_219.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_135.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_73.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_54.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_68.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_42.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_267.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_255.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_15.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_126.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_27.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_173.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_193.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_109.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_243.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_47.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_206.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_115.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_140.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_213.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_202.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_97.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_234.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_368.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_322.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_70.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_323.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_98.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_158.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_210.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_275.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_4.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_16.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_370.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_9.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_171.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_200.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_38.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_94.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_55.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_325.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_199.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_36.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_172.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_279.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_242.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_209.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_116.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_341.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_149.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_162.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_175.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_185.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_249.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_65.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_212.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_133.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_276.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_222.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_96.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_79.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_130.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_376.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_131.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_159.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_99.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_40.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_417.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_25.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_125.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_164.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_273.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_187.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_277.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_78.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_58.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_224.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_31.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_113.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_63.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_117.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_84.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_20.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_23.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_39.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_236.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_239.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_160.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_256.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_129.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_137.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_76.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_150.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_121.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_186.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_181.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_226.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_139.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_101.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_66.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_74.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_18.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_377.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_69.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_22.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_241.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_165.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_326.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_163.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_240.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_291.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_8.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_333.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_245.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_146.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_369.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_83.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_72.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_89.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_93.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_174.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_3.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_274.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_203.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_155.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_197.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_340.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_201.h5']
    else:
        # filter data_paths to exclude following crops
        valid_crops = ['crop_34.h5', 'crop_182.h5', 'crop_124.h5', 'crop_118.h5', 'crop_14.h5', 'crop_80.h5', 'crop_208.h5', 'crop_122.h5', 'crop_138.h5', 'crop_143.h5', 'crop_270.h5', 'crop_7.h5', 'crop_156.h5', 'crop_144.h5', 'crop_28.h5', 'crop_151.h5', 'crop_198.h5', 'crop_269.h5', 'crop_75.h5', 'crop_157.h5', 'crop_166.h5', 'crop_50.h5', 'crop_107.h5', 'crop_266.h5', 'crop_195.h5', 'crop_225.h5', 'crop_56.h5', 'crop_237.h5', 'crop_278.h5', 'crop_188.h5', 'crop_218.h5', 'crop_86.h5', 'crop_189.h5', 'crop_26.h5', 'crop_258.h5', 'crop_35.h5', 'crop_1.h5', 'crop_321.h5', 'crop_48.h5', 'crop_183.h5', 'crop_119.h5', 'crop_51.h5', 'crop_248.h5', 'crop_342.h5', 'crop_88.h5', 'crop_6.h5', 'crop_87.h5', 'crop_62.h5', 'crop_227.h5', 'crop_19.h5', 'crop_313.h5', 'crop_192.h5', 'crop_190.h5', 'crop_43.h5', 'crop_228.h5', 'crop_416.h5', 'crop_280.h5', 'crop_110.h5', 'crop_136.h5', 'crop_272.h5', 'crop_217.h5', 'crop_112.h5', 'crop_60.h5', 'crop_32.h5', 'crop_61.h5', 'crop_81.h5', 'crop_37.h5', 'crop_298.h5', 'crop_176.h5', 'crop_161.h5', 'crop_292.h5', 'crop_111.h5', 'crop_49.h5', 'crop_90.h5', 'crop_91.h5', 'crop_259.h5', 'crop_77.h5', 'crop_13.h5', 'crop_147.h5', 'crop_148.h5', 'crop_95.h5', 'crop_142.h5', 'crop_132.h5', 'crop_85.h5', 'crop_141.h5', 'crop_216.h5', 'crop_57.h5', 'crop_235.h5', 'crop_67.h5', 'crop_92.h5', 'crop_252.h5', 'crop_64.h5', 'crop_59.h5', 'crop_220.h5', 'crop_178.h5', 'crop_120.h5', 'crop_123.h5', 'crop_33.h5', 'crop_268.h5', 'crop_134.h5', 'crop_180.h5', 'crop_145.h5', 'crop_71.h5', 'crop_219.h5', 'crop_135.h5', 'crop_73.h5', 'crop_54.h5', 'crop_68.h5', 'crop_42.h5', 'crop_267.h5', 'crop_255.h5', 'crop_15.h5', 'crop_126.h5', 'crop_27.h5', 'crop_173.h5', 'crop_193.h5', 'crop_109.h5', 'crop_243.h5', 'crop_47.h5', 'crop_206.h5', 'crop_115.h5', 'crop_140.h5', 'crop_213.h5', 'crop_202.h5', 'crop_97.h5', 'crop_234.h5', 'crop_368.h5', 'crop_322.h5', 'crop_70.h5', 'crop_323.h5', 'crop_98.h5', 'crop_158.h5', 'crop_210.h5', 'crop_275.h5', 'crop_4.h5', 'crop_16.h5', 'crop_370.h5', 'crop_9.h5', 'crop_171.h5', 'crop_200.h5', 'crop_38.h5', 'crop_94.h5', 'crop_55.h5', 'crop_325.h5', 'crop_199.h5', 'crop_36.h5', 'crop_172.h5', 'crop_279.h5', 'crop_242.h5', 'crop_209.h5', 'crop_116.h5', 'crop_341.h5', 'crop_149.h5', 'crop_162.h5', 'crop_175.h5', 'crop_185.h5', 'crop_249.h5', 'crop_65.h5', 'crop_212.h5', 'crop_133.h5', 'crop_276.h5', 'crop_222.h5', 'crop_96.h5', 'crop_79.h5', 'crop_130.h5', 'crop_376.h5', 'crop_131.h5', 'crop_159.h5', 'crop_99.h5', 'crop_40.h5', 'crop_417.h5', 'crop_25.h5', 'crop_125.h5', 'crop_164.h5', 'crop_273.h5', 'crop_187.h5', 'crop_277.h5', 'crop_78.h5', 'crop_58.h5', 'crop_224.h5', 'crop_31.h5', 'crop_113.h5', 'crop_63.h5', 'crop_117.h5', 'crop_84.h5', 'crop_20.h5', 'crop_23.h5', 'crop_39.h5', 'crop_236.h5', 'crop_239.h5', 'crop_160.h5', 'crop_256.h5', 'crop_129.h5', 'crop_137.h5', 'crop_76.h5', 'crop_150.h5', 'crop_121.h5', 'crop_186.h5', 'crop_181.h5', 'crop_226.h5', 'crop_139.h5', 'crop_101.h5', 'crop_66.h5', 'crop_74.h5', 'crop_18.h5', 'crop_377.h5', 'crop_69.h5', 'crop_22.h5', 'crop_241.h5', 'crop_165.h5', 'crop_326.h5', 'crop_163.h5', 'crop_240.h5', 'crop_291.h5', 'crop_8.h5', 'crop_333.h5', 'crop_245.h5', 'crop_146.h5', 'crop_369.h5', 'crop_83.h5', 'crop_72.h5', 'crop_89.h5', 'crop_93.h5', 'crop_174.h5', 'crop_3.h5', 'crop_274.h5', 'crop_203.h5', 'crop_155.h5', 'crop_197.h5', 'crop_340.h5', 'crop_201.h5']
        filtered_data_paths = [path for path in data_paths if os.path.basename(path) in valid_crops]
        return filtered_data_paths
