import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import os
from glob import glob
import napari
from tqdm import tqdm
import synapse.io.util as io
from tifffile import imread
import numpy as np
from collections import Counter, defaultdict
import h5py
from elf.io import open_file
import synapse.cellmap_util as cutil

ID_GROUPS = [
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 37, 52, 53, 65],  # nucleus with pores and envelope
    # [6, 7, 40],                                            # golgi
    # [8, 9, 41],                                            # vesicle
    # [10, 11, 42],                                          # endosome
    # [12, 13, 43],                                          # lysosome
    # [14, 15, 44],                                          # lipid droplet
    # [16, 17, 18, 19, 46, 51, 64],                          # endoplasmic reticulum with exit sites
    # [47, 48, 49],                                          # peroxisome
    # [3, 4, 5, 50],                                         # mitochondria
    # [24, 25, 26, 27, 54],                                  # chromatin
    # [30, 36, 55],                                          # microtubule
    # [38, 39, 56, 57, 58, 61, 62, 60],                      # cell
    # [31, 32, 33, 66],                                      # centrosome collective
    # [34],                                                  # ribosomes
    # [35],                                                  # cytosol
    # [0, 1, 2],                                             # extracellular space + plasma membrane
    # [45],                                                  # red blood cells
]


def collect_dataset_shapes(h5file, group=None, prefix=""):
    if group is None:
        group = h5file
    shapes = []
    for key in group:
        full_key = f"{prefix}/{key}" if prefix else key
        try:
            item = group[key]
        except KeyError:
            continue
        if isinstance(item, h5py.Dataset):
            shapes.append(item.shape)
        elif isinstance(item, h5py.Group):
            shapes.extend(collect_dataset_shapes(h5file, item, full_key))
    return shapes


def percentage_files_with_inconsistent_shapes(file_paths):
    num_total = 0
    num_inconsistent = 0

    for path in tqdm(file_paths):
        try:
            with h5py.File(path, "r") as f:
                shapes = collect_dataset_shapes(f)
                if not shapes:
                    continue  # Skip files with no datasets
                num_total += 1
                first_shape = shapes[0]
                if not all(shape == first_shape for shape in shapes):
                    num_inconsistent += 1
        except Exception as e:
            print(f"[Error] {path}: {e}")

    percentage = (num_inconsistent / num_total) * 100 if num_total > 0 else 0
    print(f"\n{num_inconsistent}/{num_total} files have internally inconsistent dataset shapes.")
    print(f"That's {percentage:.2f}% inconsistent files.")
    return percentage


def write_h5_paths_to_csv(paths, output_path="h5_paths_with_mitos_new.csv", ):
    h5_files = paths

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["h5_file_path"])
        for h5 in h5_files:
            full_path = os.path.abspath(h5)
            writer.writerow([full_path])

    print(f"Wrote {len(h5_files)} paths to {output_path}")


def find_files_with_exact_mito_key(h5_paths):
    """
    Returns a list of .h5 file paths that contain a dataset with the exact key 'mito'.
    """
    matching_files = []

    for path in h5_paths:
        try:
            with h5py.File(path, 'r') as f:
                found = False

                def visitor(name, obj):
                    nonlocal found
                    if isinstance(obj, h5py.Dataset):
                        key_name = name.split('/')[-1]
                        if key_name == "mito":
                            found = True

                f.visititems(visitor)

                if found:
                    matching_files.append(path)

        except Exception as e:
            print(f"Could not read {path}: {e}")

    return matching_files


def print_shape_stats(shapes, ):
    """
    Prints the mean of the (z, y, x) shapes and the prevalence of each unique shape.

    Accepts a list where each element is either a (z, y, x) tuple or a NumPy array of shape (3,).

    Parameters:
        shapes (list): List containing either (z, y, x) tuples or NumPy arrays of shape (3,).
    """
    if not shapes:
        print("No shapes provided.")
        return

    processed_shapes = []
    for shape in shapes:
        if isinstance(shape, tuple) and len(shape) == 3:
            processed_shapes.append(shape)
        elif isinstance(shape, np.ndarray) and shape.shape == (3,):
            processed_shapes.append(tuple(shape.tolist()))
        else:
            raise TypeError("Each element in the input list must be a (z, y, x) tuple or a NumPy array of shape (3,).")

    if not processed_shapes:
        print("No valid (z, y, x) shapes found in the input.")
        return

    shape_array = np.array(processed_shapes)  # shape: (N, 3)
    mean_shape = shape_array.mean(axis=0)

    print(f"Mean shape (z, y, x): ({mean_shape[0]:.2f}, {mean_shape[1]:.2f}, {mean_shape[2]:.2f})")

    # Count occurrences of each unique shape
    counter = Counter(processed_shapes)
    print("\nPrevalence of unique shapes:")
    for shape, count in counter.most_common():
        print(f"  Shape {shape}: {count} times")


def check_any_id_group(path, ID_GROUPS, dataset="label_crop/all", min_groups=2):
    try:
        with h5py.File(path, "r") as f:
            if dataset not in f:
                return None
            data = f[dataset][:]
            n_groups = 0
            # data.shape = (z, y, x) assumed
            for group in ID_GROUPS:
                uniq = np.unique(data)
                if any(label in group for label in uniq):
                    n_groups += 1
                    if n_groups >= min_groups:
                        return path
                # z_contain = []
                # for z in range(data.shape[0]):
                #     # unique labels in this slice
                #     labels_in_slice = np.unique(data[z])
                #     if any(label in group for label in labels_in_slice):
                #         z_contain.append(1)
                #     else:
                #         z_contain.append(0)
                # fraction = sum(z_contain) / data.shape[0]
                # if fraction > min_pct_slices:
                #     return path
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return None


def get_paths_with_min_id_group(data_paths, ID_GROUPS, dataset="label_crop/all", n_workers=None, min_groups=2):
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
    print("Using", n_workers, "workers for parallel processing")
    valid_paths = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                check_any_id_group, path, ID_GROUPS, dataset, min_groups
            )
            for path in data_paths
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            result = future.result()
            if result is not None:
                valid_paths.append(result)
    return valid_paths


def get_filtered_paths():
    return [
        '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_101.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_116.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_120.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_121.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_117.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_122.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_14.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_107.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_109.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_13.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_139.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_140.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_119.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_118.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_143.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_123.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_15.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_144.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_142.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_126.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_130.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_147.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_146.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_150.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_102.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_149.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_134.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_148.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_157.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_131.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_162.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_151.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_160.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_161.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_166.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_163.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_141.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_165.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_1.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_16.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_159.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_158.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_100.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_174.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_164.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_18.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_176.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_178.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_181.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_180.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_19.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_182.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_183.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_111.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_110.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_171.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_173.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_115.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_185.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_125.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_124.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_20.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_138.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_112.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_137.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_133.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_113.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_136.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_187.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_135.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_196.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_189.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_186.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_132.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_188.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_195.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_22.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_191.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_172.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_155.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_193.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_198.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_192.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_190.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_197.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_200.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_206.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_199.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_209.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_201.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_175.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_203.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_234.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_202.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_236.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_214.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_235.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_239.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_241.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_212.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_237.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_240.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_23.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_217.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_219.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_245.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_242.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_255.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_249.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_208.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_210.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_256.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_248.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_252.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_213.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_259.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_222.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_258.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_224.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_27.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_220.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_216.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_227.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_28.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_228.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_291.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_218.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_226.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_225.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_292.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_31.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_32.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_272.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_267.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_273.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_268.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_269.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_270.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_129.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_279.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_275.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_276.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_266.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_33.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_278.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_274.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_34.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_298.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_313.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_280.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_277.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_320.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_3.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_322.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_35.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_321.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_156.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_36.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_145.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_323.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_319.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_325.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_326.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_37.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_333.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_341.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_342.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_340.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_38.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_368.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_370.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_376.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_369.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_39.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_40.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_377.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_4.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_229.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_247.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_221.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_184.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_416.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_417.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_42.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_257.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_238.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_43.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_47.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_230.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_177.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_346.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_48.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_49.h5', 
        '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_51.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_50.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_231.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_6.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_7.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_356.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_8.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_80.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_79.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_211.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_345.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_78.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_9.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_410.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_386.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_411.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_355.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_387.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_412.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_413.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_472.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_380.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_381.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_421.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_473.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_423.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_407.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_408.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_362.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_347.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_329.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_348.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_324.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_366.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_379.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_378.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_354.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_353.h5', '/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_351.h5'
    ]


def main(visualize=False):
    parser = argparse.ArgumentParser()
    # /mnt/lustre-emmy-hdd/projects/nim00007/data/synaptic-reconstruction/cooper/original_imod_data/20240909_cp_datatransfer
    parser.add_argument("--path", "-p",  type=str, default="/scratch-grete/projects/nim00007/data/cellmap", help="Path to the root data directory")
    parser.add_argument("--ext", "-e", type=str, default=".h5")
    parser.add_argument("--path2", "-p2",  type=str, default="/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2", help="Path to the root data directory")
    parser.add_argument("--ext2", "-e2", type=str, default=".tif")
    #parser.add_argument("--save_dir", type=str, default="", help="Path to save the data to")
    args = parser.parse_args()
    # print(args.base_path, "\n", args.base_path2)
    # /scratch-grete/projects/nim00007/data/mitochondria/cooper/20250308_Mito_Seg_Done/refined

    b1_paths = sorted(glob(os.path.join(args.path, "**", f"*{args.ext}"), recursive=True))#, reverse=True)
    # b2_paths = sorted(glob(os.path.join(args.path2, "**", f"*{args.ext2}"), recursive=True))#, reverse=True)
    
    # stats = percentage_files_with_inconsistent_shapes(b1_paths)
    # print("stats", stats)
    # paths = find_files_with_exact_mito_key(b1_paths)
    # paths = cutil.get_cellmap_mito_paths()
    # write_h5_paths_to_csv(paths, output_path="h5_paths_mitos.csv")
    # print(len(paths))
    # shapes = []
    # all_ids = 0
    # scales = []
    # print(get_paths_with_min_id_group(b1_paths, ID_GROUPS, dataset="label_crop/all", min_groups=0))
    
    # paths = get_filtered_paths()
    # print("len filtered paths", len(paths))
    # print("diff between b1_paths and paths", len(b1_paths) - len(paths))
    # return

    for path in tqdm(b1_paths):
        dataset = "label_crop/all"
        try:
            with h5py.File(path, "r", swmr=True) as f:
                # if dataset not in f:
                #     return None
                # data = f[dataset][:]
                # if np.all(data == data[0]):
                #     print(path)
                if dataset not in f:
                    return None
                data = f[dataset][:]
                label = data[0, 0, 0]
                for i in range(data.shape[0]):
                    for j in range(data.shape[1]):
                        for k in range(data.shape[2]):
                            if data[i, j, k] != label:
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    print(path)
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
        # print(path)
        # if "247" in path:
        #     continue
        # with open_file(path, "r") as f:
        #     # breakpoint()
        #     labels = f["label_crop/all"]
        #     # print(path)
        #     uniqs = np.unique(labels)
        #     contains = []
        #     for id in [37, 40, 41, 42, 43, 44, 49, 50, 51, 53, 55, 57, 59, 60, 64, 65, 66]:
        #         if np.any(uniqs == id):
        #             contains.append(id)
        #     if contains:
        #         print(os.path.basename(path), f"path contains group id {contains}")
            # for id in range(0, 70):
            #     if np.any(labels == id):
            #         print(f"path contains id {id}:", path)
            
    #     data = imread(path)
    #     uniq = np.unique(data)
    #     mito_ids = uniq[uniq != 0]
    #     print("np.unique", mito_ids)
    #     all_ids += len(mito_ids)
    # print("all mitos", all_ids)
    #         shapes.append(f["label_crop/mito"].shape)
    #         print("np.unique", np.unique(f["label_crop/mito"], return_counts=True))

    # print(len(shapes))
    # print_shape_stats(scales)


if __name__ == "__main__":
    main()