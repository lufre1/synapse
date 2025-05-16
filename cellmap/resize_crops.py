import argparse
from glob import glob
import os
from elf.io import open_file
from elf.parallel import label
import h5py
import numpy as np
from skimage.transform import resize


def find_datasets_with_substring(h5group, substring, prefix=""):
    paths = []
    for key in h5group.keys():
        item = h5group[key]
        this_path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Dataset):
            if substring in key or substring in this_path:
                paths.append(this_path)
        elif isinstance(item, h5py.Group):
            paths.extend(find_datasets_with_substring(item, substring, this_path))
    return paths


def save_labels_with_rescaled_voxel_size(path, out_path, labels, dataset_key, target_scale=(8, 8, 8)):
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        raw = f["raw_crop"][:]
        input_scale = np.array(f.attrs["scale"])  # e.g. [2,2,2]
        in_shape = np.array(raw.shape)
    
    # Compute the rescaling factors for each axis
    scale_factors = input_scale / np.array(target_scale)
    out_shape = tuple(np.round(in_shape * scale_factors).astype(int))
    # Check for drastic resizing (more than 2x up or down in any dimension)
    ratio = out_shape / in_shape
    # Use absolute ratio: either expansion or shrinkage should not go beyond factor 2
    too_drastic = np.any((ratio > 2))  # | (ratio < 0.5))

    if too_drastic:
        print(f"Skipping {path}: resizing factor in at least one dimension is more than a factor of 2. "
              f"in_shape={in_shape}, out_shape={out_shape}, ratios={ratio}"
              f"out path would have been: {out_path}")
        return None  # skip this sample

    # Resample raw (linear) and labels (nearest)
    # skip if in_shape == out_shape
    if np.all(in_shape == out_shape):
        raw_resized = raw
        labels_resized = labels
    else:
        raw_resized = resize(raw, out_shape, order=1, preserve_range=True, anti_aliasing=True).astype(raw.dtype)
        labels_resized = resize(labels, out_shape, order=0, preserve_range=True, anti_aliasing=False).astype(labels.dtype)

    # Update "scale" attribute to new voxel size
    attrs["scale"] = np.array(target_scale)
    attrs["resized_from_shape"] = in_shape
    attrs["resized_to_shape"] = out_shape

    with h5py.File(out_path, "a") as f:
        f.attrs.update(attrs)
        f.create_dataset(dataset_key, data=labels_resized, dtype=labels_resized.dtype)
        if "raw" not in f:
            f.create_dataset("raw", data=raw_resized, dtype=raw_resized.dtype, compression="gzip")


def extract_label_crop_ids(path, dataset):
    with open_file(path, "r") as f:
        result = None
        # Try the primary dataset first
        if dataset in f.keys():
            arr = f[dataset][:]
            result = arr.astype(np.uint8)
            return result
        else:
            print("Dataset", dataset, "not found in", path)
            return None


def main(args):
    input_path = args.input
    output_path = args.output
    # for local
    # input_path = "/home/freckmann15/data/cellmap/data_crops"
    # output_path = os.path.join("/home/freckmann15/data/cellmap/extracted_crops", args.fallback_key)
    ##
    os.makedirs(output_path, exist_ok=True)
    # breakpoint()
    if os.path.isdir(input_path):
        paths = sorted(glob(os.path.join(input_path, "**", "*.h5"), recursive=True))
    else:
        paths = [input_path]
    for path in paths:
        print("\nProcessing path:", path)
        # if "247" in path:
        #     continue
        out_path = os.path.join(output_path, os.path.basename(path))
        if os.path.exists(out_path):
            print("Output path already exist; skipping:", out_path)
            continue
        labels = extract_label_crop_ids(path, dataset=args.dataset_key)
        # debug
        if not np.any(labels):
            print("No labels found in", path)
            continue
        if labels is not None and args.debug:
            print("np.unique(labels):", np.unique(labels))
            with open_file(path, "r") as f:
                raw = f["raw_crop"][:]
            import napari
            viewer = napari.Viewer()
            viewer.add_image(raw)
            viewer.add_labels(labels)
            napari.run()
        
        save_labels_with_rescaled_voxel_size(path,
                                             out_path,
                                             labels,
                                             target_scale=args.taget_voxel_size,
                                             dataset_key=args.dataset_key
                                             )



if __name__ == "__main__":
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--input", "-i", type=str, default="/scratch-grete/projects/nim00007/data/cellmap/data_crops/")
    argsparse.add_argument("--dataset_key", "-k", type=str, default="label_crop/all")
    argsparse.add_argument("--output", "-o", type=str, default="/scratch-grete/projects/nim00007/data/cellmap/resized_crops/")
    argsparse.add_argument("--taget_voxel_size", "-tvs", type=int, nargs=3, default=(8, 8, 8))
    argsparse.add_argument("--debug", "-d", action="store_true", default=False)
    args = argsparse.parse_args()
    main(args)