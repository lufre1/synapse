import synapse.util as util
from config import *
import h5py
import argparse
import os
from glob import glob
import numpy as np
import torch_em
import napari
import elf.parallel as parallel
from elf.io import open_file
from scipy.ndimage import binary_erosion, binary_fill_holes, binary_closing
from tqdm import tqdm
from synapse_net.inference.util import apply_size_filter, get_prediction, _Scaler, _postprocess_seg_3d
from skimage.measure import regionprops, label
from tifffile import imread
import synapse.label_utils as lutil


def _read_h5(path, key, scale_factor, z_offset=None):
    with h5py.File(path, "r") as f:
        try:
            print(f"{key} data shape", f[key].shape)
            if key == "prediction" or "pred" in key or "combined" in key:
                image = f[key][::scale_factor, ::scale_factor, ::scale_factor]
                if z_offset:
                    image = image[z_offset[0]:z_offset[1], :, :]
            else:
                image = f[key][::scale_factor, ::scale_factor, ::scale_factor]
                if z_offset:
                    image = image[z_offset[0]:z_offset[1], :, :]
            print(f"{key} data shape after downsampling", image.shape)
            # if not key == "raw":
            #     print(np.unique(image))

        except KeyError:
            print(f"Error: {key} dataset not found in {path}")
            return None  # Indicate error

        return image


def get_all_keys_from_h5(file_path):
    keys = []
    with h5py.File(file_path, 'r') as h5file:
        def collect_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys.append(name)  # Add each key (path) to the list
        h5file.visititems(collect_keys)  # Visit all groups and datasets
    return keys


def get_file_paths(path, reverse=False):
    if os.path.isfile(path):
        return [path]
    else:
        paths = sorted(glob(os.path.join(path, "**", "*.h5"), recursive=True), reverse=reverse)
        print(f"Found {len(paths)} files:")
        return paths


def visualize_data(data):
    viewer = napari.Viewer()
    for key, value in data.items():
        if np.issubdtype(value.dtype, np.integer):
            viewer.add_labels(value, name=key)
        elif np.issubdtype(value.dtype, np.floating):
            viewer.add_image(value, name=key, blending="additive")
        else:
            viewer.add_image(value, name=key, blending="additive")
        # if key == "raw" or "raw" in key:
        #     viewer.add_image(value, name=f"{key}_raw")
        #     value = torch_em.transform.raw.normalize_percentile(value, lower=5, upper=95)
        #     viewer.add_image(value, name=key)
        # elif key == "prediction" or "pred" in key:
        #     viewer.add_image(value, name=key, blending="additive")
        # elif "dist" in key or "hmap" in key:
        #     viewer.add_image(value, name=key, blending="additive")
        # else:
        #     viewer.add_labels(value, name=key)
        # Get the "raw" layer
    raw_layer = next((layer for layer in viewer.layers if "raw" in layer.name or "0" == layer.name), None)
    if raw_layer:
        # Remove the "raw" layer from its current position
        viewer.layers.remove(raw_layer)
        # Add the "raw" layer to the beginning of the layer list
        viewer.layers.insert(0, raw_layer)
    napari.run()


def _segment(pred,
             block_shape=(64, 512, 512),
             halo=(32, 64, 64),
             seed_distance=2 * 1,
             boundary_threshold=0.25 - 0.0,
             min_size=5000,
             area_threshold=500 * 1,
             dist=None
             ):
    foreground, boundaries = pred
    # # #boundaries = binary_erosion(boundaries < boundary_threshold, structure=np.ones((1, 3, 3)))
    if dist is None:
        dist = parallel.distance_transform(boundaries < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape)
    # # data["pred_dist_without_fore"] = parallel.distance_transform((boundaries) < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape)
    hmap = ((dist.max() - dist) / dist.max())

    # hmap[np.logical_and(boundaries > boundary_threshold, foreground < boundary_threshold)] = (hmap + boundaries).max()
    hmap[boundaries > 1.0 - boundary_threshold * 2] = (hmap + boundaries).max()

    # # hmap = hmap.clip(min=0)
    # seeds = np.logical_and(foreground > 0.5, dist > seed_distance)
    seeds = np.zeros_like(boundaries, dtype=bool)
    for z in range(boundaries.shape[0]):
        seeds[z] = np.logical_and(foreground[z] > 0.5, dist[z] > seed_distance)

    seeds = parallel.label(seeds, block_shape=block_shape, verbose=True)
    # # #seeds = binary_fill_holes(seeds)

    # mask = (foreground + boundaries) > 0.5
    # mask = np.logical_or(foreground > 0.5, boundaries > 0.5)
    mask = foreground > 0.5
    seg = np.zeros_like(seeds)
    seg = parallel.seeded_watershed(
        hmap, seeds, block_shape=block_shape,
        out=seg, mask=mask, verbose=True, halo=halo,
    )
    seg = apply_size_filter(seg, min_size, verbose=True, block_shape=block_shape)
    seg = _postprocess_seg_3d(seg, area_threshold=area_threshold, iterations=4, iterations_3d=8)
    seg_data = {
        "seg_new": seg,
        "seeds": seeds,
        "dist": dist,
        "hmap": hmap,
        "mask": mask
    }
    return seg_data


def visualize():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--path", "-p", type=str, required=True, help="Path to the data directory or single file")
    parser.add_argument("--scale_factor", "-s", type=int, default=1, help="Scale factor for the data")
    parser.add_argument("--no_visualize", "-nv", action="store_true", default=False, help="Don't visualize data with napari")
    parser.add_argument("--z_offset", "-z", type=int, nargs=2, default=None, help="Z offset for the data e.g. 5 -5")
    parser.add_argument("--key", "-k", type=str, default=None, help="If given, only load key and raw from file to visualize")
    args = parser.parse_args()

    paths = get_file_paths(args.path, reverse=False)
    #paths.extend(get_file_paths("/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2/"))

    print("len paths", len(paths))
    label_transform = torch_em.transform.BoundaryTransform(add_binary_target=True)
    label_transform_thick = label_transform = lutil.CombinedLabelTransform(add_binary_target=True, dilation_footprint=np.ones((4, 4)))

    for path in tqdm(paths):
        print(path)
        skip = False

        all_keys = get_all_keys_from_h5(path)
        # filter keys for raw and mito
        keys = []
        if args.key is not None:
            for k in all_keys:
                if "raw" in k or args.key in k:
                    keys.append(k)
        else:
            keys = all_keys

        keys.sort(reverse=True)
        print("\ndata keys", keys)
        # if "label_crop/mito" not in keys:
        #     print(f"  -> Skipping '{path}' does not contain mitochondria.")
        #     continue
        print("in path", path)
        data = {}
        for key in keys:
            data[key] = _read_h5(path, key, args.scale_factor, z_offset=(args.z_offset))
            if "label" in key:
                data[key + "_transformed"] = label_transform(data[key])
                data[key + "_transformed_thick"] = label_transform_thick(data[key])
            
            # if "raw" in key:
            #     if data[key].ndim == 4:
            #         data[key] = util.normalize_percentile_with_channel(data[key], lower=1, upper=99, channel=0)
            #     else:
            #         data[key] = torch_em.transform.raw.normalize_percentile(data[key], lower=1, upper=99)

        preds = (data["pred/foreground"], data["pred/boundary"])
        seg_data = _segment(pred=preds)
        for key, val in seg_data.items():
            data[key] = val

        visualize_data(data)


if __name__ == "__main__":
    visualize()