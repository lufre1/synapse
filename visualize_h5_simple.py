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
from scipy.ndimage import binary_erosion, binary_fill_holes, binary_closing
from tqdm import tqdm
from synapse_net.inference.util import apply_size_filter, get_prediction, _Scaler, _postprocess_seg_3d


def _read_h5(path, key, scale_factor, z_offset=None):
    with h5py.File(path, "r") as f:
        try:
            print(f"{key} data shape", f[key].shape)
            if key == "prediction" or "pred" in key or "combined" in key:
                image = f[key][:, ::scale_factor, ::scale_factor, ::scale_factor]
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
        if key == "raw" or "raw" in key:
            value = torch_em.transform.raw.normalize_percentile(value, lower=5, upper=95)
            viewer.add_image(value, name=key)
        elif key == "prediction" or "pred" in key:
            viewer.add_image(value, name=key, blending="additive")
        else:
            viewer.add_labels(value, name=key)

    napari.run()


def _segment(pred,
             block_shape=(32, 256, 256),
             halo=(16, 64, 64),
             seed_distance=18,
             boundary_threshold=0.15,
             min_size=50000*8,
             area_threshold=1000,
             with_hmp_max_value=True,
             dist=None
             ):
    foreground, boundaries = pred
    # # #boundaries = binary_erosion(boundaries < boundary_threshold, structure=np.ones((1, 3, 3)))
    if dist is None:
        dist = parallel.distance_transform(boundaries < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape)
    # # data["pred_dist_without_fore"] = parallel.distance_transform((boundaries) < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape)
    hmap = ((dist.max() - dist) / dist.max())
    if with_hmp_max_value:
        hmap[boundaries > boundary_threshold] = (hmap + boundaries).max()
    else:
        hmap = hmap + boundaries
        hmap[boundaries > boundary_threshold] = hmap.max()
    # # hmap = hmap.clip(min=0)
    seeds = np.logical_and(foreground > 0.5, dist > seed_distance)
    seeds = parallel.label(seeds, block_shape=block_shape, verbose=True)
    # # #seeds = binary_fill_holes(seeds)
    
    mask = (foreground + boundaries) > 0.5
    seg = np.zeros_like(seeds)
    seg = parallel.seeded_watershed(
        hmap, seeds, block_shape=block_shape,
        out=seg, mask=mask, verbose=True, halo=halo,
    )
    seg = apply_size_filter(seg, min_size, verbose=True, block_shape=block_shape)
    seg = _postprocess_seg_3d(seg, area_threshold=area_threshold, iterations=4, iterations_3d=8)
    seg_data = {
        "seg": seg,
        "seeds": seeds,
        "dist": dist,
        "hmap": hmap
    }
    return seg_data
        

def visualize():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--path", "-p", type=str, required=True, help="Path to the data directory or single file")
    parser.add_argument("--scale_factor", "-s", type=int, default=1, help="Scale factor for the data")
    parser.add_argument("--no_visualize", "-nv", action="store_true", default=False, help="Don't visualize data with napari")
    parser.add_argument("--z_offset", "-z", type=int, nargs=2, default=None, help="Z offset for the data e.g. 5 -5")
    args = parser.parse_args()

    paths = get_file_paths(args.path, reverse=False)
    #paths.extend(get_file_paths("/scratch-grete/projects/nim00007/data/mitochondria/cooper/fidi_down_s2/"))

    print("len paths", len(paths))
    statistics = {}

    for path in tqdm(paths):
        #print(path)

        keys = get_all_keys_from_h5(path)
        if "labels/cristae" not in keys:
            continue
        keys.sort(reverse=True)
        print("\ndata keys", keys)
        print("in path", path)
        data = {}
        for key in keys:
            data[key] = _read_h5(path, key, args.scale_factor, z_offset=(args.z_offset))
            # if "raw" in key:
            #     data[key] = util.normalize_percentile_with_channel(data[key], lower=1, upper=99, channel=0)
                # data[key] = np.stack([torch_em.transform.raw.normalize_percentile(data[key][0]), data[key][1]], axis=0)
                # data["mitos"] = data[key][1]

            # if "mito" in key:
            #     data[key] = _read_h5(path, key, args.scale_factor, z_offset=(args.z_offset))
            # else:
            #     continue

        filtered_data = {}

        if data and not args.no_visualize:
            if filtered_data:
                visualize_data(filtered_data)
            else:
                visualize_data(data)
        else:
            #print("Calculate Statistics...")
            block_shape = (32, 256, 256)
            statistics[path] = {
                "#mitos": len(np.unique((data["labels/mitochondria"]))),
            }
    print("statistics:")
    mitos = 0
    for key in statistics.keys():
        mitos += statistics[key]["#mitos"]
        print(statistics[key])
    print("amount of all mitos", mitos)


if __name__ == "__main__":
    visualize()