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
from synapse_net.inference.util import apply_size_filter, get_prediction, _Scaler, _postprocess_seg_3d


def _read_h5(path, key, scale_factor, z_offset=None):
    with h5py.File(path, "r") as f:
        try:
            print(f"{key} data shape", f[key].shape)
            if key == "prediction" or "pred" in key:
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


def get_file_paths(path):
    if os.path.isfile(path):
        return [path]
    else:
        paths = sorted(glob(os.path.join(path, "**", "*.h5"), recursive=True))
        print(f"Found {len(paths)} files:")
        return paths


def visualize_data(data):
    viewer = napari.Viewer()
    for key, value in data.items():
        if key == "raw" or "raw" in key:
            value = torch_em.transform.raw.standardize(value)
            viewer.add_image(value, name=key)
        elif key == "prediction" or "pred" in key:
            viewer.add_image(value, name=key, blending="additive")
        else:
            viewer.add_labels(value, name=key)

    napari.run()


def visualize():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--path", "-p", type=str, required=True, help="Path to the data directory or single file")
    parser.add_argument("--scale_factor", "-s", type=int, default=1, help="Scale factor for the data")
    parser.add_argument("--no_visualize", "-nv", action="store_true", default=False, help="Don't visualize data with napari")
    parser.add_argument("--z_offset", "-z", type=int, nargs=2, default=None, help="Z offset for the data e.g. 5 -5")
    args = parser.parse_args()

    paths = get_file_paths(args.path)
    # paths = util.get_wichmann_data()

    shapes = []
    i = 0
    skip = True
    for path in paths:
        print(path)
        if "WT20_syn7_model2" in path:
            skip = False
        if skip:
            continue
        # if i < 2:
        #     i += 1
        #     continue
        keys = get_all_keys_from_h5(path)
        keys.sort(reverse=True)
        print("\ndata keys", keys)
        print("in path", path)
        data = {}
        for key in keys:
            data[key] = _read_h5(path, key, args.scale_factor, z_offset=(args.z_offset))
            # data[key] = _read_h5(path, key, args.scale_factor)
        filtered_data = {}

        # block_shape = (64, 512, 512)
        # halo = (32, 128, 128)
        # seed_distance = 6
        # boundary_threshold = 0.2
        # min_size = 50000*10
        # foreground, boundaries = data["pred"]
        # # #boundaries = binary_erosion(boundaries < boundary_threshold, structure=np.ones((1, 3, 3)))
        # dist = parallel.distance_transform(boundaries < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape)
        # # data["pred_dist_without_fore"] = parallel.distance_transform((boundaries) < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape)
        # hmap = boundaries + ((dist.max() - dist) / dist.max())
        # # hmap = hmap.clip(min=0)
        # seeds = np.logical_and(foreground > 0.5, dist > seed_distance)
        # seeds = parallel.label(seeds, block_shape=block_shape, verbose=True)
        # # #seeds = binary_fill_holes(seeds)
        
        # mask = (foreground + boundaries) > 0.5
        # seg = np.zeros_like(seeds)
        # seg = parallel.seeded_watershed(
        #     hmap, seeds, block_shape=block_shape,
        #     out=seg, mask=mask, verbose=True, halo=halo,
        # )
        # seg = apply_size_filter(seg, min_size, verbose=True, block_shape=block_shape)
        # seg = _postprocess_seg_3d(seg, area_threshold=1000, iterations=4, iterations_3d=8)

        # data["pred_hmap"] = hmap
        # data["pred_dist"] = dist
        # data["seg"] = seg
        # data["seeds"] = seeds
        #data["postprocessed_seeds"] = binary_closing(binary_closing(seeds, structure=np.ones((5, 5, 5))), structure=np.ones((5, 5, 5)))


        if data and not args.no_visualize:
            # flattened_data = data["raw"].ravel()
            # upper_threshold = np.percentile(flattened_data, 99)
            # lower_threshold = np.percentile(flattened_data, 1)
            # print("lower and upper threshold", lower_threshold, upper_threshold)
            # filtered_data["raw"] = np.clip(data["raw"], lower_threshold, upper_threshold)  # data["raw"].copy()
            # # filtered_data["raw_5_95"] =np.clip(data["raw"], np.percentile(data["raw"], 5), np.percentile(data["raw"], 95))
            # filtered_data["raw_norm"] = torch_em.transform.raw.normalize(filtered_data["raw"])
            # print("min and max of raw_norm", filtered_data["raw_norm"].min(), filtered_data["raw_norm"].max())
            
            # filtered_data["raw_stand"] = torch_em.transform.raw.standardize(filtered_data["raw"])
            # artifact_mask = data["raw"] > lower_threshold
            # # slices_to_keep = [z for z in range(data["raw"].shape[0]) if np.min(data["raw"][z, :, :]) >= lower_threshold]
            # # for z in range(data["raw"].shape[0]):
            # #     print(np.min(data["raw"][z, :, :]))
            # # print("slices to keep", len(slices_to_keep), "out of", data["raw"].shape[0])
            # # for key, value in data.items():
            # #     filtered_data[key] = value[slices_to_keep, :, :]
            # data["raw2"] = data["raw"].copy()
            # # median = np.median(data["raw"])
            # print("np.percentile(data[raw], 5)",np.percentile(data["raw"], 5))
            # data["raw2"][artifact_mask] = np.percentile(data["raw"], 5)
            # print(median)
            if filtered_data:
                visualize_data(filtered_data)
            else:
                visualize_data(data)
        # if data:
        #     shapes.append(data["raw"].shape)
        #     print("min", np.min(data["raw"]))
        #     print("max", np.max(data["raw"]))
        #     print("mean", np.mean(data["raw"]))
        #     print("std", np.std(data["raw"]))
        #     print("percentile", np.percentile(data["raw"], [0, 25, 50, 75, 100]))
            
            # print("data.keys", data.keys())
            # shapes = []
            # for key, value in data.items():
            #     print(key, value.shape)
            #     shapes.append(value.shape)
            # print("shapes", shapes)
            # avg0 = np.mean(shapes, axis=0)    
            # avg1 = np.mean(data["raw"].shape, axis=1)    
            # avg2 = np.mean(data["raw"].shape, axis=2)    
            # print(avg0)#, avg1.shape, avg2.shape)
    # for shape in shapes:
    #     print(shape)
    # print("average shapes", np.mean(shapes, axis=0))


if __name__ == "__main__":
    visualize()