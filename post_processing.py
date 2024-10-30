import util
import data_classes
from config import *
import h5py
import z5py
import argparse
import os
from glob import glob
import numpy as np
import multiprocessing as mp
# from torch_em.util.segmentation import connected_components_with_boundaries, watershed_from_components
from elf.wrapper.base import SimpleTransformationWrapper
from elf.parallel import label, size_filter, seeded_watershed
# from skimage import measure, segmentation
# from elf.wrapper.resized_volume import ResizedVolume
from skimage import morphology

import napari


def _read_h5_pred(path, key, scale_factor):
    with h5py.File(path, "r") as f:
        try:
            print(f"{key} data shape", f[key].shape)
            image = f[key][:, ::scale_factor, ::scale_factor, ::scale_factor]
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
        if key == "raw":
            viewer.add_image(value, name="Raw")
        else:
            viewer.add_labels(value, name=key)
    napari.run()


def change_file_extension(file_path, new_extension=".z5", append_to_name="segmentation"):
    """
    Returns a new file path with the specified file extension.

    Parameters:
    - file_path (str): The original file path.
    - new_extension (str): The new file extension, with or without a leading dot.

    Returns:
    - str: The new file path with the updated extension.
    """
    # Ensure the new extension starts with a dot
    if not new_extension.startswith('.'):
        new_extension = '.' + new_extension
    
    # Get the base name without the current extension and append the new one
    base_name = os.path.splitext(file_path)[0]
    base_name = os.path.join(base_name, append_to_name) 
    new_file_path = f"{base_name}{new_extension}"
    
    return new_file_path


def mitochondria_segmentation(prediction, patch_shape=(32, 256, 256), roi=False, threshold_mask=0.5, min_size=15000):
    n_threads = mp.cpu_count()
    if not roi:
        roi = np.s_[:, :, :]
    
    shape, chunks = prediction.shape[1:], patch_shape
    #print("shape", shape, "chunks", chunks)
    threshold_seeds = 0.6
    input_ = SimpleTransformationWrapper(
        prediction, lambda x: (x[0] - x[1]) > threshold_seeds,
        shape=shape, chunks=chunks, dtype=np.dtype("bool"),
        with_channels=True
    )
    
    # input_ = (prediction[0] - prediction[1]) > threshold_seeds
    # visualize_data({"seeds": input_})
    # run connected components (in parallel) to use as seeds for the watershed below
    block_shape = tuple(2 * ch for ch in chunks)
    print(f"\nshape of input: {input_.shape}, block_shape: {block_shape}")
    seeds = label(input_, with_background=True,
                  verbose=True, n_threads=n_threads, block_shape=block_shape,
                  roi=roi
                  )
    # seeds = measure.label(input_)
    # visualize_data({"seeds": input_})
    # print(f"any values in seeds {np.any(seeds[1:])} and shape {seeds.shape}")
    

    # step 2:
    # run parallel watershed to expand the seeds to the full nuclei

    # wrapper to extract the boundaries (= channel 1) from the predictions
    ws_hmap = SimpleTransformationWrapper(
        prediction, lambda x: x[1], shape=shape, chunks=chunks, with_channels=True
    )
    # ws_hmap = prediction[1]
    # visualize_data({"raw": ws_hmap})
    # print(f"shape of ws_hmap: {ws_hmap.shape} ws hamp any values other than first slice {np.any(ws_hmap[1:])}")
    # wrapper to define the foreground (max(foreground prediction, boundary) > 0.5)
    ws_mask = SimpleTransformationWrapper(
        prediction, lambda x: np.max(x, axis=0) > threshold_mask,
        shape=shape, chunks=chunks, dtype=np.dtype("bool"), with_channels=True
    )
    # ws_mask = np.max(prediction, axis=0) > threshold_mask
    # visualize_data({"ws_mask": ws_mask, "raw": ws_hmap})
    
    # print(f"ws_mask shape {ws_mask.shape} ws mask any values other than first slice {np.any(ws_mask[1:])}")

    # run the watershed
    halo = [patch_shape[0] // 8, patch_shape[1] // 8, patch_shape[2] // 8]
    output = np.zeros_like(a=prediction[0], dtype=np.dtype("uint32"))
    print(f"blanko output file shape: {output.shape}")
    seeded_watershed(
        ws_hmap, seeds, output, block_shape, halo, mask=ws_mask, n_threads=n_threads, verbose=True, roi=roi
    )
    # output = segmentation.watershed(
    #     ws_hmap, seeds
    # )
    # visualize_data({"output": output})
    # print(f"any values in output after watershed {np.any(output)}")

    # filter out small objects smaller than some minimal size
    if min_size > 0:
        size_filter(output, output, min_size=min_size, n_threads=n_threads, verbose=True, relabel=True, roi=roi, block_shape=block_shape)
    return output


def post_process():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--path", "-p", type=str, required=True, help="Path to the data directory or single file")
    parser.add_argument("--scale_factor", "-s", type=int, default=1, help="Scale factor for the data")
    parser.add_argument("--visualize", "-v", action="store_true", default=False, help="Don't visualize data with napari")
    parser.add_argument("--patch_shape", "-ps", type=int, nargs=3, default=(32, 256, 256), help="Patch shape for data loading (3D tuple) also used for chunks")
    args = parser.parse_args()

    paths = get_file_paths(args.path)

    for path in paths:
        print(path)

        key = "prediction"
        pred = _read_h5_pred(path, key, args.scale_factor)
        assert pred.ndim == 4
        seg = mitochondria_segmentation(pred, patch_shape=args.patch_shape)
        # seg = connected_components_with_boundaries(
        #     foreground=data["prediction"][0],
        #     boundaries=data["prediction"][1],
        #     threshold=0.5,
        #     )
        # seg = watershed_from_components(seg[1], seg[0])
        if args.visualize:
            print("any values in seg?", seg.any())
            vis_data = {
                #"raw": data["raw"],
                "raw": pred,
                "segmentation": seg
            }
            visualize_data(vis_data)
        new_path = change_file_extension(path, new_extension=".h5")
        with h5py.File(new_path, "a") as f:
            f.create_dataset(
                "segmentation", data=seg,
                compression="gzip",
                shape=seg.shape,
                dtype="uint32",
                chunks=args.patch_shape
            )
        print(f"Saved segmentation to {new_path}")


if __name__ == "__main__":
    post_process()