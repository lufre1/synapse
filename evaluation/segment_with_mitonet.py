import argparse
# from empanada.model import MitoNet
from mitonet_seg.inferencer import inference_3d
import tifffile
import zarr
import os
import h5py
import numpy as np


def save_data(path, data_dict, overwrite=False):
    """
    Save data to a .h5 file from given path and data dictionary.
    
    Parameters:
        path (str): Path to the .h5 file to be saved.
        data_dict (Dict[str, numpy.ndarray]): Dictionary containing dataset names as keys and the actual data as values.
        overwrite (bool): If True, overwrite the file if it already exists. If False, raise an exception if the file already exists.
    """
    if os.path.exists(path) and not overwrite:
        raise FileExistsError("File already exists at path:", path)
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not path.endswith(".h5"):
        path += ".h5"
    print(f"Saving data to {path}")
    if os.path.exists(path):
        print(f"Overwriting existing file at {path}")
    with h5py.File(path, "w") as f:
        for key, value in data_dict.items():
            f.create_dataset(key, data=value)
    print(f"Data successfully saved to {path}")


def load_zarr_image(path, key):
    """
    Load a zarr dataset from given path and key, then return the image.

    Parameters:
        path (str): Path to the zarr file or directory.
        key (str): Key to the dataset.

    Returns:
        image (numpy.ndarray): The image loaded from the zarr dataset.
    """
    if ".zarr" not in path:
        Exception("Not a zarr file; must end with '.zarr', but got:", path)

    image = zarr.open(path, mode='r')[key][:]
    return image


def main(args):
    image = load_zarr_image(args.input_path, args.key)
    image_reordered = np.transpose(image, (2, 1, 0))

    # see https://github.com/hoogenboom-group/mitonet-seg/blob/main/src/mitonet_seg/inferencer.py
    result = inference_3d(
        volume_data=image_reordered,
        config="/mnt/vast-nhr/home/freckmann15/u12103/mitonet-seg/configs/MitoNet_v1.yaml",
    )
    result = np.squeeze(result, axis=0)
    # reorder everythin back to normal z, y, x
    result = np.transpose(result, (2, 1, 0))

    # Save result
    data = {
        'segmentation': result,
        'raw': image
    }

    # Save result
    save_data(args.output_path, data)


if __name__ == "__main__":
    argsparse = argparse.ArgumentParser()
    argsparse.add_argument("--input_path", "-i", type=str, default="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/cutout_1/images/ome-zarr/raw.ome.zarr")
    argsparse.add_argument("--key", "-k", type=str, default=0)
    argsparse.add_argument("--output_path", "-o", type=str, default="/mnt/ceph-ssd/workspaces/ws/nim00007/u12103-volume-em/cutout_1_out_mitonet_reordered")
    argsparse.add_argument("--force_overwrite", "-fo", action="store_true", default=False)
    args = argsparse.parse_args()
    main(args)
