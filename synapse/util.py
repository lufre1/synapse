import fnmatch
import os
from glob import glob
import h5py
import mrcfile
import tifffile
import imageio.v3 as iio
import z5py
import zarr
from elf.io import open_file
from tqdm import tqdm
import napari
import torch
import torch_em
from torch_em.util.prediction import predict_with_halo
import torch.nn as nn
import numpy as np
import yaml
import random
from skimage.measure import regionprops
from scipy.ndimage import label, sum_labels
from skimage.transform import resize
from synapse_net.inference.util import apply_size_filter, _postprocess_seg_3d
from synapse_net.file_utils import read_ome_zarr
from torch_em.model import AnisotropicUNet
# used for combined_datasets
from typing import Dict, List, Union, Tuple, Optional, Any

# Define the data path and filename
# data_path = "/scratch-grete/projects/nim00007/data/mitochondria/moebius/em_tomograms_v1/170-PLP-wt/170_2_rec.h5"
# data_format = "*.h5"


def get_3d_model(
    out_channels: int,
    in_channels: int = 1,
    scale_factors: Tuple[Tuple[int, int, int]] = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    initial_features: int = 32,
    final_activation: str = "Sigmoid",
) -> torch.nn.Module:
    """Get the U-Net model for 3D segmentation tasks.

    Args:
        out_channels: The number of output channels of the network.
        scale_factors: The downscaling factors for each level of the U-Net encoder.
        initial_features: The number of features in the first level of the U-Net.
            The number of features increases by a factor of two in each level.
        final_activation: The activation applied to the last output layer.

    Returns:
        The U-Net.
    """
    model = AnisotropicUNet(
        scale_factors=scale_factors,
        in_channels=in_channels,
        out_channels=out_channels,
        initial_features=initial_features,
        gain=2,
        final_activation=final_activation,
    )
    return model


def read_data(path, scale=1):
    data = {}
    if ".tif" in path:
        img = tifffile.imread(path)
        ndim = img.ndim
        slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
        data["label"] = img[slicing] if scale > 1 else img
    elif (".mrc" in path or ".rec" in path):
        with open_file(path, "r") as f:
            ndim = f["data"].ndim
            slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
            data["raw"] = f["data"][slicing] if scale > 1 else f["data"][:]
    elif (".h5" in path or ".n5" in path):
        with open_file(path, "r") as f:
            for key in f.keys():
                if isinstance(f[key], (zarr.Group, h5py.Group, z5py.Group)):
                    print(f"Loading group: {key}")
                    extract_data(f[key], data, scale=scale)
                    continue
            ndim = f["data"].ndim
            slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
            data[key] = f[key][slicing] if scale > 1 else f[key][:]
    elif (".zarr" in path):
        img, voxel_size = read_ome_zarr(path)
        ndim = img.ndim
        slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
        data["raw"] = img[slicing] if scale > 1 else img
        
    return data


def extract_data(group: Any, data: Dict[str, Any], prefix: str = "", scale: int = 1):
    """
    Recursively extract datasets from a group and store them in a dictionary.
    """
    for key, item in group.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(item, (zarr.Group, h5py.Group, z5py.Group)):
            # Recursively extract data from subgroups
            extract_data(item, data, prefix=full_key, scale=scale)
        else:
            ndim = item.ndim
            # Generate a slicing tuple based on the number of dimensions
            slicing = tuple(slice(None, None, scale) if i >= (ndim - 3) else slice(None) for i in range(ndim))
            
            # Apply downsampling while preserving batch/channel dimensions
            data[full_key] = item[slicing] if scale > 1 else item[:]
            # # Store the dataset in the dictionary
            # data[full_key] = item[:]


def downsample_to_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Downsample an array to the target shape by selecting voxels evenly (every ith voxel) along each axis.

    Parameters:
    - arr: np.ndarray
        The input array to be downsampled.
    - target_shape: tuple
        The desired shape. Each dimension must be <= the corresponding input dimension.

    Returns:
    - np.ndarray
        The downsampled array with the exact target_shape.
    """
    if len(target_shape) != arr.ndim:
        raise ValueError("Target shape must have the same number of dimensions as input array.")
    
    for i, (s, t) in enumerate(zip(arr.shape, target_shape)):
        if t > s:
            raise ValueError(f"Target shape {target_shape} must not exceed input shape {arr.shape}")

    slices = tuple(
        np.linspace(0, s - 1, t, dtype=int)
        for s, t in zip(arr.shape, target_shape)
    )
    
    # Use np.ix_ for advanced indexing in N-D
    mesh = np.ix_(*slices)
    return arr[mesh]


def get_file_paths(path, ext=".h5", reverse=False):
    if ext in path:
        return [path]
    else:
        paths = sorted(glob(os.path.join(path, "**", f"*{ext}"), recursive=True), reverse=reverse)
        return paths


def upsample_data(data, factor, is_segmentation=True, target_size=None):
    if factor is None and target_size is None:
        print("Need factor or target size!")
        return
    if factor:
        out_shape = tuple(dim * factor for dim in data.shape)
    elif target_size:
        out_shape = target_size
    if is_segmentation:
        output = resize(data, out_shape, preserve_range=True, order=0, anti_aliasing=False).astype(data.dtype)
    else:
        output = resize(data, out_shape, preserve_range=True).astype(data.dtype)
    return output


def export_data(export_path: str, data):
    """Export data to the specified path, determining format from the file extension.
    
    Args:
        data (np.ndarray | dict): The data to save. For HDF5/Zarr, a dict of named datasets is required.
        export_path (str): The file path where the data should be saved.
    
    Raises:
        ValueError: If the file format is unsupported or if data format does not match the expected type.
    """
    ext = export_path.lower().split(".")[-1]

    if ext == "tif":
        if isinstance(data, dict):
            data = next(iter(data.values()))
        if not isinstance(data, np.ndarray):
            raise ValueError("For .tif format, data must be a NumPy array.")
        tifffile.imwrite(export_path, data, dtype=data.dtype, compression="zlib")
        # iio.imwrite(export_path, data, compression="zlib")
    
    elif ext in {"mrc", "rec"}:
        if not isinstance(data, np.ndarray):
            raise ValueError("For .mrc and .rec formats, data must be a NumPy array.")
        with mrcfile.new(export_path, overwrite=True) as mrc:
            mrc.set_data(data.astype(data.dtype))
    
    elif ext == "zarr":
        if not isinstance(data, dict):
            raise ValueError("For .zarr format, data must be a dictionary with dataset names as keys.")
        root = zarr.open(export_path, mode="w")
        for key, value in data.items():
            root.create_dataset(key, data=value.astype(data.dtype))

    elif ext in {"h5", "hdf5"}:
        if not isinstance(data, dict):
            raise ValueError("For .h5 and .hdf5 formats, data must be a dictionary with dataset names as keys.")
        with h5py.File(export_path, "w") as f:
            for key, value in data.items():
                f.create_dataset(key, data=value.astype(value.dtype))
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    print(f"Data successfully exported to {export_path}")


def filter_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Removes small objects and keeps only the largest connected component per label.

    Args:
        segmentation (np.ndarray): Labeled segmentation array.
        min_size (int): Minimum allowed size for connected components.

    Returns:
        np.ndarray: Cleaned segmentation with small components removed.
    """
    unique_labels = np.unique(segmentation)
    cleaned_seg = np.zeros_like(segmentation)

    for lbl in unique_labels:
        if lbl == 0:  # Skip background
            continue
        
        # Isolate current label
        mask = segmentation == lbl

        # Label connected components within this mask
        labeled_mask, num_features = label(mask)

        if num_features == 1:  # If there's only one connected component, keep it
            cleaned_seg[mask] = lbl
            continue

        # Compute sizes of each component
        # sizes = sum_labels(np.array(mask, dtype=np.uint8), np.array(labeled_mask, dtype=np.uint8), index=np.arange(1, num_features + 1))
        sizes = sum_labels(mask, labeled_mask, index=np.arange(1, num_features + 1))

        # Find the largest component
        largest_component = np.argmax(sizes) + 1  # +1 because labels start at 1

        # Keep only the largest component
        cleaned_seg[labeled_mask == largest_component] = lbl

    return cleaned_seg


def filter_small_objects(segmentation: np.ndarray, min_size: int) -> np.ndarray:
    """Removes small connected components from a labeled segmentation.

    Args:
        segmentation (np.ndarray): Labeled segmentation array.
        min_size (int): Minimum allowed size for connected components.

    Returns:
        np.ndarray: Cleaned segmentation with small components removed.
    """
    # Label connected components
    labeled_array, num_features = label(segmentation)
    
    # Compute size of each component
    sizes = sum_labels(segmentation > 0, labeled_array, index=np.arange(1, num_features + 1))
    
    # Find small components
    small_components = np.where(sizes < min_size)[0] + 1  # +1 because labels start at 1
    
    # Remove small components
    mask = np.isin(labeled_array, small_components)
    segmentation[mask] = 0  # Set small components to background (0)
    
    return segmentation


def refine_seg(seg,
               min_size=50000*1,
               area_threshold=1000 * 2,
               block_shape=(128, 256, 256),
               halo=(48, 48, 48)
               ):
    seg = filter_segmentation(seg)
    seg = filter_small_objects(seg, min_size)
    seg = _postprocess_seg_3d(seg, area_threshold=area_threshold, iterations=4, iterations_3d=8)
    return seg


def find_label_file(raw_path: str, label_paths: list) -> str:
    """
    Find the corresponding label file for a given raw file.

    Args:
        raw_path (str): The path to the raw file.
        label_paths (list): A list of label file paths.

    Returns:
        str: The path to the matching label file, or None if no match is found.
    """
    raw_base = os.path.splitext(os.path.basename(raw_path))[0]  # Remove extension

    for label_path in label_paths:
        label_base = os.path.splitext(os.path.basename(label_path))[0]  # Remove extension
        if raw_base in label_base:  # Ensure raw name is contained in label name
            return label_path

    return None  # No match found


def export_to_h5(data, export_path):
    with h5py.File(export_path, mode='a') as h5f:
        for key in data.keys():
            if key in h5f:
                print(f"Skipping {key} as it already exists in {export_path}")
                continue
            h5f.create_dataset(key, data=data[key], compression="gzip")
    print("exported to", export_path)


def _extract_zdim_and_save_h5(data, save_dir, start_z, end_z, prefix="cropped"):
    """
    Crops mitochondria and raw data based on given z-dimension range, then saves them to HDF5.

    Parameters:
    - data (dict): Dictionary with labeled "mitochondria" data and "raw" image data.
    - save_dir (str): Directory to save the extracted regions.
    - start_z (int): Starting slice of the z-dimension.
    - end_z (int): Ending slice of the z-dimension.
    - prefix (str): Prefix for saved files.

    Returns:
    - None (saves files to `save_dir`)
    """
    os.makedirs(save_dir, exist_ok=True)
    export_data = {}

    if "raw" not in data:
        raise ValueError("No 'raw' key found in the dataset.")

    for key, value in data.items():

        # Crop the mitochondria and raw data along the given z range
        cropped = value[start_z:end_z]  # Cropped 
        #raw_cropped = data["raw"][start_z:end_z]  # Corresponding raw data

        # Downscale y and x dimensions
        max_y, max_x = cropped.shape[1], cropped.shape[2]
        cropped = cropped[:, :max_y, :max_x]
        #raw_cropped = raw_cropped[:, :max_y, :max_x]

        # Prepare data for export
        export_data[key] = cropped

    # Save to HDF5
    export_path = os.path.join(save_dir, f"{prefix}_z{start_z}-{end_z}-mito.h5")
    export_to_h5(export_data, export_path)


def export_mrc(filename: str, data: np.ndarray, voxel_size: tuple[float, float, float]):
    """
    Export a 3D NumPy array to an .mrc file with a specified voxel size.

    Args:
        filename (str): Output .mrc file path.
        data (np.ndarray): 3D NumPy array (Z, Y, X).
        voxel_size (tuple[float, float, float]): Voxel size in (Z, Y, X) order.
    """
    assert data.ndim == 3, "Input data must be a 3D array (Z, Y, X)."
    assert len(voxel_size) == 3, "Voxel size must be a tuple of (Z, Y, X)."

    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))  # Ensure data is in correct dtype
        mrc.voxel_size = voxel_size  # Set voxel size metadata
        mrc.update_header_from_data()  # Ensure header consistency
        mrc.header.d = np.array(voxel_size, dtype=np.float32)  # Alternative metadata setting
        mrc.flush()  # Write changes to disk

    print(f"Saved {filename} with voxel size {voxel_size}")


def standardize_channel(raw, channel=0):

    if raw.ndim != 4:
        raise ValueError(f"Expected a 4D input (C, Z, Y, X), got shape {raw.shape}")

    if not (0 <= channel < raw.shape[0]):
        raise ValueError(f"Invalid channel index {channel}, must be in range [0, {raw.shape[0]-1}]")

    raw_norm = np.float32(raw)
    raw_norm[channel] = torch_em.transform.raw.standardize(raw[channel])

    return raw_norm

def normalize_percentile_with_channel(raw, lower=1, upper=99, channel=0):
    """
    Normalize a specific channel of a multi-channel array using percentile normalization.

    Args:
        raw (np.ndarray): Input array of shape (C, Z, Y, X).
        lower (float): Lower percentile for normalization.
        upper (float): Upper percentile for normalization.
        channel (int, optional): The channel index to normalize. Defaults to 0.

    Returns:
        np.ndarray: Normalized array with shape (C, Z, Y, X).
    """
    if raw.ndim != 4:
        raise ValueError(f"Expected a 4D input (C, Z, Y, X), got shape {raw.shape}")

    if not (0 <= channel < raw.shape[0]):
        raise ValueError(f"Invalid channel index {channel}, must be in range [0, {raw.shape[0]-1}]")

    raw_norm = np.float32(raw)
    raw_norm[channel] = torch_em.transform.raw.normalize_percentile(raw[channel], lower=lower, upper=upper)

    return raw_norm


def get_all_datasets(file_path):
    dataset_names = []

    def visit_func(name, obj):
        if isinstance(obj, h5py.Dataset):
            dataset_names.append(name)

    with h5py.File(file_path, 'r') as hdf5_file:
        hdf5_file.visititems(visit_func)

    return dataset_names


def get_filename_and_inter_dirs(file_path, base_path):
    # Extract the base name (filename with extension)
    base_name = os.path.basename(file_path)
    # Split the base name into name and extension to get the filename
    file_name = os.path.splitext(base_name)[0]
    # Get the relative path of file_path from base_path
    relative_path = os.path.relpath(file_path, base_path)
    # Get the intermediate directories by removing the filename from the relative path
    inter_dirs = os.path.dirname(relative_path)
    return file_name, inter_dirs


def create_directories_if_not_exists(base_path, inter_dirs):
    # Construct the full path from base_path and inter_dirs
    full_path = os.path.join(base_path, inter_dirs)
    
    # Check if the path exists
    if not os.path.exists(full_path):
        # If it doesn't exist, create the directories
        os.makedirs(full_path)
        print(f"\nCreated directories: {full_path}")
    else:
        print(f"\nDirectories already exist: {full_path}")


def get_wichmann_data():
    data = [
        "mitos_and_cristae/Otof-KO_M6/KO8_eb2_model.h5",
        "mitos_and_cristae/Otof-KO_M6/KO9_eb11_model.h5",
        "mitos_and_cristae/Otof-KO_M6/KO9_eb13_model.h5",
        "mitos_and_cristae/Otof-KO_M6/KO9_eb4_model.h5",
        "mitos_and_cristae/Otof-KO_M6/KO9_eb6_model.h5",
        "mitos_and_cristae/Otof-KO_M6/KO9_eb9_model.h5",
        "mitos_and_cristae/Otof-KO_M6/M10_eb8_model.h5",
        "mitos_and_cristae/Otof-KO_P10/M1_eb8_model.h5",
        "mitos_and_cristae/Otof-KO_P10/M2_eb10_model.h5",
        "mitos_and_cristae/Otof-KO_P10/M2_eb1_model.h5",
        "mitos_and_cristae/Otof-KO_P10/M2_eb8_model.h5",
        "mitos_and_cristae/Otof-KO_P22/M5_eb3_model.h5",
        "mitos_and_cristae/Otof-KO_P22/M6_eb2_model.h5",
        "mitos_and_cristae/Otof-KO_P22/M7_eb15_model.h5",
        "mitos_and_cristae/Otof-WT_M6/WT40_eb10_model.h5",
        "mitos_and_cristae/Otof-WT_M6/WT40_eb3_model.h5",
        "mitos_and_cristae/Otof-WT_M6/WT40_eb8_model.h5",
        "mitos_and_cristae/Otof-WT_M6/WT41_eb4_model.h5",
        "mitos_and_cristae/Otof-WT_P10/WT13_syn1_model2.h5",
        "mitos_and_cristae/Otof-WT_P10/WT13_syn4_model2.h5",
        "mitos_in_endbuld/Otof_AVCN03_429C_WT_M.Stim_G3_3_model.h5",
        "mitos_in_endbuld/Otof_AVCN03_429C_WT_M.Stim_G3_4_model.h5",
        "mitos_in_endbuld/Otof_AVCN03_429C_WT_M.Stim_G3_5_model.h5",
        "mitos_in_endbuld/Otof_AVCN03_429D_WT_Rest_G3_4_model.h5",
        "mitos_in_endbuld/Otof_AVCN03_429D_WT_Rest_H5_1_35461_model.h5",
        "mitos_in_endbuld/Otof_AVCN03_429D_WT_Rest_H5_3_35461_model.h5",
        "mitos_in_endbuld/Otof_AVCN03_429D_WT_Rest_H5_4_35461_model.h5",
    ]
    for i in range(len(data)):
        # data[i] = "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/extracted/" + data[i]
        data[i] = "/home/freckmann15/data/mitochondria/wichmann/extracted/" + data[i]
    return data


class CombinedDatasets(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_path: Union[List[Any], str, os.PathLike],
        raw_key: str,
        raw2_key: str,
        label_path: Union[List[Any], str, os.PathLike],
        label_key: str,
        patch_shape: Tuple[int, ...],
        raw_transform=None,
        raw2_transform=None,
        label_transform=None,
        label_transform2=None,
        transform=None,
        roi: Optional[dict] = None,
        dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.float32,
        n_samples: Optional[int] = None,
        sampler=None,
        ndim: Optional[int] = None,
        with_channels: bool = False,
        with_label_channels: bool = False,
        with_padding: bool = True,
    ):
        self.ds1 = torch_em.data.SegmentationDataset(
            raw_path,
            raw_key,
            label_path,
            label_key,
            patch_shape=patch_shape,
            raw_transform=raw_transform,
            label_transform=label_transform,
            label_transform2=label_transform2,
            transform=transform,
            roi=roi,
            n_samples=n_samples,
            sampler=sampler,
            ndim=ndim,
            dtype=dtype,
            label_dtype=label_dtype,
            with_channels=with_channels,
            with_label_channels=with_label_channels,
            with_padding=with_padding,
        )
        # Additional raw data key for the second dataset
        self.ds2 = torch_em.data.SegmentationDataset(
            raw_path,
            raw2_key,
            label_path,
            label_key,
            patch_shape=patch_shape,
            raw_transform=raw2_transform,
            transform=transform,
            roi=roi,
            n_samples=n_samples,
            sampler=sampler,
            ndim=ndim,
            dtype=dtype,
            label_dtype=label_dtype,
            with_channels=with_channels,
            with_label_channels=with_label_channels,
            with_padding=with_padding,
        )

    def __len__(self):
        return len(self.ds1)

    def __getitem__(self, index):
        data1 = self.ds1.super().__getitem__(index)
        raw1, labels = data1[0], data1[1]

        data2 = self.ds2.super().__getitem__(index)
        raw2 = data2[0]

        # Return the combined result (raw1, raw2, labels)
        return (raw1, raw2), labels


# not in use atm
def get_loaders(
        data, patch_shape, ndim=3, batch_size=1, n_workers=16, 
        label_transform=None, with_channels=True, with_label_channels=True, 
        rois_dict=None):
    """
    Generates data loaders for training and validation using the given data, patch shape, and other parameters.

    Args:
        data (dict): A dictionary containing the paths to the training and validation data.
        patch_shape (tuple): The shape of the patches to be extracted from the data.
        ndim (int, optional): The number of dimensions of the data. Defaults to 3.
        batch_size (int, optional): The batch size for the data loaders. Defaults to 1.
        n_workers (int, optional): The number of workers for data loading. Defaults to 16.
        label_transform (callable, optional): A callable that transforms the labels. Defaults to None.
        with_channels (bool, optional): Whether to include the channels in the data. Defaults to True.
        with_label_channels (bool, optional): Whether to include the label channels in the data. Defaults to True.
        rois_dict (dict, optional): A dictionary containing the regions of interest (ROIs) for training and validation. Defaults to None.

    Returns:
        tuple: A tuple containing the training data loader and the validation data loader.
    """
    if rois_dict is not None:
        train_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key="raw",
            label_paths=data["train"], label_key="labels/mitochondria",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            rois=rois_dict["train"]
        )
        val_loader = torch_em.default_segmentation_loader(
            raw_paths=data["val"], raw_key="raw",
            label_paths=data["val"], label_key="labels/mitochondria",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            rois=rois_dict["val"]
        )
    else:
        train_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key="raw",
            label_paths=data["train"], label_key="labels/mitochondria",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
        )
        val_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key="raw",
            label_paths=data["val"], label_key="labels/mitochondria",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
        )
    
    return train_loader, val_loader


def remove_prefix_from_keys(state_dict, prefix="_orig_mod."):
    """
    Removes the specified prefix from the beginning of all keys in a dictionary.

    Args:
        state_dict (dict): The dictionary containing keys with the prefix to remove.
        prefix (str): The string prefix to remove from the beginning of keys.

    Returns:
        dict: A new dictionary with the prefix removed from all keys.
    """
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            # Remove the prefix and store the value in the new dictionary with the modified key
            new_key = key[len(prefix):]
            filtered_state_dict[new_key] = value
        else:
            # If the key doesn't start with the prefix, keep it as is
            filtered_state_dict[key] = value
    return filtered_state_dict


def get_rois_coordinates_skimage(file, label_key, min_shape, euler_threshold=None, min_amount_pixels=None):
    """
    Calculates the average coordinates for each unique label in a 3D label image using skimage.regionprops.

    Args:
        file (h5py.File): Handle to the open HDF5 file.
        label_key (str): Key for the label data within the HDF5 file.
        min_shape (tuple): A tuple representing the minimum size for each dimension of the ROI.

    Returns:
        dict: A dictionary mapping unique labels to lists of average coordinates
            for each dimension, or None if no labels are found.
    """

    label_data = file[label_key]
    label_shape = label_data.shape

    # Ensure data type is suitable for regionprops (usually uint labels)
    # if label_data.dtype != np.uint:
    #     label_data = label_data.astype(np.uint).value

    # Find connected regions (objects) using regionprops
    regions = regionprops(label_data)

    # Check if any regions were found
    if not regions:
        return None

    label_extents = {}
    for region in regions:
        if euler_threshold is not None:
            if region.euler_number != euler_threshold:
                continue
        if min_amount_pixels is not None:
            if region["area"] < min_amount_pixels:
                continue
        
        # # Extract relevant information for ROI calculation
        label = region.label  # Get the label value
        min_coords = region.bbox[:3]  # Minimum coordinates (excluding intensity channel)
        max_coords = region.bbox[3:6]  # Maximum coordinates (excluding intensity channel)

        # Clip coordinates and create ROI extent (similar to previous approach)
        clipped_min_coords = np.clip(min_coords, 0, label_shape[0] - min_shape[0])
        clipped_max_coords = np.clip(max_coords, min_shape[1], label_shape[1])
        roi_extent = tuple(slice(min_val, min_val + min_shape[dim]) for dim, (min_val, max_val) in enumerate(zip(clipped_min_coords, clipped_max_coords)))

        # Check for labels within the ROI extent (new part)
        roi_data = file[label_key][roi_extent]
        amount_label_pixels = np.count_nonzero(roi_data)
        if amount_label_pixels < 100:  # Check for any non-zero values (labels)
            continue  # Skip this ROI if no labels present
        if min_amount_pixels is not None:
            if amount_label_pixels < min_amount_pixels:
                continue

        label_extents[label] = roi_extent

    return label_extents


def get_data_paths(data_dir, data_format="*.h5"):
    # check if data_dir is file
    if os.path.isfile(data_dir) or fnmatch.fnmatch(data_dir, data_format):
        return [data_dir]
    data_paths = glob(os.path.join(data_dir, "**", data_format), recursive=True)
    return data_paths


def get_data_paths_and_rois(data_dir, min_shape,
                            data_format="*.h5",
                            image_key="raw",
                            label_key_mito="labels/mitochondria",
                            label_key_cristae="labels/cristae",
                            with_thresholds=True):
    """
    Retrieves all HDF5 data paths, their corresponding image and label data keys,
    and extracts Regions of Interest (ROIs) for labels.

    Args:
        data_dir (str): Path to the directory containing HDF5 files.
        data_format (str, optional): File format to search for (default: "*.h5").
        image_key (str, optional): Key for image data within the HDF5 file (default: "raw").
        label_key_mito (str, optional): Key for the first label data (default: "labels/mitochondria").
        label_key_cristae (str, optional): Key for the second label data (default: "labels/cristae").
        roi_halo (tuple, optional): A fixed tuple representing the halo radius for ROIs in each dimension (default: (2, 3, 1)).

    Returns:
        tuple: A tuple containing three lists:
            - data_paths: List of paths to all HDF5 files in the directory and subdirectories.
            - rois_list: List containing ROIs for each valid HDF5 file.
                - Each ROI is a list of tuples representing slices for each dimension.
    """

    data_paths = glob(os.path.join(data_dir, "**", data_format), recursive=True)
    rois_list = []
    new_data_paths = [] # one data path for each ROI

    for data_path in data_paths:
        try:
            # Open the HDF5 file in read-only mode
            with h5py.File(data_path, "r") as f:
                # Check for existence of image and label datasets (considering key flexibility)
                if image_key not in f:
                    print(f"Warning: Key(s) missing in {data_path}. Skipping {image_key}")
                    continue

                #label_data_mito = f[label_key_mito][()] if label_key_mito is not None else None

                # Extract ROIs (assuming ndim of label data is the same as image data)
                if with_thresholds:
                    rois = get_rois_coordinates_skimage(f, label_key_mito, min_shape, min_amount_pixels=100) # euler_threshold=1,
                else:
                    rois = get_rois_coordinates_skimage(f, label_key_mito, min_shape, euler_threshold=None, min_amount_pixels=None)
                for label_id, roi in rois.items():
                    rois_list.append(roi)
                    new_data_paths.append(data_path)
        except OSError:
            print(f"Error accessing file: {data_path}. Skipping...")

    return new_data_paths, rois_list


def split_data_paths_to_dict(data_paths, rois_list, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
    """
    Splits data paths and ROIs into training, validation, and testing sets without shuffling.

    Args:
        data_paths (list): List of paths to all HDF5 files.
        rois_dict (dict): Dictionary mapping data paths (or indices) to corresponding ROIs.
        train_ratio (float, optional): Proportion of data for training (0.0-1.0) (default: 0.8).
        val_ratio (float, optional): Proportion of data for validation (0.0-1.0) (default: 0.1).
        test_ratio (float, optional): Proportion of data for testing (0.0-1.0) (default: 0.1).

    Returns:
        tuple: A tuple containing two dictionaries:
            - data_split: Dictionary containing "train", "val", and "test" keys with data paths.
            - rois_split: Dictionary containing "train", "val", and "test" keys with corresponding ROIs.

    Raises:
        ValueError: If the sum of ratios exceeds 1 or the length of data paths and number of ROIs don't match.
    """
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Sum of train, validation, and test ratios must equal 1.0.")
    num_data = len(data_paths)
    if rois_list is not None:
        if len(rois_list) != num_data:
            raise ValueError(f"Length of data paths and number of ROIs in the dictionary must match: len rois {len(rois_list)}, len data_paths {len(data_paths)}")

    train_size = int(num_data * train_ratio)
    val_size = int(num_data * val_ratio)  # Optional validation set
    test_size = num_data - train_size - val_size
    remaining = num_data - (train_size + val_size + test_size)
    if remaining > 0:
        test_size += remaining

    data_split = {
        "train": data_paths[:train_size],
        "val": data_paths[train_size:train_size+val_size],
        "test": data_paths[train_size+val_size:]
    }

    if rois_list is not None:
        rois_split = {
            "train": rois_list[:train_size],
            "val": rois_list[train_size:train_size+val_size],
            "test": rois_list[train_size+val_size:]
        }

        return data_split, rois_split
    else:
        return data_split


def get_filename_from_path(path):
    """
    Extracts the filename from a given path by splitting on the last '/'.

    Args:
        path: The path string.

    Returns:
        The filename portion of the path.
    """
    return path.split("/")[-1]


def split_data_paths(data_paths, key_dicts, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    """
    Splits data paths and key information into training, validation, and testing sets.

    Args:
        data_paths (list): List of paths to all HDF5 files.
        key_dicts (list): List of dictionaries containing image and label data keys for each HDF5 file.
        train_ratio (float, optional): Proportion of data for training (0.0-1.0) (default: 0.8).
        val_ratio (float, optional): Proportion of data for validation (0.0-1.0) (default: 0.1).
        test_ratio (float, optional): Proportion of data for testing (0.0-1.0) (default: 0.1).
        seed (int, optional): Random seed for shuffling data paths (default: None).

    Returns:
        tuple: A tuple containing three dictionaries:
            - train_data: Dictionary containing "data_paths" and "key_dicts" for training data.
            - val_data: Dictionary containing "data_paths" and "key_dicts" for validation data (if applicable).
            - test_data: Dictionary containing "data_paths" and "key_dicts" for testing data.

    Raises:
        ValueError: If the sum of ratios exceeds 1.
    """

    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Sum of train, validation, and test ratios must equal 1.0.")

    if seed is not None:
        random.seed(seed)
    random.shuffle(data_paths)
    random.shuffle(key_dicts)

    num_data = len(data_paths)
    train_size = int(num_data * train_ratio)
    val_size = int(num_data * val_ratio)  # Optional validation set
    test_size = num_data - train_size - val_size

    train_data = {
        "data_paths": data_paths[:train_size],
        "key_dicts": key_dicts[:train_size]
    }
    val_data = {"data_paths": [], "key_dicts": []}  # Optional validation set
    test_data = {
        "data_paths": data_paths[train_size+val_size:],
        "key_dicts": key_dicts[train_size+val_size:]
    }

    if val_size > 0:
        val_data = {
            "data_paths": data_paths[train_size:train_size+val_size],
            "key_dicts": key_dicts[train_size:train_size+val_size]
        }

    return train_data, val_data, test_data


def get_data_paths_and_keys(data_dir, data_format="*.h5", image_key="raw", label_key="labels/mitochondria"):
    """
    Retrieves all HDF5 data paths and their corresponding image and label data keys.

    Args:
        data_dir (str): Path to the directory containing HDF5 files.
        data_format (str, optional): File format to search for (default: "*.h5").
        image_key (str, optional): Key for image data within the HDF5 file (default: "raw").
        label_key (str, optional): Key for label data within the HDF5 file (default: "labels/mitochondria").

    Returns:
        tuple: A tuple containing two lists:
            - data_paths: List of paths to all HDF5 files in the directory and subdirectories.
            - key_dicts: List of dictionaries containing image and label data keys for each HDF5 file.
                - Each dictionary has keys: "image_key" and "label_key".
    """

    data_paths = glob(os.path.join(data_dir, "**", data_format), recursive=True)
    key_dicts = []

    for data_path in data_paths:
        try:
            # Open the HDF5 file in read-only mode
            with h5py.File(data_path, "r") as f:
                # Check for existence of image and label datasets (considering key flexibility)
                if image_key not in f or (label_key is not None and label_key not in f):
                    print(f"Warning: Key(s) missing in {data_path}. Skipping...")
                    continue
                key_dicts.append({"image_key": image_key, "label_key": label_key})
        except OSError:
            print(f"Error accessing file: {data_path}. Skipping...")

    return data_paths, key_dicts


def create_directory(directory):
    """
    Creates a directory if it doesn't already exist.

    Args:
        directory (str): The path to the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def visualize_data_napari(data):
    """
    Visualizes the 3D raw data and all labels using napari.

    Args:
        data (dict): Dictionary containing loaded raw data ("raw" key) 
                    and labels ("labels" dictionary with loaded labels).
    """
    # Create a napari viewer
    viewer = napari.Viewer()
    
    if "raw" in data.keys():
        if isinstance(data["raw"], torch.Tensor):
            raw_data = data["raw"].cpu().detach().numpy()
        else:
            raw_data = data["raw"]

        viewer.add_image(raw_data, name="Raw")

    if "label" in data.keys():
        if isinstance(data["label"], torch.Tensor):
            label_data = data["label"].cpu().detach().numpy()
        else:
            label_data = data["label"]

        viewer.add_labels(label_data.astype(int), name="Label")  # Ensure labels are integers
    if "label2" in data.keys():
        if isinstance(data["label"], torch.Tensor):
            label_data = data["label2"].cpu().detach().numpy()
        else:
            label_data = data["label2"]

        viewer.add_labels(label_data.astype(int), name="Label2") 
    if "pred1" in data.keys():
        if isinstance(data["pred1"], torch.Tensor):
            label_data = data["pred1"].cpu().detach().numpy()
        else:
            label_data = data["pred1"]

        viewer.add_image(label_data.astype(float), blending="additive", name="Foreground Prediction")
    if "pred2" in data.keys():
        if isinstance(data["pred2"], torch.Tensor):
            label_data = data["pred2"].cpu().detach().numpy()
        else:
            label_data = data["pred2"]

        viewer.add_image(label_data.astype(float), blending="additive", name="Boundary Prediction")

    # Show the napari viewer
    napari.run()


def run_prediction(data, model, block_shape=[32, 512, 512], halo=[8, 32, 32]):
    """
    Run a prediction using a trained model on the given data.

    Args:
        data (array-like): The input data on which predictions are to be made.
        model (torch.nn.Module): The loaded model.
        block_shape (List[int], optional): The block shape to use for prediction.
            Defaults to [32, 256, 256].
        halo (List[int], optional): The halo shape to use for prediction.
            Defaults to [8, 32, 32].

    Returns:
        array-like: The predicted output from the model.
    """

    gpu_ids = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    with torch.no_grad():
        pred = predict_with_halo(
            data, model, gpu_ids=gpu_ids,
            block_shape=block_shape, halo=halo,
            preprocess=None,
        )
    return pred


def get_label_transform(label_data):
    """
    Transforms all the label ids in a label image to ones.
    Args:
        label_data (np.ndarray): A 3D array representing the label data.
            - Assumed to use integer values to represent unique labels (adjust as needed).

    Returns:
        np.ndarray: A 3D array with all label ids replaced by ones.
    """
    return np.where(label_data != 0, 1, label_data)


def get_loss_function(loss_name, affinities=False):
    loss_names = ["bce", "ce", "dice"]
    if isinstance(loss_name, str):
        assert loss_name in loss_names, f"{loss_name}, {loss_names}"
        if loss_name == "dice":
            loss_function = torch_em.loss.DiceLoss()
        elif loss_name == "ce":
            loss_function = nn.CrossEntropyLoss()
        elif loss_name == "bce":
            loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = loss_name

    # we need to add a loss wrapper for affinities
    if affinities:
        loss_function = torch_em.loss.LossWrapper(
            loss_function, transform=torch_em.loss.ApplyAndRemoveMask()
        )
    return loss_function


def get_all_metadata(data_dir, data_format="*.h5"):
    """
    Retrieves metadata for all HDF5 files in a directory and its subdirectories.

    Args:
        data_dir (str): Path to the directory containing HDF5 files.
        data_format (str, optional): File format to search for (default: "*.h5").

    Returns:
        list: A list of dictionaries containing the retrieved metadata for 
              each HDF5 file, or None if errors encountered.
    """

    # Get all file paths matching the format
    data_paths = glob(os.path.join(data_dir, "**", data_format), recursive=True)
    metadata_list = []

    # Loop through all files and get metadata
    for data_path in tqdm(data_paths):
        metadata = get_data_metadata(data_path)
        if metadata:
            metadata_list.append(metadata)

    return metadata_list


def get_data_metadata(data_path):
    """
    Retrieves metadata about the data in an HDF5 file without loading it entirely.

    Args:
        data_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary with filename as key and nested dictionary containing 
              the retrieved metadata, or None if error.
    """

    try:
        # Open the HDF5 file in read-only mode
        with h5py.File(data_path, "r") as f:

            # Check for existence of datasets
            if "raw" not in f:
                print(f"Error: 'raw' dataset not found in {data_path}")
                return None  # Indicate error

            # Get image size (assuming 'raw' dataset holds the 3D image)
            image_size = f["raw"].shape

            # Check for labels group
            has_cristae_label = False
            if "labels" in f:
                labels_group = f["labels"]
                has_cristae_label = "cristae" in labels_group.keys()

            # Calculate exact min and max values using NumPy functions
            min_value = np.amin(f["raw"], axis=None)
            max_value = np.amax(f["raw"], axis=None)

            # Calculate basic statistics
            average_value = np.mean(f["raw"])  # Calculate mean across all axes
            try:
                std_dev = np.std(f["raw"])  # Calculate standard deviation
            except RuntimeWarning:
                # Handle potential runtime warnings (e.g., all elements equal)
                std_dev = None

            # Extract filename from data_path
            filename = os.path.basename(data_path)  # Get filename from path

            # Create nested dictionary for metadata
            metadata = {
                "image_size": list(image_size),
                "has_cristae_label": has_cristae_label,  # Boolean
                "value_range": (float(min_value), float(max_value)),
                "average_value": float(average_value),
                "std_dev": float(std_dev) if std_dev is not None else None,
            }
            print(metadata)

            # Return dictionary with filename as key and metadata nested
            return {filename: metadata}

    except Exception as e:
        print(f"Error getting metadata for {data_path}: {e}")
        return None


def load_metadata(data_path):
    """
    Loads metadata from a single YAML file in a directory.

    Args:
        data_path (str): Path to the directory containing the metadata.yaml file.

    Returns:
        dict: The loaded metadata from the YAML file, 
            or None if no file found or parsing error.
    """

    # Construct the expected filename for metadata
    filename = os.path.join(data_path, "metadata.yaml")

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"Error: No metadata.yaml file found in {data_path}")
        return None

    # Load metadata from the file using YAML (safe_load)
    try:
        with open(filename, 'r') as f:
            metadata = yaml.full_load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {filename}: {e}")

    average_values = []
    with_cristae_labels = []
    images_sizes = []
    std_devs = []
    value_ranges = []
    for item in metadata:
        for name, data_dict in item.items():
            for key, value in data_dict.items():
                if key == "average_value":
                    average_values.append(value)
                if key == "has_cristae_label":
                    with_cristae_labels.append(value)
                if key == "image_size":
                    images_sizes.append(value)
                if key == "std_dev":
                    std_devs.append(value)
                if key == "value_range":
                    value_ranges.append(value)
        #print(len(average_values) , len(with_cristae_labels) , len(images_sizes) , len(std_devs) , (value_ranges))
    # Assuming average_values and std_devs contain numerical data
    average_average_value = sum(average_values) / len(average_values)
    average_std_dev = sum(std_devs) / len(std_devs)
    average_image_sizes = []
    for dim in range(len(images_sizes[0])):  # Assuming all images have the same number of dimensions
        # Calculate the average for each dimension
        dimension_values = [image_size[dim] for image_size in images_sizes]
        average_image_sizes.append(sum(dimension_values) / len(dimension_values))

    min_values = [range_tuple[0] for range_tuple in value_ranges]  # Extract minimum values
    max_values = [range_tuple[1] for range_tuple in value_ranges]  # Extract maximum values

    average_min_value = sum(min_values) / len(min_values)
    average_max_value = sum(max_values) / len(max_values)

    print("Average minimum value:", average_min_value)
    print("Average maximum value:", average_max_value)

    # Print the calculated averages
    print("Average Average Value:", average_average_value)
    print("Average Std Dev:", average_std_dev)
    print("Images with cristae labels:", sum(with_cristae_labels), "/", len(with_cristae_labels)) # sum(label for label in with_cristae_labels if label)
    print("Average Image Sizes:", average_image_sizes)
    # Return None if any exceptions occur
    return metadata