import os
from glob import glob
import h5py
from tqdm import tqdm
import napari
import torch
import torch_em
import torch.nn as nn
import numpy as np
import yaml

# Define the data path and filename
# data_path = "/scratch-grete/projects/nim00007/data/mitochondria/moebius/em_tomograms_v1/170-PLP-wt/170_2_rec.h5"
# data_format = "*.h5"

import random


def get_data_paths_and_rois(data_dir, data_format="*.h5",
                            image_key="raw",
                            label_key_mito="labels/mitochondria",
                            label_key_cristae="labels/cristae"):
    """
    Retrieves all HDF5 data paths, their corresponding image and label data keys,
    and extracts Regions of Interest (ROIs) for labels.

    Args:
        data_dir (str): Path to the directory containing HDF5 files.
        data_format (str, optional): File format to search for (default: "*.h5").
        image_key (str, optional): Key for image data within the HDF5 file (default: "raw").
        label_key_mito (str, optional): Key for the first label data (default: "labels/mitochondria").
        label_key_cristae (str, optional): Key for the second label data (default: "labels/cristae").

    Returns:
        tuple: A tuple containing three lists:
            - data_paths: List of paths to all HDF5 files in the directory and subdirectories.
            - rois_list: List containing ROIs for each valid HDF5 file.
                - Each ROI is a list of tuples or slices representing where labels are True (or 1).
    """

    data_paths = glob(os.path.join(data_dir, "**", data_format), recursive=True)
    # key_dicts = []
    rois_list = []

    for data_path in data_paths:
        try:
            # Open the HDF5 file in read-only mode
            with h5py.File(data_path, "r") as f:
                # Check for existence of image and label datasets (considering key flexibility)
                if image_key not in f or (label_key_mito is not None and label_key_mito not in f):
                    print(f"Warning: Key(s) missing in {data_path}. Skipping {image_key}")
                    continue

                # Get label data (assuming labels are boolean or 1/0 for True/False)
                label_data_mito = f[label_key_mito][()] if label_key_mito is not None else None
                #label_data_cristae = f[label_key_cristae][()] if label_key_cristae is not None else None

                # Extract ROIs (assuming ndim of label data is the same as image data)
                rois = []
                if label_data_mito is not None:
                    # Find non-zero elements (assuming True is represented by non-zero values)
                    # roi = tuple(slice(co - rh, co + rh) for co, rh in zip(coord, roi_halo))
                    non_zero_indices = np.nonzero(label_data_mito) # .astype(int)
                    for dim in range(label_data_mito.ndim):
                        rois.append(slice(non_zero_indices[dim].min(), non_zero_indices[dim].max() + 1))  # +1 for inclusive upper bound
                # if label_data_cristae is not None:
                #     # Repeat for cristae label (if it exists)
                #     non_zero_indices = np.nonzero(label_data_cristae)
                #     for dim in range(label_data_cristae.ndim):
                #         rois.append(slice(non_zero_indices[dim].min(), non_zero_indices[dim].max() + 1))

                # Create a dictionary for this file's key information
                # key_dict = {
                #     "image_key": image_key,
                #     "label_key_mito": label_key_mito,
                #     "label_key_cristae": label_key_cristae,
                # }
                # key_dicts.append(key_dict)
                rois_list.append(rois)  # Add ROIs for this file

        except OSError:
            print(f"Error accessing file: {data_path}. Skipping...")

    return data_paths, rois_list


def split_data_paths_without_key_dicts(data_paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    """
    Splits data paths into training, validation, and testing sets.

    Args:
        data_paths (list): List of paths to all HDF5 files.
        train_ratio (float, optional): Proportion of data for training (0.0-1.0) (default: 0.8).
        val_ratio (float, optional): Proportion of data for validation (0.0-1.0) (default: 0.1).
        test_ratio (float, optional): Proportion of data for testing (0.0-1.0) (default: 0.1).
        seed (int, optional): Random seed for shuffling data paths (default: None).

    Returns:
        tuple: A tuple containing three dictionaries:
            - train_data: Dictionary containing "data_paths" for training data.
            - val_data: Dictionary containing "data_paths" for validation data (if applicable).
            - test_data: Dictionary containing "data_paths" for testing data.

    Raises:
        ValueError: If the sum of ratios exceeds 1.
    """

    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Sum of train, validation, and test ratios must equal 1.0.")

    if seed is not None:
        random.seed(seed)
    random.shuffle(data_paths)

    num_data = len(data_paths)
    train_size = int(num_data * train_ratio)
    val_size = int(num_data * val_ratio)  # Optional validation set
    test_size = num_data - train_size - val_size

    train_data = {"data_paths": data_paths[:train_size]}
    val_data = {"data_paths": []}  # Optional validation set
    test_data = {"data_paths": data_paths[train_size+val_size:]}

    if val_size > 0:
        val_data = {"data_paths": data_paths[train_size:train_size+val_size]}

    return train_data, val_data, test_data


def split_data_paths_to_dict(data_paths, rois_list, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Splits data paths and ROIs into training, validation, and testing sets without shuffling.

    Args:
        data_paths (list): List of paths to all HDF5 files.
        rois_list (list): List of ROIs corresponding to each data path.
        train_ratio (float, optional): Proportion of data for training (0.0-1.0) (default: 0.8).
        val_ratio (float, optional): Proportion of data for validation (0.0-1.0) (default: 0.1).
        test_ratio (float, optional): Proportion of data for testing (0.0-1.0) (default: 0.1).

    Returns:
        tuple: A tuple containing two dictionaries:
            - data_split: Dictionary containing "train", "val", and "test" keys with data paths.
            - rois_split: Dictionary containing "train", "val", and "test" keys with corresponding ROIs.

    Raises:
        ValueError: If the sum of ratios exceeds 1 or the length of data paths and ROIs don't match.
    """

    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Sum of train, validation, and test ratios must equal 1.0.")

    if len(data_paths) != len(rois_list):
        raise ValueError("Length of data paths and ROIs must be the same.")

    num_data = len(data_paths)
    train_size = int(num_data * train_ratio)
    val_size = int(num_data * val_ratio)  # Optional validation set
    test_size = num_data - train_size - val_size

    data_split = {
        "train": data_paths[:train_size],
        "val": data_paths[train_size:train_size+val_size],
        "test": data_paths[train_size+val_size:]
    }
    rois_split = {
        "train": rois_list[:train_size],
        "val": rois_list[train_size:train_size+val_size],
        "test": rois_list[train_size+val_size:]
    }

    if val_size == 0:
        # Remove empty val key if validation is not used
        del data_split["val"]
        del rois_split["val"]

    return data_split, rois_split


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


def extract_data(data_list, train_ratio, test_ratio, label_key="mitochondria"):
    """
    Extracts images and labels for training, validation, and testing from a data list.

    Args:
        data_list (list): List of dictionaries containing loaded data from HDF5 files.
        train_ratio (float): Proportion of data for training (0.0-1.0).
        test_ratio (float): Proportion of data for testing (0.0-1.0).
        label_key (str, optional): Key for label data within the dictionary (default: "mitochondria").

    Returns:
        tuple: A tuple containing three dictionaries:
            - train_data: Dictionary containing training images and labels.
            - val_data: Dictionary containing validation images and labels (if applicable).
            - test_data: Dictionary containing testing images and labels.
    """

    num_images = len(data_list)
    train_size = int(num_images * train_ratio)
    val_size = int(num_images * (1 - train_ratio - test_ratio))  # Optional validation set
    test_size = num_images - train_size - val_size

    train_images, train_labels = [], []
    val_images, val_labels = [], []  # Optional validation set
    test_images, test_labels = [], []

    for i in range(num_images):
        data_dict = data_list[i]
        image = data_dict["raw"]
        label = data_dict["labels"][label_key]

    # Efficiently distribute data based on pre-calculated sizes
    if i < train_size:
        train_images.append(image)
        train_labels.append(label)
    elif i < train_size + val_size:  # Optional validation set
        val_images.append(image)
        val_labels.append(label)
    else:
        test_images.append(image)
        test_labels.append(label)

    return {
        "train_data": {"images": train_images, "labels": train_labels},
        "val_data": {"images": val_images, "labels": val_labels} if val_size > 0 else None,  # Optional validation set
        "test_data": {"images": test_images, "labels": test_labels}
    }


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


def load_all_hdf5_data(data_dir, data_format="*.h5", amount=None):
    """
    Loads all HDF5 data files from a directory and its subdirectories.

    Args:
        data_dir (str): Path to the directory containing HDF5 files.
        data_format (str, optional): File format to search for (default: "*.h5").

    Returns:
        list: A list of dictionaries containing loaded data (raw and labels) 
                from each HDF5 file in the directory and its subdirectories.
    """

    # Get all file paths matching the format in the main directory and subdirectories
    data_paths = glob(os.path.join(data_dir, "**", data_format), recursive=True)

    # List to store loaded data information
    all_data = []

    for data_path in tqdm(data_paths):
        # Load data from each file (unchanged logic)
        data = load_single_hdf5_data(data_path)

        if data is not None:  # Check if data loaded successfully
            # Extract filename without extension
            filename = data_path.split("/")[-1].split(".")[0]
            all_data.append({"filename": filename, **data})  # Unpack data dictionary
        if amount is not None and amount == len(all_data):
            return all_data
    return all_data


def load_single_hdf5_data(data_path):
    """
    Loads raw data and available labels from a single HDF5 file.

    Args:
        data_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary containing loaded raw data and all available labels 
            (or None if error).
    """

    # Open the HDF5 file in read-only mode
    with h5py.File(data_path, "r") as f:

        # Check for existence of datasets
        if "raw" not in f:
            print(f"Error: 'raw' dataset not found in {data_path}")
            return None  # Indicate error

        # Get the raw data as a NumPy array
        raw_data = f["raw"][()]

        # Load all datasets within the "labels" group (if it exists)
        labels_data = {}
        if "labels" in f:
            labels_group = f["labels"]
            for key in labels_group.keys():
                labels_data[key] = labels_group[key][()]

        # Return data as a dictionary
        return {"raw": raw_data, "labels": labels_data}


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


def visualize_data_napari(data):
    """
    Visualizes the 3D raw data and all labels using napari.

    Args:
        data (dict): Dictionary containing loaded raw data ("raw" key) 
                    and labels ("labels" dictionary with loaded labels).
    """
    if isinstance(data["raw"], torch.Tensor):
        raw_data = data["raw"].cpu().detach().numpy()
    else:
        raw_data = data["raw"]

    # Create a napari viewer
    viewer = napari.Viewer()

    # Add raw data as a volume
    viewer.add_image(raw_data, name="Raw Data")
    
    if isinstance(data["label"], torch.Tensor):
        label_data = data["label"].cpu().detach().numpy()
    else:
        label_data = data["label"]

    viewer.add_labels(label_data.astype(int), name="Label")  # Ensure labels are integers
            

    # Show the napari viewer
    napari.run()
# def visualize_data_napari(data):
#     """
#     Visualizes the 3D raw data and available labels using napari.

#     Args:
#         data (dict): Dictionary containing loaded raw data ("raw" key) 
#                     and labels ("labels" dictionary with loaded labels).
#     """
#     if isinstance(data, torch.Tensor):
#         raw_data = data.cpu().detach().numpy()
#     else:
#         raw_data = data
#     #print(raw_data)
#     # # Extract the raw data
#     raw_data = data["raw"].cpu().detach().numpy()

#     # Create a napari viewer
#     viewer = napari.Viewer()

#     # Add raw data as a volume
#     viewer.add_image(raw_data, name="Raw Data")

#     # Add all available labels from "labels" data
#     # for label_name, label_data in data["labels"].items():
#     #     viewer.add_labels(label_data, name=label_name)
    
#     label = data["label"].cpu().detach().numpy()
#     #print(label.shape)
#     viewer.add_labels(label.astype(int), name="Label")
    

#     # Show the napari viewer
#     napari.run()


# Example usage (assuming util.py is in the same directory as your main script)
# data_dir = "/scratch-grete/projects/nim00007/data/mitochondria/moebius/em_tomograms_v1/170-PLP-wt"
# all_data = load_all_hdf5_data(data_dir)

# if all_data:
#     # Process all loaded data (access raw data and labels from each dictionary in all_data)
#     for entry in all_data:
#         data = entry["raw"]
#         labels = entry["labels"]
#         filename = entry["filename"]
# else:
#     print("No HDF5 data files found in the specified directory.")
