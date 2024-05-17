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
    Visualizes the 3D raw data and available labels using napari.

    Args:
        data (dict): Dictionary containing loaded raw data ("raw" key) 
                    and labels ("labels" dictionary with loaded labels).
    """
    if isinstance(data, torch.Tensor):
        raw_data = data.cpu().detach().numpy()
    else:
        raw_data = data
    #print(raw_data)
    # # Extract the raw data
    raw_data = data["raw"].cpu().detach().numpy()

    # Create a napari viewer
    viewer = napari.Viewer()

    # Add raw data as a volume
    viewer.add_image(raw_data, name="Raw Data")

    # Add all available labels from "labels" data
    # for label_name, label_data in data["labels"].items():
    #     viewer.add_labels(label_data, name=label_name)
    
    label = data["label"].cpu().detach().numpy()
    print("Image shape: ", raw_data.shape, "Label shape: ", label.shape)
    viewer.add_labels(label.astype(int), name="Label")
    

    # Show the napari viewer
    napari.run()


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


def check_h5_data_correctness(data_dir, data_format="*.h5", amount=None):
    """
    Checks for basic correctness of data in HDF5 files within a directory.

    Args:
        data_dir (str): Path to the directory containing HDF5 files.
        data_format (str, optional): File format to search for (default: "*.h5").
        amount (int, optional): Limit the number of files to check (default: None).

    Returns:
        None: The function doesn't return anything, it prints messages 
            indicating any encountered errors.
    """

    # Get data paths and key information
    data_paths, key_dicts = get_data_paths_and_keys(data_dir, data_format)

    for data_path, key_dict in tqdm(zip(data_paths, key_dicts)):
        # Load data using existing function
        data = load_single_hdf5_data(data_path)

        if data is None:
            print(f"Error: Failed to load data from {data_path}.")
            continue  # Skip to the next file if loading fails

        # Print detailed information for files with issues
        missing_labels = [label for label in key_dict["label_key"] if label not in data["labels"]]
        if missing_labels:
            print(f"\nWarning: Missing labels in {data_path}: {', '.join(missing_labels)}")
            print("  - File structure:")
            for key, value in data.items():
                print(f"    - '{key}': {type(value)}  {value.shape if hasattr(value, 'shape') else ''}")  # Print data type and shape (if applicable)
            print(f"  - Raw image shape: {data['raw'].shape if 'raw' in data else 'Not found'}")

            # Access and print label data shapes
            for label_key, label_data in data["labels"].items():
                if label_key == "mitochondria":  # Check for your desired label
                    print(f"  - Label '{label_key}' shape: {label_data.shape if hasattr(label_data, 'shape') else 'Not found'}")

        if amount is not None and len(data_paths) >= amount:
            break  # Limit the number of files checked (if specified)

    print("Finished checking data correctness.")