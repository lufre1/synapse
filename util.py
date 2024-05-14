import os
from glob import glob
import h5py
from tqdm import tqdm
import napari
import torch_em
import torch.nn as nn
import numpy as np
import yaml

# Define the data path and filename
# data_path = "/scratch-grete/projects/nim00007/data/mitochondria/moebius/em_tomograms_v1/170-PLP-wt/170_2_rec.h5"
# data_format = "*.h5"


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
            metadata = yaml.full_load(f) #yaml.safe_load(f)
            return metadata
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {filename}: {e}")

    # Return None if any exceptions occur
    return None


def visualize_data_napari(data):
    """
    Visualizes the 3D raw data and available labels using napari.

    Args:
        data (dict): Dictionary containing loaded raw data ("raw" key) 
                    and labels ("labels" dictionary with loaded labels).
    """

    # Extract the raw data
    raw_data = data["raw"]

    # Create a napari viewer
    viewer = napari.Viewer()

    # Add raw data as a volume
    viewer.add_image(raw_data, name="Raw Data")

    # Add all available labels from "labels" data
    for label_name, label_data in data["labels"].items():
        viewer.add_labels(label_data, name=label_name)

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
