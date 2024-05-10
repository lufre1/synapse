import os
from glob import glob
import h5py
from tqdm import tqdm
import napari
import torch_em
import torch.nn as nn

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


def load_all_hdf5_data(data_dir, data_format="*.h5"):
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
