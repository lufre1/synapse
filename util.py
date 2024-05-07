import os
from glob import glob
import h5py
from tqdm import tqdm

# Define the data path and filename
data_path = "/scratch-grete/projects/nim00007/data/mitochondria/moebius/em_tomograms_v1/170-PLP-wt/170_2_rec.h5"
data_format = "*.h5"


def load_all_hdf5_data(data_dir, data_format="*.h5"):
    """
    Loads all HDF5 data files from a directory.

    Args:
        data_dir (str): Path to the directory containing HDF5 files.
        data_format (str, optional): File format to search for (default: "*.h5").

    Returns:
        list: A list of dictionaries containing loaded data (raw and labels) 
              from each HDF5 file in the directory.
    """

    # Get all file paths matching the format in the main directory
    data_paths = glob(os.path.join(data_dir, data_format))

    # List to store loaded data information
    all_data = []

    for data_path in tqdm(data_paths):
        # Load data from each file
        data = load_single_hdf5_data(data_path)

        if data is not None:  # Check if data loaded successfully
            # Extract filename without extension
            filename = data_path.split("/")[-1].split(".")[0]
            all_data.append({"filename": filename, **data})  # Unpack data dictionary

    return all_data


def load_single_hdf5_data(data_path):
    """
    Loads raw data and labels from a single HDF5 file.

    Args:
        data_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary containing loaded raw data and labels (or None if error).
    """

    # Open the HDF5 file in read-only mode
    with h5py.File(data_path, "r") as f:

        # Check for existence of datasets (optional)
        if "labels" in f and "raw" in f:

            # Access the "labels" group
            labels_group = f["labels"]

            # Assuming a single dataset named "mitochondria" within the "labels" group (based on h5dump)
            labels_data = labels_group["mitochondria"][()]

            # Get the raw data as a NumPy array
            raw_data = f["raw"][()]

            # Return data as a dictionary
            return {"raw": raw_data, "labels": labels_data}
        else:
            print(f"Error: 'labels' or 'raw' datasets not found in {data_path}")
            return None  # Indicate error for this file


# Example usage (assuming util.py is in the same directory as your main script)
data_dir = "/scratch-grete/projects/nim00007/data/mitochondria/moebius/em_tomograms_v1/170-PLP-wt"
all_data = load_all_hdf5_data(data_dir)

# if all_data:
#     # Process all loaded data (access raw data and labels from each dictionary in all_data)
#     for entry in all_data:
#         data = entry["raw"]
#         labels = entry["labels"]
#         filename = entry["filename"]
# else:
#     print("No HDF5 data files found in the specified directory.")
