import argparse
import napari
import synapse.util as util
from config import *
import os
from skimage import io
import h5py
from glob import glob
from tifffile import TiffFile
import numpy as np


def visualize_lucchi_data(args):
    """
    Visualizes a file from the specified lucchi_data_dir using napari.

    Args:
        lucchi_data_dir (str): Path to the directory containing the data file.

    Returns:
        napari.Viewer, None: A napari viewer instance if a file is successfully visualized,
                              None otherwise.
    """
    lucchi_data_dir = args.lucchi_data_dir

    # Check if lucchi_data_dir exists
    if not os.path.exists(lucchi_data_dir):
        print(f"Error: Directory '{lucchi_data_dir}' does not exist.")
        return None

    # Get a list of files in the directory
    files = glob(os.path.join(lucchi_data_dir, "**", "*.h5"), recursive=True)  # lucchi_data_dir + "*.h5")  # os.listdir(lucchi_data_dir)

    # Check if there are any files
    if not files:
        print(f"Error: Directory '{lucchi_data_dir}' is empty.")
        return None
    print("Found files:", files)
    filename = files[0]

    try:
        with h5py.File(filename, 'r') as f:
            # Access the dataset containing your image (replace 'image_dataset' with the actual name)
            image = f['raw'][:]
            label = f['labels'][:]
            print("image shape", image.shape)
            print("label shape", label.shape)
    except (IOError, OSError) as e:
        print(f"Error loading file '{filename}': {e}")
        return None
    try:
        with TiffFile('/home/freckmann15/data/predictions/lucchi_test.tif') as tif:
            print("File Information:")
            print(f"  - Number of Image Series: {len(tif.series)}")
            print(f"  - Shape of First Image Series: {tif.series[0].shape}")
            pred1 = tif.series[0].asarray()

    except ValueError as e:
        print(e)

    
    vis_data = {
            "raw": image,
            "label": label,
            "pred1": pred1,
        }
    util.visualize_data_napari(vis_data)



def main():
    parser = argparse.ArgumentParser(description="Visualize data with napari")
    parser.add_argument("--lucchi_data_dir", type=str, default=TEST_DATA_DIR, help="Path to the lucchi data director")
    parser.add_argument("--pred_file, type", type=str, default="/home/freckmann15/data/predictions/lucchi_test.tif", help="Path to the pred file")

    # Parse arguments
    args = parser.parse_args()
    
    # Example usage (assuming lucchi_data_dir is set correctly)
    visualize_lucchi_data(args)

if __name__ == "__main__":
    main()