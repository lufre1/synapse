import torch
import numpy as np
from torch_em.model import UNet3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import napari
# Import your util.py for data loading
import util
from unet import UNet3D


def visualize_data(data):
    """
    Visualizes the 3D raw data using Matplotlib.

    Args:
        data (dict): Dictionary containing loaded raw data ("raw" key).
    """

    # Extract the raw data
    raw_data = data["raw"]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Isolate a specific slice (adjust slice indices as needed)
    slice_data = raw_data[50, :, :]  # Assuming 3D data (x, y, z)

    # Plot the slice data as a surface
    X, Y = np.meshgrid(np.arange(len(slice_data[0])), np.arange(len(slice_data)))
    ax.plot_surface(X, Y, slice_data, cmap='viridis')  # Adjust colormap as desired

    # Set axis labels and title (adjust as needed)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Raw Data Intensity")
    ax.set_title("3D Raw Data Slice")

    # Show the plot
    plt.show()


def visualize_data_napari(data):
    """
    Visualizes the 3D raw data using napari.

    Args:
        data (dict): Dictionary containing loaded raw data ("raw" key).
    """

    # Extract the raw data
    raw_data = data["raw"]

    # Create a napari viewer
    viewer = napari.Viewer()

    # Add raw data as a volume
    viewer.add_image(raw_data, name="Raw Data")

    # You can add additional layers here (e.g., labels if available)
    viewer.add_labels(data["labels"])

    # Show the napari viewer
    # viewer.show()
    napari.run()


def main():
    # Load data from the specified path (assuming util.py handles single file)
    data_dir = "/home/freckmann15/data/mitochondria/moebius/em_tomograms_v1/170-PLP-wt"
    all_data = util.load_all_hdf5_data(data_dir)

    if all_data:
        # Assuming there's at least one entry (modify if needed)
        data = all_data[0]
        # visualize_data(data)
        visualize_data_napari(data)
        
    else:
        print("No HDF5 data files found in the specified directory.")


if __name__ == "__main__":
    main()
