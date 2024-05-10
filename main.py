import torch
import numpy as np
from torch_em.model import UNet3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from torch.utils.tensorboard import SummaryWriter
import torch_em

# Import your util.py for data loading
import util
from unet import UNet3D


def main():
    # Load data from the specified path (assuming util.py handles single file)
    # data_dir = "/home/freckmann15/data/mitochondria/moebius/em_tomograms_v1/170-PLP-wt"
    # data_dir = "/home/freckmann15/data/mitochondria/cooper/example_cristae"
    data_dir = "/home/freckmann15/data/mitochondria/"
    all_data = util.load_all_hdf5_data(data_dir)

    if all_data:
        # Assuming there's at least one entry (modify if needed)
        data = all_data[0]

        util.visualize_data_napari(data)

    else:
        print("No HDF5 data files found in the specified directory.")


if __name__ == "__main__":
    main()
