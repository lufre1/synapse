import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from torch.utils.tensorboard import SummaryWriter
import yaml
import os

import torch_em
import torch_em.data.datasets as torchem_data
from torch_em.model import UNet3d
from torch_em.util.debug import check_loader, check_trainer

# Import your util.py for data loading
import util
from config import *
from unet import UNet3D as MyUnet3d


def main():
    # Load data from the specified path (assuming util.py handles single file)
    data_dir = DATA_DIR
    test_data_dir = TEST_DATA_DIR
    all_data = None  # util.load_all_hdf5_data(data_dir, amount=2)
    visualize = False

    if all_data and visualize:
        # Assuming there's at least one entry (modify if needed)
        for i in range(min(len(all_data), 3)):
            data = all_data[i]
            util.visualize_data_napari(data)

    else:
        print("No visualization with napari.")
    metadata_list = util.get_all_metadata(data_dir, data_format="*.h5")
    # Save metadata to YAML file
    metadata_file = os.path.join(data_dir, "metadata.yaml")
    with open(metadata_file, "w") as f:
        yaml.dump(metadata_list, f, default_flow_style=False)

    print(f"Metadata saved to: {metadata_file}")
    return None
    # Define experiment and model parameters
    experiment_name = "cristae-and-mito-net"
    batch_size = 1
    patch_shape = (32, 256, 256)
    loss_name = "dice"
    metric_name = "dice"
    n_iterations = 1
    learning_rate = 1.0e-4
    loss_function = util.get_loss_function(loss_name)
    metric_function = util.get_loss_function(metric_name)
    in_channels, out_channels = 1, 1
    initial_features = 32
    final_activation = None
    if final_activation is None and loss_name == "dice":
        final_activation = "Sigmoid"

    print("Creating 3d UNet with", in_channels, "input channels and", out_channels, "output channels.")
    model = UNet3d(
        in_channels=in_channels, out_channels=out_channels, initial_features=initial_features,
        final_activation=final_activation
    )

    # create loader
    train_loader = torchem_data.get_lucchi_loader(test_data_dir, split="train", patch_shape=patch_shape,
                                                  batch_size=batch_size, download=True
                                                  )
    val_loader = torchem_data.get_lucchi_loader(test_data_dir, split="train", patch_shape=patch_shape,
                                                batch_size=batch_size, download=True
                                                )

    trainer = torch_em.default_segmentation_trainer(
        name=experiment_name, model=model,
        train_loader=train_loader, val_loader=val_loader,
        loss=loss_function, metric=metric_function,
        learning_rate=learning_rate,
        mixed_precision=True,
        log_image_interval=50,
        # logger=None
    )
    trainer.fit(n_iterations)


if __name__ == "__main__":
    main()
