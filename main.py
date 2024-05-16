import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import yaml
import os
import random

import torch_em
import torch_em.data.datasets as torchem_data
from torch_em.model import UNet3d
from torch_em.util.debug import check_loader, check_trainer

# Import your util.py for data loading
import util
import data_classes
from config import *
from unet import UNet3D as MyUnet3d


def label_aware_crop(image, mask, patch_shape):
    """
    Performs random crop while ensuring at least one label is present (for 3D data).

    Args:
        image (np.ndarray): 3D image data.
        mask (np.ndarray): 3D label data.
        patch_shape (tuple): Desired output size of the crop (spatial dimensions).

    Returns:
        tuple: A tuple containing the cropped image and label data (both np.ndarray).
    """

    image_shape = image.shape
    mask_shape = mask.shape

    # Validate input shapes
    assert image_shape == mask_shape, "Image and mask shapes must be the same."

    # Calculate maximum possible crop coordinates considering patch size and image dimensions
    max_i = image_shape[0] - patch_shape[0] + 1
    max_j = image_shape[1] - patch_shape[1] + 1
    max_k = image_shape[2] - patch_shape[2] + 1

    while True:
        # Generate random crop coordinates within valid ranges
        i = random.randint(0, max_i)
        j = random.randint(0, max_j)
        k = random.randint(0, max_k)

        # Crop the image and mask based on the generated coordinates
        cropped_image = image[i:i+patch_shape[0], j:j+patch_shape[1], k:k+patch_shape[2]]
        cropped_mask = mask[i:i+patch_shape[0], j:j+patch_shape[1], k:k+patch_shape[2]]

        # Check if at least one non-zero element exists in the cropped label
        if cropped_mask.sum() > 0:
            return cropped_image, cropped_mask

    # If the loop exits without finding a valid crop (unlikely scenario), raise an error
    raise ValueError("Unable to find a crop with at least one label.")


def main():
    # Load data from the specified path (assuming util.py handles single file)
    data_dir = DATA_DIR
    lucchi_data_dir = TEST_DATA_DIR
    #all_data = util.load_all_hdf5_data(data_dir, amount=None) # None
    data_paths, key_dicts = util.get_data_paths_and_keys(data_dir)
    train_data, val_data, test_data = util.split_data_paths(data_paths, key_dicts, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None)
    visualize = False
    # if all_data and visualize:
    #     # Assuming there's at least one entry (modify if needed)
    #     for i in range(min(len(all_data), 3)):
    #         data = all_data[i]
    #         util.visualize_data_napari(data)

    # else:
    #     print("No visualization with napari.")

    # metadata_list = util.get_all_metadata(data_dir, data_format="*.h5")
    # # Save metadata to YAML file
    # metadata_file = os.path.join(data_dir, "metadata.yaml")
    # with open(metadata_file, "w") as f:
    #     yaml.dump(metadata_list, f, default_flow_style=False)

    # print(f"Metadata saved to: {metadata_file}")

    util.load_metadata(data_dir)

    # Define experiment and model parameters
    experiment_name = "cristae-and-mito-net"
    batch_size = 1
    n_workers = 2
    label_transform = None
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

    # # create lucchi loader
    # train_loader = torchem_data.get_lucchi_loader(lucchi_data_dir, split="train", patch_shape=patch_shape,
    #                                               batch_size=batch_size, download=True
    #                                               )
    # val_loader = torchem_data.get_lucchi_loader(lucchi_data_dir, split="train", patch_shape=patch_shape,
    #                                             batch_size=batch_size, download=True
    #                                               )

    # train_ratio = 0.8  # 80% for training
    # test_ratio = 0.1   # 10% for testing
    # data_sets = util.extract_data(all_data, train_ratio, test_ratio, label_key="mitochondria")
    # # create Datasets
    # train_dataset = data_classes.CustomDataset(
    #     data_sets["train_data"]["images"], data_sets["train_data"]["labels"],
    #     patch_shape=patch_shape, label_aware_crop=label_aware_crop
    #     )
    # val_dataset = data_classes.CustomDataset(
    #     data_sets["val_data"]["images"], data_sets["val_data"]["labels"],
    #     patch_shape=patch_shape, label_aware_crop=label_aware_crop
    #     )
    # # create DataLoader
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    # val_loader = DataLoader(val_dataset, batch_size=batch_size) 
    train_loader = torch_em.default_segmentation_loader(
        raw_paths=train_data["data_paths"], raw_key="raw",
        label_paths=train_data["data_paths"], label_key="labels/mitochondria",
        patch_shape=patch_shape, ndim=2, batch_size=batch_size,
        label_transform=label_transform, num_workers=n_workers,
    )
    val_loader = torch_em.default_segmentation_loader(
        raw_paths=val_data["data_paths"], raw_key="raw",
        label_paths=val_data["data_paths"], label_key="labels/mitochondria",
        patch_shape=patch_shape, ndim=2, batch_size=batch_size,
        label_transform=label_transform, num_workers=n_workers,
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
    check_loader(train_loader, n_samples=1)
    check_trainer(trainer, n_samples=1)
    #trainer.fit(n_iterations)


if __name__ == "__main__":
    main()
