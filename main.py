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


def main():
    # Load data from the specified path (assuming util.py handles single file)
    data_dir = DATA_DIR
    lucchi_data_dir = TEST_DATA_DIR
    #all_data = util.load_all_hdf5_data(data_dir, amount=None) # None
    data_paths, rois_dict = util.get_data_paths_and_rois(data_dir)#util.get_data_paths_and_keys(data_dir)
    data, rois_dict = util.split_data_paths_to_dict(data_paths, rois_dict, train_ratio=.5, val_ratio=0.5, test_ratio=0)
    # split_data_paths(data_paths, key_dicts, train_ratio=0.5, val_ratio=0.5, test_ratio=0, seed=None)
    visualize = False
    
    #util.check_h5_data_correctness(data_dir)
    
    
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

    #util.load_metadata(data_dir)

    # Define experiment and model parameters
    experiment_name = "cristae-and-mito-net"
    batch_size = 1
    n_workers = 4 if torch.cuda.is_available() else 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} with {n_workers} workers.")
    label_transform = None
    patch_shape = (32, 256, 256)
    #patch_shape = (64, 512, 512)
    #patch_shape = (128, 1024, 1024)
    loss_name = "dice"
    metric_name = "dice"
    ndim = 3
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

    #print(data["train"])
    with_channels = False
    with_label_channels = False

    print("train", len(data["train"]), "val", len(data["val"]))

    train_loader = torch_em.default_segmentation_loader(
        raw_paths=data["train"], raw_key="raw", # raw_key=train_data["key_dicts"][0]["image_key"],
        label_paths=data["train"], label_key="labels/mitochondria", # label_key=train_data["key_dicts"][0]["label_key"],
        patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
        label_transform=label_transform, num_workers=n_workers,
        with_channels=with_channels, with_label_channels=with_label_channels,
        rois=rois_dict["train"]
    )
    val_loader = torch_em.default_segmentation_loader(
        raw_paths=data["val"], raw_key="raw", #raw_key=val_data["key_dicts"][0]["image_key"],
        label_paths=data["val"], label_key="labels/mitochondria", #label_key=val_data["key_dicts"][0]["label_key"],
        patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
        label_transform=label_transform, num_workers=n_workers,
        with_channels=with_channels, with_label_channels=with_label_channels,
        rois=rois_dict["val"]
    )
    image, label = next(iter(train_loader))

    vis_data = {
        "raw": image,
        "label": label
    }
    util.visualize_data_napari(vis_data)

    trainer = torch_em.default_segmentation_trainer(
        name=experiment_name, model=model,
        train_loader=train_loader, val_loader=val_loader,
        loss=loss_function, metric=metric_function,
        learning_rate=learning_rate,
        mixed_precision=True,
        log_image_interval=50,
        device=device,
        # logger=None
    )
    #check_loader(train_loader, n_samples=1)
    #check_trainer(trainer, n_samples=1)
    trainer.fit(n_iterations)


if __name__ == "__main__":
    main()
