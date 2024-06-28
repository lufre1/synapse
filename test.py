import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import h5py
import torch_em.transform
import yaml
import os
import random
import argparse
import time
import torch_em
import torch_em.data.datasets as torchem_data
from torch_em.model import UNet3d, AnisotropicUNet
from torch_em.util.debug import check_loader, check_trainer

# Import your util.py for data loading
import util
import data_classes
from config import *

def test():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to the data directory")
    parser.add_argument("--lucchi_data_dir", type=str, default=TEST_DATA_DIR, help="Path to the lucchi data directory (optional)")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="Path to save data")
    parser.add_argument("--visualize", action="store_true", default=False, help="Visualize data with napari")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(32, 448, 448), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--n_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint used to load model's state_dict")
    parser.add_argument("--experiment_name", type=str, default="default-mito-net", help="Name that is used for the experiment and store the model's weights")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to be used")
    parser.add_argument("--feature_size", type=int, default=32, help="Initial feature size of the 3D UNet")
    parser.add_argument("--file_path", type=str, default="", help="File path to a specific file to segment")
    
    # Parse arguments
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    n_iterations = args.n_iterations
    learning_rate = args.learning_rate
    data_dir = args.data_dir
    lucchi_data_dir = args.lucchi_data_dir
    visualize = args.visualize
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    patch_shape = args.patch_shape
    initial_features = args.feature_size
    save_dir = args.save_dir
    file_path = args.file_path

    n_workers = 4 if torch.cuda.is_available() else 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Experiment: {experiment_name}\n")
    print(f"Using {device} with {n_workers} workers.")
    label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
    loss_name = "dice"
    metric_name = "dice"
    ndim = 3

    loss_function = util.get_loss_function(loss_name)
    metric_function = util.get_loss_function(metric_name)
    in_channels, out_channels = 1, 2
    depth = 4
    gain = 2
    scale_factors = [[2, 2, 2]] * depth
    scale_factors = [
        [1, 2, 2],
        [1, 2, 2],
        [2, 2, 2],
        [2, 2, 2]
    ]
    final_activation = None
    if final_activation is None and loss_name == "dice":
        final_activation = "Sigmoid"
    
    model = AnisotropicUNet(
        in_channels=in_channels, out_channels=out_channels, initial_features=initial_features,
        final_activation=final_activation, scale_factors=scale_factors, gain=gain
    )

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state"] #torch.device("cpu")
        state_dict = util.remove_prefix_from_keys(state_dict)
        print("\n", state_dict.keys(), "\n")
        model.load_state_dict(state_dict)
        model.to(device)
        
    print(model)
    data_paths = []
    ### load data
    if file_path is None or file_path == "":
        data_paths = util.get_data_paths(data_dir)
    else:
        data_paths.append(file_path)
    #data, rois_dict = util.split_data_paths_to_dict(data_paths, rois_dict, train_ratio=.8, val_ratio=0.2, test_ratio=0)
    
    # test_loader = torch_em.default_segmentation_loader(
    #     raw_paths=data["train"], raw_key="raw",
    #     label_paths=data["train"], label_key="labels/mitochondria",
    #     patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
    #     label_transform=label_transform, num_workers=n_workers,
    #     rois=rois_dict["train"]
    # )
    
    # Create the "predictions" directory inside DATA_DIR
    predictions_dir = os.path.join(save_dir, "predictions")
    util.create_directory(predictions_dir)
    print(f"Using {device} with {n_workers} workers.")
    down_scale_factor = 2
    for i, data_path in enumerate(data_paths):
        # image, label = next(iter(test_loader))
        # pred = model(image)
        # pred_foreground = pred[:, 0, :, :]
        # pred_boundaries = pred[:, 1, :, :]
        with h5py.File(data_path, "r") as f:
            print("file number and file path:", i, data_path)
            image = f["raw"][:]
            # label = label["labels/mitochondria"]
            image = torch_em.transform.raw.standardize(image, mean=np.mean(image), std=np.std(image))
            pred = util.run_prediction(image, model)
            prediction_filepath = os.path.join(predictions_dir, f"{experiment_name}_prediction_{util.get_filename_from_path(data_path)}")
            with h5py.File(prediction_filepath, "w") as prediction_file:
                prediction_file.create_dataset("prediction", data=pred[:, :, ::down_scale_factor, ::down_scale_factor])
        # pred_foreground = pred[:, 0, :, :]
        # pred_boundaries = pred[:, 1, :, :]

        # vis_data = {
        #     "raw": image,
        #     "label": label,
        #     "pred1": pred_foreground,
        #     "pred2": pred_boundaries
        # }
        # util.visualize_data_napari(vis_data)


if __name__ == "__main__":
    test()
