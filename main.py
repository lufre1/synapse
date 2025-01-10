import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import yaml
import os
import random
import argparse
import time
import torch_em
# import torch_em.data.datasets as torchem_data
from torch_em.data import MinInstanceSampler
from torch_em.model import AnisotropicUNet
from torch_em.util.debug import check_loader, check_trainer
from tqdm import tqdm
import random

# Import your util.py for data loading
import synapse.util as util
# import data_classes
from config import DATA_DIR, SAVE_DIR, TEST_DATA_DIR
# from unet import UNet3D


def main():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR, help="Path to the data directory")
    parser.add_argument("--data_dir2", type=str, default=None, help="Path to a second data directory")
    parser.add_argument("--lucchi_data_dir", type=str, default=TEST_DATA_DIR, help="Path to the lucchi data directory (optional)")
    parser.add_argument("--visualize", action="store_true", default=False, help="Visualize data with napari")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(32, 256, 256), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--n_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint used to load model's state_dict")
    parser.add_argument("--experiment_name", type=str, default="default-mito-net", help="Name that is used for the experiment and store the model's weights")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to be used")
    parser.add_argument("--feature_size", type=int, default=32, help="Initial feature size of the 3D UNet")
    parser.add_argument("--without_rois", type=bool, default=False, help="Train without Regions Of Interest (ROI)")
    parser.add_argument("--early_stopping", type=int, default=10, help="Number of epochs without improvement before stopping training")

    # Parse arguments
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    n_iterations = args.n_iterations
    learning_rate = args.learning_rate
    data_dir = args.data_dir
    data_dir2 = args.data_dir2
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    patch_shape = args.patch_shape
    initial_features = args.feature_size
    with_rois = not args.without_rois
    early_stopping = args.early_stopping

    n_workers = 12 if torch.cuda.is_available() else 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Experiment: {experiment_name}\n")
    print(f"Using {device} with {n_workers} workers.")
    label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
    raw_transform = None  # util.custom_raw_transform

    loss_name = "dice"
    metric_name = "dice"
    ndim = 3

    loss_function = util.get_loss_function(loss_name)
    metric_function = util.get_loss_function(metric_name)
    in_channels, out_channels = 1, 2
    # depth = 4
    gain = 2

    # scale_factors = 4*[[2, 2, 2]]
    scale_factors = [
        [1, 2, 2],
        [1, 2, 2],
        [2, 2, 2],
        [2, 2, 2]
    ]

    final_activation = None
    if final_activation is None and loss_name == "dice":
        final_activation = "Sigmoid"

    # load data paths etc.
    start_time = time.time()
    print(f"Start time {time.ctime()}")
    print(f"Loading Data paths and ROIs if with_rois={with_rois}...")

    if with_rois:
        data_paths, rois_dict = util.get_data_paths_and_rois(data_dir, min_shape=patch_shape, with_thresholds=True)
        random.seed(42)
        random.shuffle(data_paths)
        data, rois_dict = util.split_data_paths_to_dict(data_paths, rois_dict, train_ratio=.8, val_ratio=0.2, test_ratio=0)
    else:
        data_paths = util.get_data_paths(data_dir)
        #data_paths = util.get_wichmann_data()
        #print(data_paths)
        if data_dir2 is not None:
            data_paths2 = util.get_data_paths(data_dir2)
            data_paths.extend(data_paths2)

        for path in data_paths:
            if "combined" in path:
                data_paths.remove(path)
        random.seed(42)
        random.shuffle(data_paths)
        # data_paths.sort(reverse=True)
        data = util.split_data_paths_to_dict(data_paths, rois_list=None, train_ratio=.8, val_ratio=0.15, test_ratio=0.05)

    end_time = time.time()
    # Calculate execution time in seconds
    execution_time = end_time - start_time
    print(f"Data and ROI preprocessing execution time: {execution_time:.6f} seconds")

    print("Creating 3d UNet with", in_channels, "input channels and", out_channels, "output channels.")
    # UNet3d
    model = AnisotropicUNet(
        in_channels=in_channels, out_channels=out_channels, initial_features=initial_features,
        final_activation=final_activation, gain=gain, scale_factors=scale_factors
    )
    print("does checkpoint exist at", os.path.join(SAVE_DIR, "checkpoints", experiment_name, "best.pt"), "?")
    print(os.path.exists(os.path.join(SAVE_DIR, "checkpoints", experiment_name, "best.pt")))
    if checkpoint_path or os.path.exists(os.path.join(SAVE_DIR, "checkpoints", experiment_name, "best.pt")):
        if not checkpoint_path:
            checkpoint_path = os.path.join(SAVE_DIR, "checkpoints", experiment_name)
        model = torch_em.util.load_model(checkpoint=checkpoint_path, device=device)
        print("loaded model from checkpoint:", os.path.join(SAVE_DIR, "checkpoints", experiment_name))
        # state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))["model_state"]
        # model.load_state_dict(state_dict)

        model.to(device)
    print(model)
    with_channels = False
    with_label_channels = False
    sampler = MinInstanceSampler(p_reject=0.95)

    print("train", len(data["train"]), "val", len(data["val"]), "test", len(data["test"]))
    print("data['test']", data["test"])

    if with_rois:
        train_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key="raw",
            label_paths=data["train"], label_key="labels/mitochondria",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            rois=rois_dict["train"]
        )
        val_loader = torch_em.default_segmentation_loader(
            raw_paths=data["val"], raw_key="raw",
            label_paths=data["val"], label_key="labels/mitochondria",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            rois=rois_dict["val"]
        )
    else:
        train_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key="raw",
            label_paths=data["train"], label_key="labels/mitochondria",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            raw_transform=raw_transform,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            sampler=sampler
        )
        val_loader = torch_em.default_segmentation_loader(
            raw_paths=data["val"], raw_key="raw",
            label_paths=data["val"], label_key="labels/mitochondria",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            raw_transform=raw_transform,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            sampler=sampler
        )
    # print("all val paths:")
    # for path in data["val"]:
    #     print(path)

    val_iter = iter(val_loader)
    try:
        image, label = next(val_iter)
        print("Validation data exists:", image.shape, label.shape)
    except StopIteration:
        print("val_loader is empty!")
        return 0
    # for i in tqdm(range(100)):
    #     image, label = next(iter(train_loader))
    #     if image.shape is not None:
    #         print("image shape and label shape", image.shape, label.shape)
        # vis_data = {
        #     "raw": image,
        #     "label": label
        # }
    #     util.visualize_data_napari(vis_data)

    trainer = torch_em.default_segmentation_trainer(
        name=experiment_name, model=model,
        train_loader=train_loader, val_loader=val_loader,
        loss=loss_function, metric=metric_function,
        learning_rate=learning_rate,
        mixed_precision=True,
        log_image_interval=50,
        device=device,
        compile_model=False,
        save_root=SAVE_DIR,
        early_stopping=early_stopping,
        # logger=None
    )
    # check_loader(train_loader, n_samples=10)
    # check_trainer(trainer, n_samples=1)
    trainer.fit(n_iterations)


if __name__ == "__main__":
    main()
