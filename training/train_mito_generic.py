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
from synapse_net.training.supervised_training import supervised_training, get_supervised_loader, get_3d_model
from synapse_net.file_utils import read_ome_zarr

# Import your util.py for data loading
import synapse.util as util
import synapse.cellmap_util as cutil
import synapse.label_utils as lutil
# import data_classes
SAVE_DIR = "/mnt/lustre-grete/usr/u12103/mitochondria/volem"


def main():
    parser = argparse.ArgumentParser(description="3D UNet for medium organelle segmentation")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the data directory")
    parser.add_argument("--raw_file_extension", type=str, default="*.h5", help="File extension to read data")
    parser.add_argument("--label_dir", type=str, default=None, help="Path to a second data directory | labels")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(32, 512, 512), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--n_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint used to load model's state_dict")
    parser.add_argument("--experiment_name", type=str, default="3d_unet_mitochondria", help="Name that is used for the experiment and store the model's weights")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size to be used")
    parser.add_argument("--feature_size", type=int, default=32, help="Initial feature size of the 3D UNet")
    parser.add_argument("--early_stopping", type=int, default=10, help="Number of epochs without improvement before stopping training")
    parser.add_argument("--raw_key", type=str, default="raw", help="Raw key to be used for training e.g. raw_crop")
    parser.add_argument("--label_key", type=str, default=None, help="Label key to be used for training e.g. label_crop/all | None for tiff")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples to be used for training per dataset")
    parser.add_argument("--second_data_dir", "-sdd", type=str, default=None, help="Path to a second data directory")
    parser.add_argument("--second_label_dir", type=str, default=None, help="Path to a second label directory")
    parser.add_argument("--third_data_dir", "-tdd", type=str, default=None, help="Path to a third data directory - loads hdf5")
    parser.add_argument("--with_rois", "-wr", default=False, action='store_true', help="Use to train with ROIs (manually set ROI in script!!)")
    parser.add_argument("--use_synapse_training", "-ust", default=False, action='store_true')

    # Parse arguments
    args = parser.parse_args()
    n_iterations = args.n_iterations
    data_dir = args.data_dir
    label_dir = args.label_dir
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    patch_shape = args.patch_shape

    n_workers = 8 if torch.cuda.is_available() else 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Experiment: {experiment_name}\n")
    print(f"Using {device} with {n_workers} workers.")
    
    if torch.cuda.is_available():
        os.makedirs(SAVE_DIR, exist_ok=True)
    # load model from checkpoint if exists
    if os.path.exists(os.path.join(SAVE_DIR, "checkpoints", experiment_name, "best.pt")):
        checkpoint_path = os.path.join(SAVE_DIR, "checkpoints", experiment_name, "best.pt")
        print("Checkpoint exists, loading model from checkpoint", checkpoint_path)
    elif args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
        print("Loading model from given checkpoint", checkpoint_path)
    else:
        checkpoint_path = None

    raw_transform = torch_em.transform.raw.normalize_percentile  # util.custom_raw_transform

    in_channels, out_channels = 1, 2

    # load data paths etc.
    start_time = time.time()
    print(f"Start time {time.ctime()}")

    data_paths = util.get_data_paths(data_dir, data_format=args.raw_file_extension)
    if args.second_data_dir is not None:
        print("Adding", args.second_data_dir, "to data paths with", args.raw_file_extension)
        data_paths += util.get_data_paths(args.second_data_dir, data_format=args.raw_file_extension)
    if args.third_data_dir is not None:
        print("Adding", args.third_data_dir, "to data paths with", args.raw_file_extension)
        data_paths += util.get_data_paths(args.third_data_dir, data_format=args.raw_file_extension)

    print("data_paths", data_paths)
    if label_dir is not None:
        label_paths = util.get_data_paths(label_dir, data_format="*.tif")
        if args.second_label_dir is not None:
            label_paths += util.get_data_paths(args.second_label_dir, data_format="*.tif")
    else:
        label_paths = None
        random.seed(42)
        random.shuffle(data_paths)
    data = util.split_data_paths_to_dict(data_paths, rois_list=None, train_ratio=.85, val_ratio=0.15, test_ratio=0.0)

    end_time = time.time()
    # Calculate execution time in seconds
    execution_time = end_time - start_time

    sampler = MinInstanceSampler(p_reject=0.95)
    # label_transform = lutil.CombinedLabelTransform(add_binary_target=True, dilation_footprint=np.ones((3, 3)))
    label_transform = torch_em.transform.BoundaryTransform(add_binary_target=True)

    # print("label paths", label_paths)
    print("Path for this model", os.path.join(SAVE_DIR, experiment_name))
    with_channels = False
    with_label_channels = False
    loss_name = "dice"
    metric_name = "dice"
    ndim = 3
    scale_factors = [[1, 2, 2], [1, 2, 2], [2, 2, 2], [2, 2, 2]]

    loss_function = util.get_loss_function(loss_name)
    metric_function = util.get_loss_function(metric_name)

    if label_paths is None and args.use_synapse_training:
        print(f"Data preprocessing execution time: {execution_time:.6f} seconds")
        print("Creating 3d UNet with", in_channels, "input channels and", out_channels, "output channels.")
        print("Saving model to", SAVE_DIR)
        print("data['train']", data["train"])
        print("data['val']", data["val"])
        print("data['test']", data["test"])
        print("train", len(data["train"]), "val", len(data["val"]), "test", len(data["test"]))
        model = util.get_3d_model(out_channels=out_channels, in_channels=in_channels, scale_factors=scale_factors,
                                  initial_features=args.feature_size)
        if checkpoint_path is not None:
            # model.load_state_dict(torch.load(checkpoint_path))
            model = torch_em.util.load_model(checkpoint=checkpoint_path, device=device)
            print("Successfully loaded model from checkpoint", checkpoint_path)

        train_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key=args.raw_key,
            label_paths=data["train"], label_key=args.label_key,
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            raw_transform=raw_transform,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            sampler=sampler, n_samples=args.n_samples,
        )
        val_loader = torch_em.default_segmentation_loader(
            raw_paths=data["val"], raw_key=args.raw_key,
            label_paths=data["val"], label_key=args.label_key,
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            raw_transform=raw_transform,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            sampler=sampler, n_samples=args.n_samples,
        )
        trainer = torch_em.default_segmentation_trainer(
            name=experiment_name, model=model,
            train_loader=train_loader, val_loader=val_loader,
            loss=loss_function, metric=metric_function,
            learning_rate=args.learning_rate,
            mixed_precision=False,
            log_image_interval=50,
            device=device,
            compile_model=False,
            save_root=SAVE_DIR,
            early_stopping=args.early_stopping,
        )

        trainer.fit(n_iterations)
        # supervised_training(
        #     name=experiment_name,
        #     train_paths=data["train"],
        #     raw_key=args.raw_key,
        #     val_paths=data["val"],
        #     label_key=args.label_key,
        #     patch_shape=patch_shape,
        #     save_root=SAVE_DIR,
        #     batch_size=batch_size,
        #     n_iterations=n_iterations,
        #     sampler=sampler,
        #     out_channels=out_channels,
        #     label_transform=label_transform,
        #     raw_transform=raw_transform,
        #     check=(False if torch.cuda.is_available() else True),
        # )
    else:
        print("Creating 3d UNet with", in_channels, "input channels and", out_channels, "output channels.")
        print("data paths", data_paths)
        print("label paths", label_paths)
        # use torch em trainer and torchem loader to customize paths
        model = util.get_3d_model(out_channels=out_channels, in_channels=in_channels, scale_factors=scale_factors,
                                  initial_features=args.feature_size)
        if checkpoint_path is not None:
            # model.load_state_dict(torch.load(checkpoint_path))
            model = torch_em.util.load_model(checkpoint=checkpoint_path, device=device)
            print("Successfully loaded model from checkpoint", checkpoint_path)
        train_loader = torch_em.default_segmentation_loader(
            raw_paths=data_paths, raw_key=args.raw_key,
            label_paths=label_paths, label_key=args.label_key,
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            raw_transform=raw_transform,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            sampler=sampler, n_samples=args.n_samples,
            rois=[np.s_[:80, :, :], np.s_[:160, :, :]] if args.with_rois else None,
        )
        val_loader = torch_em.default_segmentation_loader(
            raw_paths=data_paths, raw_key=args.raw_key,
            label_paths=label_paths, label_key=args.label_key,
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            raw_transform=raw_transform,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            sampler=sampler, n_samples=args.n_samples,
            rois=[np.s_[80:, :, :], np.s_[160:, :, :]] if args.with_rois else None,
        )
        trainer = torch_em.default_segmentation_trainer(
            name=experiment_name, model=model,
            train_loader=train_loader, val_loader=val_loader,
            loss=loss_function, metric=metric_function,
            learning_rate=args.learning_rate,
            mixed_precision=True,
            log_image_interval=50,
            device=device,
            compile_model=False,
            save_root=SAVE_DIR,
            early_stopping=args.early_stopping,
        )
        if (False if torch.cuda.is_available() else True):
            check_loader(train_loader, n_samples=5)
            check_loader(val_loader, n_samples=5)
            check_trainer(trainer, n_samples=1)
        else:
            trainer.fit(n_iterations)


if __name__ == "__main__":
    main()
