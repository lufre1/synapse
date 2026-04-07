import pprint
import torch
import numpy as np
import os
import random
import argparse
import time
import napari
import torch_em
# import torch_em.data.datasets as torchem_data
from torch_em.data import MinInstanceSampler
from torch_em.model import AnisotropicUNet
from torch_em.util.debug import check_loader, check_trainer
from tqdm import tqdm
from synapse_net.training.supervised_training import supervised_training, get_supervised_loader, get_3d_model

# Import your util.py for data loading
import synapse.util as util
import synapse.io.util as io
import synapse.cellmap_util as cutil
import synapse.label_utils as lutil
import synapse.sam_util as sutil
import synapse.h5_util as h5_util


def main():
    parser = argparse.ArgumentParser(description="3D UNet for medium organelle segmentation")
    parser.add_argument("--data_dir", type=str, default="/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/",  # "/scratch-grete/projects/nim00007/data/cellmap/resized_crops/",
                        help="Path to the data directory")
    parser.add_argument("--data_dir2", type=str, default=None, help="Path to a second data directory")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(1, 256, 256), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--n_iterations", type=int, default=15000, help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint used to load model's state_dict")
    parser.add_argument("--experiment_name", "-ep", type=str, default="cellmap-organelles", help="Name that is used for the experiment and store the model's weights")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to be used")
    parser.add_argument("--early_stopping", type=int, default=5, help="Number of epochs without improvement before stopping training")
    parser.add_argument("--raw_key", type=str, default="raw", help="Raw key to be used for training e.g. raw_crop")
    parser.add_argument("--label_key", type=str, default="labels/axons", help="Label key to be used for training e.g. label_crop/all")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples to be used for training per dataset")
    parser.add_argument("--min_size", type=int, default=10, help="Minimal pixel size for organelles 2D")
    parser.add_argument("--model_type", type=str, default="vit_b", help="Model type to be used")
    parser.add_argument("--save_dir", type=str, default="/mnt/lustre-grete/usr/u15205/volume-em/microsam")

    # Parse arguments
    args = parser.parse_args()
    n_iterations = args.n_iterations
    data_dir = args.data_dir
    data_dir2 = args.data_dir2
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    patch_shape = args.patch_shape
    save_dir = args.save_dir

    # load model from checkpoint if exists
    if os.path.exists(os.path.join(save_dir, "checkpoints", experiment_name, "best.pt")):
        checkpoint_path = os.path.join(save_dir, "checkpoints", experiment_name, "best.pt")
        print("Checkpoint exists, loading model from checkpoint", checkpoint_path)
    elif args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
        print("Loading model from given checkpoint", checkpoint_path)
    else:
        checkpoint_path = None

    print(f"\n Experiment: {experiment_name}\n")

    label_transform = None

    # load data paths etc.
    start_time = time.time()
    print(f"Start time {time.ctime()}")

    data_paths = util.get_data_paths(data_dir)
    if data_dir2 is not None:
        data_paths2 = util.get_data_paths(data_dir2)
        data_paths.extend(data_paths2)
    
    filtered = []
    for path in data_paths:
        keys = h5_util.get_all_keys_from_h5(path)
        if "labels/axons" in keys:
            filtered.append(path)
    data_paths = filtered

    print(data_paths)

    random.seed(42)
    random.shuffle(data_paths)
    data = util.split_data_paths_to_dict(data_paths, rois_list=None, train_ratio=0.9, val_ratio=0.1, test_ratio=0)

    end_time = time.time()
    # Calculate execution time in seconds
    execution_time = end_time - start_time
    print(f"Data preprocessing execution time: {execution_time:.6f} seconds")

    # print("Creating 3d UNet with", in_channels, "input channels and", out_channels, "output channels.")

    sampler = MinInstanceSampler()

    # semantic_ids: List[int], min_fraction: float, min_fraction_per_id: bool = False, p_reject: float = 1.0
    # sampler = torch_em.data.sampler.MinSemanticLabelForegroundSampler() 

    print("train", len(data["train"]), "val", len(data["val"]), "test", len(data["test"]))
    print("data['test']", data["test"])
    print("Export path for model is", os.path.join(save_dir, "checkpoints", experiment_name))


    sutil.finetune_sam_v2(
        name=experiment_name,
        train_images=data["train"],
        raw_key=args.raw_key,
        val_images=data["val"],
        label_key=args.label_key,
        patch_shape=patch_shape,
        save_root=save_dir,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        n_iterations=n_iterations,
        sampler=sampler,
        early_stopping=args.early_stopping,
        # out_channels=out_channels,
        label_transform=label_transform,
        # raw_transform=raw_transform, # added in sam_util.py
        n_samples=args.n_samples,
        min_size=args.min_size,
        model_type=args.model_type,
        check=(False if torch.cuda.is_available() else True),
    )


if __name__ == "__main__":
    main()
