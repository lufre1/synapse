import pprint
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

# Import your util.py for data loading
import synapse.util as util
import synapse.cellmap_util as cutil
import synapse.label_utils as lutil
# import data_classes
SAVE_DIR = "/scratch-grete/usr/nimlufre/synapse/mito_segmentation"
# ids from https://janelia.figshare.com/articles/online_resource/CellMap_Segmentation_Challenge/28034561?file=51215543 
# page 9
ID_GROUPS = [
    # [3, 4, 5, 50],             # mitochondria
    [6, 7, 40],                # golgi
    [14, 15, 44],              # liquid droplets
    # [
    #     16, 17, 18, 19,
    #     46, 51, 64
    # ],                         # endo reticulum
    [47, 48, 49]               # peroxysomes
    # Add more groups as desired
]
OUT_IDS = list(range(1, len(ID_GROUPS) + 1))  # Assigned class numbers in the output


def main():
    parser = argparse.ArgumentParser(description="3D UNet for medium organelle segmentation")
    parser.add_argument("--data_dir", type=str, default="/scratch-grete/projects/nim00007/data/cellmap/resized_crops/",
                        help="Path to the data directory")
    parser.add_argument("--data_dir2", type=str, default=None, help="Path to a second data directory")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(128, 128, 128), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--n_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint used to load model's state_dict")
    parser.add_argument("--experiment_name", type=str, default="cellmap-medium-organelles", help="Name that is used for the experiment and store the model's weights")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size to be used")
    parser.add_argument("--feature_size", type=int, default=32, help="Initial feature size of the 3D UNet")
    parser.add_argument("--early_stopping", type=int, default=10, help="Number of epochs without improvement before stopping training")

    # Parse arguments
    args = parser.parse_args()
    n_iterations = args.n_iterations
    data_dir = args.data_dir
    data_dir2 = args.data_dir2
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    patch_shape = args.patch_shape

    n_workers = 12 if torch.cuda.is_available() else 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Experiment: {experiment_name}\n")
    print(f"Using {device} with {n_workers} workers.")
    mito_transform = lutil.CombinedLabelTransform(add_binary_target=True, dilation_footprint=np.ones((3, 3)))

    label_transform = lutil.LabelAggregator(
        id_groups=ID_GROUPS,
        out_ids=OUT_IDS,
        # group_transforms={1: mito_transform}
    )

    raw_transform = torch_em.transform.raw.normalize_percentile  # util.custom_raw_transform

    in_channels, out_channels = 1, len(ID_GROUPS)

    # load data paths etc.
    start_time = time.time()
    print(f"Start time {time.ctime()}")

    data_paths = cutil.get_resized_cellmap_paths(organelle_size="medium")
    data_paths = cutil.get_paths_with_any_id_group(data_paths, ID_GROUPS=ID_GROUPS)
    stats = cutil.parallel_group_stats_in_h5(data_paths, ID_GROUPS, n_workers=128)
    pretty_stats = dict(stats)  # Convert nested defaultdicts to dicts if needed
    pprint.pprint(pretty_stats)
    return
    # data_paths = util.get_data_paths(data_dir)
    data_paths = cutil.get_paths_with_any_id_group(data_paths, ID_GROUPS=ID_GROUPS)

    print(data_paths)
    if data_dir2 is not None:
        data_paths2 = util.get_data_paths(data_dir2)
        data_paths.extend(data_paths2)

    for path in data_paths:
        if "combined" in path:
            data_paths.remove(path)
            print("Found path with multiple channels as raw and removed:", path)
    random.seed(42)
    random.shuffle(data_paths)
    # data_paths.sort(reverse=True)
    data = util.split_data_paths_to_dict(data_paths, rois_list=None, train_ratio=.8, val_ratio=0.15, test_ratio=0.05)

    end_time = time.time()
    # Calculate execution time in seconds
    execution_time = end_time - start_time
    print(f"Data preprocessing execution time: {execution_time:.6f} seconds")

    print("Creating 3d UNet with", in_channels, "input channels and", out_channels, "output channels.")

    focus_groups = ID_GROUPS
    # [
    #     ID_GROUPS[0]
    # ]
    # focus_groups = [
    #     [3, 4, 5, 50],             # mitochondria
    #     # [6, 7, 40],                # golgi
    #     # [14, 15, 44],              # liquid droplets
    #     [
    #         16, 17, 18, 19,
    #         46, 51, 64
    #     ],                         # endo reticulum
    #     # [47, 48, 49]               # peroxysomes
    # ]
    # sampler = MinInstanceSampler(min_num_instances=20, p_reject=0.95)
    sampler = cutil.AtLeastNGroupsSampler(id_groups=focus_groups, min_num_instances=1, p_reject=0.95)
    # prevalence = np.array([141, 18, 24, 185, 29])
    # weights = 1 / prevalence         # Rarer groups get bigger weight
    # weights = weights / weights.sum()  # Normalize to sum to 1
    # print("Weights:", weights)
    # sampler = cutil.WeightedGroupSampler(ID_GROUPS, weights)

    print("train", len(data["train"]), "val", len(data["val"]), "test", len(data["test"]))
    print("data['test']", data["test"])

    supervised_training(
        name=experiment_name,
        train_paths=data["train"],
        raw_key="raw",
        val_paths=data["val"],
        label_key="label_crop/all",
        patch_shape=patch_shape,
        save_root=SAVE_DIR,
        batch_size=batch_size,
        n_iterations=n_iterations,
        sampler=sampler,
        out_channels=out_channels,
        label_transform=label_transform,
        raw_transform=raw_transform,
        # check=True,
    )


if __name__ == "__main__":
    main()
