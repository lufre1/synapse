import pprint
import torch
import numpy as np
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
from synapse_net.training.supervised_training import supervised_training, get_supervised_loader, get_3d_model

# Import your util.py for data loading
import synapse.util as util
import synapse.cellmap_util as cutil
import synapse.label_utils as lutil
import synapse.sam_util as sutil
# import data_classes
SAVE_DIR = "/scratch-grete/usr/nimlufre/cellmap/"
# ids from https://janelia.figshare.com/articles/online_resource/CellMap_Segmentation_Challenge/28034561?file=51215543 
# page 9
# ID_GROUPS = [
#     [3, 4, 5, 50],             # mitochondria
#     # [6, 7, 40],                # golgi
#     # [14, 15, 44],              # liquid droplets
#     # [
#     #     16, 17, 18, 19,
#     #     46, 51, 64
#     # ],                         # endo reticulum
#     # [16,17,51,64],              # ER
#     # [18,19,46],                  # ER exit sites (eres)
#     # [47, 48, 49]               # peroxisomes
# ]
# ID_GROUPS = [
#     [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],  # nucleus 20:29
#     [24, 25, 26, 27, 54]  # chromatin also available [24, 25, 26, 27]
# ]
# ID_GROUPS = [
#     [8,9,41],                   # vesicles
#     [10,11,42],                 # endosomes
#     # [12,13,43],                 # lysosomes
#     # [47,48,49],                 # peroxysomes
#     # [39],                       # glycogen
#     # [62],                       # t-bar
# ]
# all organelles
ID_GROUPS = [
    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 37, 52, 53, 65],  # nucleus with pores and envelope
    [6, 7, 40],                                            # golgi
    [8, 9, 41],                                            # vesicle
    [10, 11, 42],                                          # endosome
    [12, 13, 43],                                          # lysosome
    [14, 15, 44],                                          # lipid droplet
    [16, 17, 18, 19, 46, 51, 64],                          # endoplasmic reticulum with exit sites
    [47, 48, 49],                                          # peroxisome
    [3, 4, 5, 50],                                         # mitochondria
    [24, 25, 26, 27, 54],                                  # chromatin
    [30, 36, 55],                                          # microtubule
    [38, 39, 56, 57, 58, 61, 62, 60],                      # cell
    [31, 32, 33, 66],                                      # centrosome collective
    [34],                                                  # ribosomes
    [35],                                                  # cytosol
    # [0, 1, 2],                                             # extracellular space + plasma membrane
    [45],                                                  # red blood cells
]

OUT_IDS = list(range(1, len(ID_GROUPS) + 1))  # Assigned class numbers in the output


def main():
    parser = argparse.ArgumentParser(description="3D UNet for medium organelle segmentation")
    parser.add_argument("--data_dir", type=str, default="/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/",  # "/scratch-grete/projects/nim00007/data/cellmap/resized_crops/",
                        help="Path to the data directory")
    parser.add_argument("--data_dir2", type=str, default=None, help="Path to a second data directory")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(1, 256, 256), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--n_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint used to load model's state_dict")
    parser.add_argument("--experiment_name", type=str, default="cellmap-organelles", help="Name that is used for the experiment and store the model's weights")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to be used")
    parser.add_argument("--early_stopping", type=int, default=10, help="Number of epochs without improvement before stopping training")
    parser.add_argument("--label_key", type=str, default="label_crop/all", help="Label key to be used for training e.g. label_crop/all")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to be used for training per dataset")

    # Parse arguments
    args = parser.parse_args()
    n_iterations = args.n_iterations
    data_dir = args.data_dir
    data_dir2 = args.data_dir2
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    patch_shape = args.patch_shape

    if torch.cuda.is_available():
        os.makedirs("/scratch-grete/usr/nimlufre/cellmap/", exist_ok=True)
    #load model from checkpoint if exists
    if os.path.exists(os.path.join(SAVE_DIR, "checkpoints", experiment_name, "best.pt")):
        checkpoint_path = os.path.join(SAVE_DIR, "checkpoints", experiment_name, "best.pt")
        print("Checkpoint exists, loading model from checkpoint", checkpoint_path)
    elif args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
        print("Loading model from given checkpoint", checkpoint_path)
    else:
        checkpoint_path = None
    # n_workers = 12 if torch.cuda.is_available() else 1
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Experiment: {experiment_name}\n")
    # we do not need boundary + foreground for microsam (we use distances)
    # if [3, 4, 5, 50] in ID_GROUPS:
    #     mito_transform = {1: lutil.CombinedLabelTransform(add_binary_target=True, dilation_footprint=np.ones((3, 3)))}
    # else:
    #     mito_transform = None
    mito_transform = None

    label_transform = lutil.LabelAggregatorSAM(
        id_groups=ID_GROUPS,
        out_ids=OUT_IDS,
        #group_transforms=mito_transform if mito_transform is not None else None,
    )

    in_channels, out_channels = 1, len(ID_GROUPS)

    # load data paths etc.
    start_time = time.time()
    print(f"Start time {time.ctime()}")

    # data_paths = cutil.get_resized_cellmap_paths(organelle_size="medium")
    data_paths = util.get_data_paths(data_dir)
    # data_paths = cutil.get_cellmap_paths_without_cell_and_nuclei()
    # remove paths 
    exclude_strings = [
        "_243.h5", "_25.h5", "_26.h5", "_55.h5",
        "_56.h5", "_57.h5", "_58.h5", "_59.h5", "_60.h5", "_61.h5",
        "_63.h5", "_64.h5", "_65.h5", "_66.h5", "_67.h5", "_68.h5",
        "_69.h5", "_70.h5", "_71.h5", "_72.h5", "_73.h5", "_74.h5",
        "_75.h5", "_77.h5", "_81.h5", "_83.h5", "_84.h5", "_85.h5",
        "_86.h5", "_87.h5", "_88.h5", "_90.h5", "_91.h5", "_92.h5",
        "_93.h5", "_94.h5", "_95.h5", "_96.h5", "_97.h5", "_98.h5",
        "_99.h5",
        ]
    data_paths = [p for p in data_paths if not any(s in p for s in exclude_strings)]


    # print("Filter paths for ID_GROUPS to keep...")
    # data_paths = cutil.get_paths_with_any_id_group(data_paths, ID_GROUPS=ID_GROUPS, min_pct_slices=0, n_workers=4)
    # print("Calculate statistics for filtered files...")
    # stats = cutil.parallel_group_stats_in_h5(data_paths, ID_GROUPS, n_workers=None)
    # pretty_stats = dict(stats)  # Convert nested defaultdicts to dicts if needed
    # print("ID_GROUPS", ID_GROUPS)
    # pprint.pprint(pretty_stats)

    print(data_paths)

    random.seed(42)
    random.shuffle(data_paths)
    data = util.split_data_paths_to_dict(data_paths, rois_list=None, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05)

    end_time = time.time()
    # Calculate execution time in seconds
    execution_time = end_time - start_time
    print(f"Data preprocessing execution time: {execution_time:.6f} seconds")

    # print("Creating 3d UNet with", in_channels, "input channels and", out_channels, "output channels.")

    sampler = cutil.AtLeastNGroupsSampler(
        id_groups=ID_GROUPS, min_num_instances=1, min_num_groups=2, p_reject=1, min_size=100
        )

    # semantic_ids: List[int], min_fraction: float, min_fraction_per_id: bool = False, p_reject: float = 1.0
    # sampler = torch_em.data.sampler.MinSemanticLabelForegroundSampler() 

    print("train", len(data["train"]), "val", len(data["val"]), "test", len(data["test"]))
    print("data['test']", data["test"])

    # import napari
    # from elf.io import open_file
    # default_label_transform = torch_em.transform.label.PerObjectDistanceTransform(
    #         distances=True,
    #         boundary_distances=True,
    #         directed_distances=False,
    #         foreground=True,
    #         instances=True,
    #         min_size=25,
    #     )
    # custom_label_transform = label_transform
    # label_transform = torch_em.transform.generic.Compose(label_transform, default_label_transform, is_multi_tensor=False)
    # for i in range(0, 5):
    #     with open_file(data["train"][i]) as f:
    #         raw = f["raw"][:]
    #         labels = f["label_crop/all"][:]
    #         v = napari.Viewer()
    #         v.add_image(raw)
    #         v.add_labels(labels, name="labels")
    #         transformed = custom_label_transform(labels)
    #         v.add_image(transformed, name="my transformed")
    #         default_transfromed = default_label_transform(labels)
    #         # v.add_image(default_transfromed, name="transformed distance")
    #         v.add_image(label_transform(labels), name="combined")
    #         napari.run()
    
    from micro_sam.training import train_sam_for_configuration, default_sam_loader

    roi_train, roi_val = None, None

    # train_loader = default_sam_loader(
    #     raw_paths=data["train"], raw_key="raw",
    #     label_paths=data["train"], label_key=args.label_key,
    #     patch_shape=patch_shape, with_segmentation_decoder=True, with_channels=False,
    #     batch_size=batch_size, rois=roi_train, raw_transform=None,
    #     label_transform=label_transform,
    #     sampler=sampler, n_samples=args.n_samples
    # )
    # check_loader(train_loader, n_samples=args.n_samples)
    # return
    # for i in tqdm(range(0, 10000)):
    #     x, y = next(iter(train_loader))
    #     # print("i", i)
    #     # print("x and y shapes:", x.shape, y.shape)
    #     uniq = np.unique(y[0, 0, :, :])
    #     if len(uniq) == 1:
    #         print("np uniq y[0]", np.unique(uniq))
    #         return
    
    # val_loader = default_sam_loader(
    #     raw_paths=data["val"], raw_key="raw",
    #     label_paths=data["val"], label_key=args.label_key,
    #     patch_shape=patch_shape, with_segmentation_decoder=True, with_channels=False,
    #     batch_size=batch_size, rois=roi_train, raw_transform=None,
    #     label_transform=label_transform,
    #     sampler=sampler
    # )
    # stop_at = 100
    # count_zeros = 0
    # with torch.no_grad():
    #     for i, (x, y) in enumerate(tqdm(val_loader, start=1, desc="Processing files")):
    #         # If 'y' is a torch.Tensor, y[0, 0].max() returns a scalar tensor:
    #         # .item() converts it to a Python number.
    #         uniq = np.unique(y[0, 0, :, :])
    #         if len(uniq) == 1:
    #             print("Found patch with uniqs", uniq)
    #         # Stop once we've processed the desired number of batches
    #         if i >= stop_at:
    #             break

    #     # print(f"Train: Saw {count_zeros} batches where y[0, 0, :, :] was all zeros out of {i} total checked.")
    #     for i in tqdm(range(0, stop_at)):
    #         x, y = next(iter(train_loader))
    #         # print("i", i)
    #         # print("x and y shapes:", x.shape, y.shape)
    #         uniq = np.array(y[0, 0, :, :]).max()  # np.unique(y[0, 0, :, :])
    #         if uniq == 0:
    #             print("train np uniq y[0]", uniq)
    #     count_zeros = 0
    #     for i in tqdm(range(0, stop_at)):
    #         x, y = next(iter(val_loader))
    #         # print("i", i)
    #         # print("x and y shapes:", x.shape, y.shape)
    #         uniq = np.array(y[0, 0, :, :]).max()  # np.unique(y[0, 0, :, :])
    #         if uniq == 0:
    #             print("val np uniq y[0]", uniq)
        
    #     # for i, (x, y) in enumerate(tqdm(val_loader), start=1):
    #     #     # If 'y' is a torch.Tensor, y[0, 0].max() returns a scalar tensor:
    #     #     # .item() converts it to a Python number.
    #     #     if not torch.any(y[0, 0]):
    #     #         count_zeros += 1
    #     #     # Stop once we've processed the desired number of batches
    #     #     if i >= stop_at:
    #     #         break

    #     print(f"Val: Saw {count_zeros} batches where y[0, 0, :, :] was all zeros out of {i} total checked.")
    # return

    sutil.finetune_sam_v2(
        name=experiment_name,
        train_images=data["train"],
        raw_key="raw",
        val_images=data["val"],
        label_key=args.label_key,
        patch_shape=patch_shape,
        save_root=SAVE_DIR,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        n_iterations=n_iterations,
        sampler=sampler,
        early_stopping=args.early_stopping,
        # out_channels=out_channels,
        label_transform=label_transform,
        # raw_transform=raw_transform,
        n_samples=args.n_samples,
        check=(False if torch.cuda.is_available() else True),
    )


if __name__ == "__main__":
    main()
