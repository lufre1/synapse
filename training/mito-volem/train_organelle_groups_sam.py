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
    # [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 37, 52, 53, 54, 65],  # nucleus with pores and envelope
    # [20, 21, 22, 23, 65],                                      # nuclear envelope
    [6, 7, 40],                                            # golgi
    [8, 9, 41],                                            # vesicle
    [10, 11, 42],                                          # endosome
    [12, 13, 43],                                          # lysosome
    [14, 15, 44],                                          # lipid droplet
    [16, 17, 18, 19, 46, 51, 64],                          # endoplasmic reticulum with exit sites
    [47, 48, 49],                                          # peroxisome
    [3, 4, 5, 50],                                         # mitochondria
    [30, 36, 55],                                          # microtubule
    [38],                                                   # vimentin
    [39],                                                   # glycogen
    [61],                                                   # actin
    [62],                                                   # t-bar
    [56, 57, 58, 60],                                   # cell
    [31, 32, 33, 66],                                      # centrosome collective
    [34],                                                  # ribosomes
    # [63],                                                  # basement membrane
    # [2, 35],                                                  # cytosol
    # [0, 1, 2],                                             # extracellular space + plasma membrane
    [45],                                                  # red blood cells
]

OUT_IDS = list(range(1, len(ID_GROUPS) + 1))  # Assigned class numbers in the output

# import multiprocessing as mp
# cell_ids = [38, 39, 56, 57, 58, 61, 62, 60]
# cell_ids_set = set(cell_ids)


# def process_file(path):
#     data = io.load_data_from_file(path)
#     if "all" in data.keys():
#         uniq = np.unique(data["all"])
#         if cell_ids_set.intersection(uniq):
#             return path


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
    parser.add_argument("--label_key", type=str, default="label_crop/all", help="Label key to be used for training e.g. label_crop/all")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples to be used for training per dataset")
    parser.add_argument("--min_size", type=int, default=10, help="Minimal pixel size for organelles 2D")
    parser.add_argument("--model_type", type=str, default="vit_b", help="Model type to be used")

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
    # load model from checkpoint if exists
    if os.path.exists(os.path.join(SAVE_DIR, "checkpoints", experiment_name, "best.pt")):
        checkpoint_path = os.path.join(SAVE_DIR, "checkpoints", experiment_name, "best.pt")
        print("Checkpoint exists, loading model from checkpoint", checkpoint_path)
    elif args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
        print("Loading model from given checkpoint", checkpoint_path)
    else:
        checkpoint_path = None

    print(f"\n Experiment: {experiment_name}\n")

    label_transform = lutil.LabelAggregatorSAM(
        id_groups=ID_GROUPS,
        out_ids=OUT_IDS,
        # group_transforms=mito_transform if mito_transform is not None else None,
    )

    # load data paths etc.
    start_time = time.time()
    print(f"Start time {time.ctime()}")

    # data_paths = cutil.get_resized_cellmap_paths(organelle_size="medium")
    data_paths = util.get_data_paths(data_dir)

    # print("Filter paths for ID_GROUPS to keep...")
    # data_paths = cutil.get_paths_with_any_id_group(data_paths, ID_GROUPS=ID_GROUPS, min_pct_slices=0, n_workers=4)


# Return path or other identifier for files with matches
    print("Before filtering data_paths", len(data_paths))
    data_paths = cutil.get_cellmaps_paths_fully_annotated(data_paths)
    print("After filtering data_paths for fully annotated", len(data_paths))
    data_paths = cutil.filter_paths_for_only_foreground_parallel(data_paths, dataset="label_crop/all", n_workers=8)
    print("After filtering data_paths for foreground and not only one label", len(data_paths))

    # specified_ids = [31, 32, 33, 66]
    # data_paths = cutil.get_paths_with_any_id_group(data_paths, ID_GROUPS=[specified_ids], min_pct_slices=0, n_workers=4)

    # with mp.Pool(8) as pool:
    #     results = pool.map(process_file, data_paths)

    # # Filter out None results where no match was found
    # matched_paths = [result for result in results if result]
    # print("Files with matches:", matched_paths)
    # for p in tqdm(data_paths):
    #     print(p)
    #     data = io.load_data_from_file(p)
    #     print(data.keys())
    #     print("labels np uniq", np.unique(data[args.label_key]))
    #     v = napari.Viewer()
    #     v.add_image(data[args.raw_key])
    #     v.add_labels(data[args.label_key])
    #     # Filter labels to only include specified ids
    #     label_data = data[args.label_key]
    #     filtered_labels = np.isin(label_data, specified_ids) * label_data

    #     # Add the filtered labels to the viewer
    #     v.add_labels(filtered_labels, name="Filtered Labels")

    #     napari.run()
    # return

    # cell_ids = [38, 39, 56, 57, 58, 61, 62, 60]
    # for path in data_paths:
    #     data = io.load_data_from_file(path)
    #     print(data.keys())
    #     if "all" in data.keys():
    #         uniq = np.unique(data["all"])
    #         print("np uniq", uniq)
    #         if any(cell_id in uniq for cell_id in cell_ids):
    #             print("\n FOUND ONE!\n")
    #             v = napari.Viewer()
    #             v.add_image(data["raw_crop"])
    #             v.add_labels(data["all"])
    #             napari.run()

    # data_paths = cutil.get_cellmaps_paths_fully_annotated()
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
        id_groups=ID_GROUPS, min_num_groups=1, p_reject=1, min_size=args.min_size
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
    # custom_label_transform = torch_em.transform.generic.Compose(label_transform, default_label_transform, is_multi_tensor=False)
    # for i in range(0, 5):
    #     with open_file("/mnt/lustre-grete/usr/u12103/cellmap/resized_crops/crop_172.h5") as f:
    #         raw = f["raw"][:]
    #         labels = f["label_crop/all"][:]
    #         v = napari.Viewer()
    #         v.add_image(raw)
    #         v.add_labels(labels, name="labels")
    #         transformed = custom_label_transform(labels)
    #         v.add_image(transformed, name="custom")
    #         my_transfromed = label_transform(labels)
    #         # v.add_image(default_transfromed, name="transformed distance")
    #         v.add_image(my_transfromed, name="just my label transform")
    #         napari.run()
    
    # from micro_sam.training import train_sam_for_configuration, default_sam_loader
    # from micro_sam.training.training import _check_loader

    # roi_train, roi_val = None, None
    
    # data_paths = "/mnt/lustre-emmy-ssd/projects/nim00007/data/cellmap/data_crops/crop_172.h5"
    # data_paths = [path for path in data_paths if "172.h5" in path]
    # print("data paths left", data_paths)

    # all_loader = default_sam_loader(
    #     raw_paths=data_paths, raw_key=args.raw_key,
    #     label_paths=data_paths, label_key=args.label_key,
    #     patch_shape=patch_shape, with_segmentation_decoder=True, with_channels=False,
    #     batch_size=batch_size, rois=roi_train, raw_transform=None,
    #     label_transform=label_transform, min_size=args.min_size,
    #     sampler=sampler, n_samples=args.n_samples * 10
    # )
    # _check_loader(all_loader, with_segmentation_decoder=True, verify_n_labels_in_loader=args.n_samples * 10)
    # return
    
    # for i in tqdm(range(0, 10000)):
    #     x, y = next(iter(all_loader))
        # print("i", i)
        # print("x and y shapes:", x.shape, y.shape)
        # uniq = np.unique(y[0, 0, :, :])
        # if len(uniq) == 1:
        #     print("np uniq y[0]", np.unique(uniq))
        #     return
    
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
        raw_key=args.raw_key,
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
        # raw_transform=raw_transform, # added in sam_util.py
        n_samples=args.n_samples,
        min_size=args.min_size,
        model_type=args.model_type,
        check=(False if torch.cuda.is_available() else True),
    )


if __name__ == "__main__":
    main()
