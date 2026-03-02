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

# Import your util.py for data loading
import synapse.util as util
# import data_classes
# SAVE_DIR = "/scratch-grete/usr/nimlufre/synapse/mito_segmentation"
SAVE_DIR = "/mnt/lustre-grete/usr/u12103/cristae/"
# from unet import UNet3D


def main():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to the data directory")
    parser.add_argument("--data_dir2", type=str, default=None, help="Path to the second data directory")
    parser.add_argument("--data_dir3", type=str, default=None, help="Path to the third data directory")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(32, 256, 256), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--n_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint used to load model's state_dict")
    parser.add_argument("--experiment_name", type=str, default="default-cristae-net", help="Name that is used for the experiment and store the model's weights")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size to be used")
    parser.add_argument("--feature_size", type=int, default=32, help="Initial feature size of the 3D UNet")
    parser.add_argument("--with_rois", action="store_true", default=False, help="Train without Regions Of Interest (ROI)")
    parser.add_argument("--early_stopping", type=int, default=15, help="Number of epochs without improvement before stopping training")
    parser.add_argument("--save_dir", "-sd", default=None, help="Savedir to store logs and checkpoints to.")
    parser.add_argument("--ignore_label", type=int, default=None, help="Label to ignore during training")
    parser.add_argument("--ignore_state_value", type=int, default=None, help="During loss computation ignore this state value")
    parser.add_argument("--state_channel", type=int, default=None, help="Use this channel as state channel")

    # Parse arguments
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    n_iterations = args.n_iterations
    learning_rate = args.learning_rate
    data_dir = args.data_dir
    # lucchi_data_dir = args.lucchi_data_dir
    # visualize = args.visualize
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    patch_shape = args.patch_shape
    initial_features = args.feature_size
    with_rois = args.with_rois

    n_workers = 4 if torch.cuda.is_available() else 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n Experiment: {experiment_name}\n")
    print(f"Using {device} with {n_workers} workers.")
    label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)

    loss_name = "dice"
    metric_name = "dice"
    ndim = 3

    loss_function = util.get_loss_function(loss_name, **{
        "ignore_label": args.ignore_label,
        "ignore_state_value": args.ignore_state_value,
        "state_channel": args.state_channel
    })
    metric_function = util.get_loss_function(metric_name, **{
        "ignore_label": args.ignore_label,
        "ignore_state_value": args.ignore_state_value,
        "state_channel": args.state_channel
    })
    gain = 2
    in_channels, out_channels = 2, 2
    scale_factors = [
        [1, 2, 2],
        [2, 2, 2],
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
        data, rois_dict = util.split_data_paths_to_dict(data_paths, rois_dict, train_ratio=.8, val_ratio=0.2, test_ratio=0)
    else:
        data_paths = util.get_data_paths(data_dir)
        if args.data_dir2 is not None:
            data_paths.extend(util.get_data_paths(args.data_dir2))
        if args.data_dir3 is not None:
            data_paths.extend(util.get_data_paths(args.data_dir3))
        substring = "_combined.h5"
        data_paths = [s for s in data_paths if substring in s]
        print("len data paths", len(data_paths))
        random.seed(42)
        random.shuffle(data_paths)
        ensure_strs = ["wichmann", "cooper"] if torch.cuda.is_available() else None
        data = util.split_data_paths_to_dict_with_ensure(
            data_paths, train_ratio=.8, val_ratio=0.1, test_ratio=0.1,
            ensure_strings=ensure_strs
            )

    end_time = time.time()
    # Calculate execution time in seconds
    execution_time = end_time - start_time
    print(f"Data and ROI preprocessing execution time: {execution_time:.6f} seconds")

    print("Creating 3d UNet with", in_channels, "input channels and", out_channels, "output channels.")
    #UNet3d
    model = AnisotropicUNet(
        in_channels=in_channels, out_channels=out_channels, initial_features=initial_features,
        final_activation=final_activation, gain=gain, scale_factors=scale_factors
    )
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = SAVE_DIR
    # load model from checkpoint if exists
    if os.path.exists(os.path.join(save_dir, "checkpoints", experiment_name, "best.pt")):
        checkpoint_path = os.path.join(save_dir, "checkpoints", experiment_name)
        print("Checkpoint exists, loading model from checkpoint", checkpoint_path)
    elif args.checkpoint_path is not None:
        checkpoint_path = args.checkpoint_path
        print("Loading model from given checkpoint", checkpoint_path)
    else:
        checkpoint_path = None
    if checkpoint_path:
        model = torch_em.util.load_model(checkpoint=checkpoint_path, device=device)
        # state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))["model_state"]
        # model.load_state_dict(state_dict)
        
        model.to(device)

    with_channels = True
    with_label_channels = False
    sampler = MinInstanceSampler(p_reject=0.95)
    # raw2_transform = torch_em.transform.label.labels_to_binary
    print("Path for this model", os.path.join(SAVE_DIR, experiment_name))
    print("train", len(data["train"]), "val", len(data["val"]), "test", len(data["test"]))
    print("data['train']", data["train"])
    print("data['val']", data["val"])
    print("data['test']", data["test"])

    if with_rois:
        train_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key="raw_mitos_combined",
            label_paths=data["train"], label_key="labels/cristae",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            rois=rois_dict["train"]
        )
        val_loader = torch_em.default_segmentation_loader(
            raw_paths=data["val"], raw_key="raw_mitos_combined",
            label_paths=data["val"], label_key="labels/cristae",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            rois=rois_dict["val"]
        )
    else:
        train_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key="raw_mitos_combined",
            label_paths=data["train"], label_key="labels/cristae",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            sampler=sampler,
            raw_transform=util.standardize_channel
        )
        val_loader = torch_em.default_segmentation_loader(
            raw_paths=data["val"], raw_key="raw_mitos_combined",
            label_paths=data["val"], label_key="labels/cristae",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            sampler=sampler,
            raw_transform=util.standardize_channel
        )

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
        early_stopping=args.early_stopping
        # logger=None
    )
    if not torch.cuda.is_available():
        import napari
        print("CUDA is not available, debugging instead.")
        # check_loader(train_loader, n_samples=5)
        # check_trainer(trainer, n_samples=5)
        samples = []
        it = iter(trainer.train_loader)

        for i in range(100):
            image, label = next(it)  # don't recreate iter(...) each time

            # pick one sample from batch for visualization
            x = image[0].detach().cpu()   # [C_in,D,H,W]
            y = label[0].detach().cpu()   # [C_out,D,H,W]
            
            if 2 not in np.unique(x[1].numpy().astype(int)):
                print("No mito-state channel id=2 in sample", i)
                continue

            # create a sane dummy prediction in [0,1] with correct shape
            pred = torch.rand_like(label)  # [B,C_out,...]
            p = pred[0].detach().cpu()

            # compute loss (state should be the input that contains mito-state channel)
            loss_value = loss_function(pred, label, image)

            # reconstruct ignore/valid mask the same way DiceLoss does (for inspection)
            state_channel = loss_function.state_channel
            ignore_value = loss_function.ignore_state_value
            state_ch = image[:, state_channel:state_channel + 1]  # [B,1,...]
            ignore_mask = (state_ch == ignore_value)[0, 0].detach().cpu()  # [D,H,W] bool
            valid_mask = (~ignore_mask).to(torch.uint8)
            viewer = napari.Viewer()
            # visualize: input channels
            viewer.add_image(x[0].numpy(), name=f"{i}/raw_em", contrast_limits=(x[0].min().item(), x[0].max().item()))
            viewer.add_labels(x[1].numpy().astype(int), name=f"{i}/mito_state")

            # visualize: targets and preds (per output channel)
            viewer.add_image(y[0].numpy(), name=f"{i}/target_cristae")
            viewer.add_image(y[1].numpy(), name=f"{i}/target_boundary")
            viewer.add_image(p[0].numpy(), name=f"{i}/pred_cristae")
            viewer.add_image(p[1].numpy(), name=f"{i}/pred_boundary")

            # visualize mask + print sanity checks
            viewer.add_labels(valid_mask.numpy(), name=f"{i}/valid_mask (1=used)")
            print(
                f"sample {i}: loss={loss_value.item():.4f}, "
                f"valid_frac={(valid_mask.float().mean().item()):.3f}, "
                f"ignore_value={ignore_value}, state_channel={state_channel}"
            )

            # optional: compare to loss without masking to verify masking effect
            if hasattr(loss_function, "ignore_state_value"):
                # temporarily disable masking
                old = loss_function.ignore_state_value
                loss_function.ignore_state_value = None
                loss_nomask = loss_function(pred, label, image).item()
                loss_function.ignore_state_value = old
                print(f"           loss_no_mask={loss_nomask:.4f}")
                viewer.grid.enabled = True
            napari.run()
    else:
        trainer.fit(n_iterations)


if __name__ == "__main__":
    main()
