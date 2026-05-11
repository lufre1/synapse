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
from synapse_net.training.supervised_training import supervised_training, get_supervised_loader, get_3d_model

# Import your util.py for data loading
import synapse.util as util
import synapse.label_utils as lutil
import synapse.h5_util as h5_util

SAVE_DIR = "/mnt/lustre-grete/usr/u15205/volume-em/models/"


class SigmoidDiceLoss(torch.nn.Module):
    """DiceLoss that applies sigmoid to logits first (for use with BCEWithLogitsLoss)."""
    def __init__(self):
        super().__init__()
        self.dice = torch_em.loss.DiceLoss()
        self.init_kwargs = {}

    def forward(self, input_, target):
        return self.dice(torch.sigmoid(input_), target)


def main():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to the data directory")
    parser.add_argument("--data_dir2", type=str, default=None, help="Path to a second data directory")
    parser.add_argument("--data_dir3", type=str, default=None, help="Path to a third data directory")
    parser.add_argument("--raw_key", "-rk", type=str, default="raw")
    parser.add_argument("--label_key", "-lk", type=str, default="labels/axons")
    parser.add_argument("--visualize", action="store_true", default=False, help="Visualize data with napari")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(32, 256, 256), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--n_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples to be used for training per dataset")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to checkpoint used to load model's state_dict")
    parser.add_argument("--experiment_name", type=str, default="default-mito-net", help="Name that is used for the experiment and store the model's weights")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size to be used")
    parser.add_argument("--feature_size", type=int, default=32, help="Initial feature size of the 3D UNet")
    parser.add_argument("--early_stopping", type=int, default=10, help="Number of epochs without improvement before stopping training")
    parser.add_argument("--with_batchrenorm", action="store_true", default=False, help="Create UNet with batchrenorm. xor with instancenorm")
    parser.add_argument("--with_instancenorm", action="store_true", default=False, help="Create UNet with instancenorm. xor with batchrenorm")
    parser.add_argument("--use_synapsenet_training", "-ust", action="store_true", default=False, help="Use synapse net supervised training.")
    parser.add_argument("--save_dir", "-sd", default=None, help="directory to store checkpoits and logs to")

    # Parse arguments
    args = parser.parse_args()
    n_iterations = args.n_iterations
    data_dir = args.data_dir
    data_dir2 = args.data_dir2
    experiment_name = args.experiment_name
    batch_size = args.batch_size
    patch_shape = args.patch_shape
    save_dir = SAVE_DIR if args.save_dir is None else args.save_dir
    
    print(f"\n Experiment: {experiment_name}\n")

    label_transform = torch_em.transform.labels_to_binary

    if os.path.exists(os.path.join(save_dir, "checkpoints", experiment_name, "best.pt")):
        # torch_em default is to load "best.pt" (do not include it in path)
        checkpoint_path = os.path.join(save_dir, "checkpoints", experiment_name)
        print("Checkpoint exists, loading model from checkpoint", checkpoint_path)
    elif args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        print("Loading model from given checkpoint", checkpoint_path)
    else:
        checkpoint_path = None
    # if checkpoint_path:
    #     print("synapse-net supervised training has no checkpoint loading!")

    loss_name = "dice"
    in_channels, out_channels = 1, 1

    final_activation = None  # BCEWithLogitsLoss handles sigmoid internally

    # load data paths etc.
    start_time = time.time()
    print(f"Start time {time.ctime()}")

    data_paths = util.get_data_paths(data_dir)
    # data_paths = util.get_wichmann_data()
    print(data_paths)
    if data_dir2 is not None:
        data_paths2 = util.get_data_paths(data_dir2)
        data_paths.extend(data_paths2)
    if args.data_dir3 is not None:
        data_paths3 = util.get_data_paths(args.data_dir3)
        data_paths.extend(data_paths3)
    filtered = []
    for path in data_paths:
        keys = h5_util.get_all_keys_from_h5(path)
        if "labels/axons" in keys:
            filtered.append(path)
    data_paths = filtered
    random.seed(42)
    random.shuffle(data_paths)
    # data_paths.sort(reverse=True)
    # data = util.split_data_paths_to_dict(data_paths, rois_list=None, train_ratio=.75, val_ratio=0.25, test_ratio=0.0)
    data = util.split_data_paths_to_dict_with_ensure(
        data_paths=data_paths,
        ensure_strings=("4007", "4009")
    )

    end_time = time.time()
    # Calculate execution time in seconds
    execution_time = end_time - start_time
    print(f"Data and ROI preprocessing execution time: {execution_time:.6f} seconds")

    print("Creating 3d UNet with", in_channels, "input channels and", out_channels, "output channels.")

    sampler = MinInstanceSampler(p_reject=0.95)
    print("Path for this model", os.path.join(save_dir, "checkpoints", experiment_name))
    print("train", len(data["train"]), "val", len(data["val"]), "test", len(data["test"]))
    print("data['train']", data["train"])
    print("data['val']", data["val"])
    print("data['test']", data["test"])

    if args.use_synapsenet_training:
        print("Training with synapse-net supervised training")
        supervised_training(
            name=experiment_name,
            train_paths=data["train"],
            val_paths=data["val"],
            label_key=args.label_key,
            patch_shape=patch_shape,
            save_root=save_dir,
            batch_size=batch_size,
            n_iterations=n_iterations,
            sampler=sampler,
            out_channels=out_channels,
            label_transform=label_transform,
            raw_transform=torch_em.transform.raw.normalize_percentile,  # default is standardize
            checkpoint_path=checkpoint_path,
        )
    else:
        print("Training with torch_em trainer")
        # create model
        with_channels = False
        with_label_channels = False
        loss_name = "dice"
        metric_name = "dice"
        ndim = 3
        scale_factors = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        raw_transform = torch_em.transform.raw.normalize_percentile
        train_raw_transform = torch_em.transform.Compose(
            torch_em.transform.raw.normalize_percentile,
            torch_em.transform.raw.RandomContrast(alpha=(0.5, 2.0)),
            torch_em.transform.raw.AdditiveGaussianNoise(scale=(0.0, 0.1)),
            is_multi_tensor=False,
        )
        n_workers = 8
        device = "cuda" if torch.cuda.is_available() else "cpu"

        loss_function = torch_em.loss.CombinedLoss(
            SigmoidDiceLoss(),
            torch.nn.BCEWithLogitsLoss(),
        )
        metric_function = SigmoidDiceLoss()
        if args.with_batchrenorm:
            norm = "BatchRenorm"
        elif args.with_instancenorm:
            norm = "InstanceNorm"
        else:
            norm = None
        print(f"Using norm layer {norm} - where None is defaulting to InstanceNorm")

        model = util.get_3d_model(
            out_channels=out_channels,
            in_channels=in_channels,
            scale_factors=scale_factors,
            initial_features=args.feature_size,
            norm=norm,
            final_activation=final_activation
        )
        if checkpoint_path is not None:
            # model.load_state_dict(torch.load(checkpoint_path))
            model = torch_em.util.load_model(checkpoint=checkpoint_path, device=device)
            print("Successfully loaded model from checkpoint", checkpoint_path)

        train_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key=args.raw_key,
            label_paths=data["train"], label_key=args.label_key,
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            raw_transform=train_raw_transform,
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
            save_root=save_dir,
            early_stopping=args.early_stopping,
        )

        if not torch.cuda.is_available():
            import napari
            print("CUDA not available — running debug instead of training.")

            # --- initial stats over a few batches ---
            model.eval()
            it = iter(train_loader)
            n_debug = 3
            debug_raw, debug_label, debug_pred_init = None, None, None
            with torch.no_grad():
                for i in range(n_debug):
                    raw, label = next(it)
                    pred = model(raw)
                    loss_val = loss_function(pred, label)
                    # label may be [B,1,D,H,W] or [B,D,H,W] depending on loader
                    fg_frac = label.float().mean().item()
                    print(
                        f"  batch {i}: raw [{raw.min():.3f}, {raw.max():.3f}] mean={raw.mean():.3f} | "
                        f"label shape={tuple(label.shape)} unique={label.unique().tolist()} fg={fg_frac:.3f} | "
                        f"pred [{pred.min():.3f}, {pred.max():.3f}] mean={pred.mean():.3f} | "
                        f"loss={loss_val.item():.4f}"
                    )
                    if i == 0:
                        debug_raw, debug_label, debug_pred_init = raw, label, pred

            # --- simulate N gradient steps to watch for collapse ---
            n_steps = 30
            print(f"\nSimulating {n_steps} gradient steps to watch for collapse...")
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            it2 = iter(train_loader)
            for step in range(n_steps):
                raw_s, label_s = next(it2)
                optimizer.zero_grad()
                pred_s = model(raw_s)
                loss_s = loss_function(pred_s, label_s)
                loss_s.backward()
                grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                optimizer.step()
                print(
                    f"  step {step:3d}: loss={loss_s.item():.4f}  "
                    f"pred=[{pred_s.min():.3f}, {pred_s.max():.3f}] mean={pred_s.mean():.3f}  "
                    f"|grad|={grad_norm:.3f}"
                )

            # --- prediction on the same batch after training steps ---
            model.eval()
            with torch.no_grad():
                debug_pred_after = model(debug_raw)

            # --- unpack volumes (handle label being 4D or 5D) ---
            r  = debug_raw[0, 0].numpy()   # [D, H, W]
            p0 = debug_pred_init[0, 0].numpy()
            pN = debug_pred_after[0, 0].numpy()
            if debug_label.ndim == 5:
                lb = debug_label[0, 0].numpy().astype(int)
            else:
                lb = debug_label[0].numpy().astype(int)

            print(f"\nBefore {n_steps} steps: pred mean={p0.mean():.4f}  After: pred mean={pN.mean():.4f}")

            viewer = napari.Viewer()
            viewer.add_image(r,       name="raw",                    colormap="gray")
            viewer.add_labels(lb,     name="label")
            viewer.add_image(p0,      name=f"pred_init (mean={p0.mean():.3f})",             colormap="bop orange", opacity=0.7)
            viewer.add_image(pN,      name=f"pred_after_{n_steps}_steps (mean={pN.mean():.3f})", colormap="bop blue",   opacity=0.7)
            viewer.add_image(p0 - pN, name="delta init−after",       colormap="RdBu",       opacity=0.7)
            viewer.grid.enabled = True
            napari.run()
        else:
            trainer.fit(n_iterations)


if __name__ == "__main__":
    main()
