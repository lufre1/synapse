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
import h5py
import torch_em
# import torch_em.data.datasets as torchem_data
from torch_em.data import MinInstanceSampler
from torch_em.model import AnisotropicUNet
from torch_em.util.debug import check_loader, check_trainer

# Import your util.py for data loading
import synapse.util as util
import synapse.training_util as tu
# import data_classes
# SAVE_DIR = "/scratch-grete/usr/nimlufre/synapse/mito_segmentation"
SAVE_DIR = "/mnt/lustre-grete/usr/u12103/cristae/"

# Pinned test split from training run 2026-06-01 (slurm job 14037451).
# These 15 files are held out as a fixed test set when --test_split synapsenetv1-testsplit is passed.
SYNAPSENETV1_TEST_SPLIT = [
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT22_eb5_model2_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT40_eb10_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/M5_eb1_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae/2026/36859_J1_66K_TS_PS_03_rec_2kb1dawbp_crop_combined.h5",
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae/2026/36194_B4_66K_TS_SC_22_rec_crop_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT21_eb5_model2_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/KO8_eb4_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/2026-05-26-dataset/2026-05-26_corrected_combined/Otof_AVCN07_455L_KO_M.Stim_B3_2_35933_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/M8_eb6_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT20_eb5_model2_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/M1_eb6_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/M2_eb5_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/Otof_AVCN03_429A_WT_M.Stim_D3_4model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae/2026/36194_B4_66K_TS_R01A_SC_01_rec_crop_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT21_syn4_model2_combined.h5",
]

NAMED_TEST_SPLITS = {
    "synapsenetv1-testsplit": SYNAPSENETV1_TEST_SPLIT,
}
# from unet import UNet3D


def explude_string(list_of_strings, string_to_exlude):
    return [s for s in list_of_strings if string_to_exlude not in s]


def log_dataset_stats(paths, split_name):
    unique_paths = list(dict.fromkeys(paths))  # deduplicate while preserving order
    print(f"\n--- Dataset statistics [{split_name}] ---  ({len(unique_paths)} files)")

    total_bg = total_mito_ann = total_mito_unann = total_cristae = 0

    for path in unique_paths:
        try:
            with h5py.File(path, "r") as f:
                state   = f["raw_mitos_combined"][1]   # mito-state channel
                cristae = f["labels/cristae"][()]

            n_bg        = int(np.sum(state == 0))
            n_mito_ann  = int(np.sum(state == 1))
            n_mito_unan = int(np.sum(state == 2))
            n_cristae   = int(np.sum(cristae > 0))
            n_total     = state.size

            total_bg        += n_bg
            total_mito_ann  += n_mito_ann
            total_mito_unann += n_mito_unan
            total_cristae   += n_cristae

            cristae_of_ann = 100 * n_cristae / n_mito_ann if n_mito_ann > 0 else float("nan")
            print(
                f"  {os.path.basename(path)}"
                f"  shape={cristae.shape}"
                f"  | mito_ann={n_mito_ann:>10,} ({100*n_mito_ann/n_total:4.1f}%)"
                f"  mito_unann={n_mito_unan:>10,} ({100*n_mito_unan/n_total:4.1f}%)"
                f"  | cristae={n_cristae:>8,} ({cristae_of_ann:4.1f}% of ann mito)"
            )
        except Exception as e:
            print(f"  WARNING: could not read {path}: {e}")

    grand_total = total_bg + total_mito_ann + total_mito_unann
    if grand_total > 0:
        cristae_of_ann = 100 * total_cristae / total_mito_ann if total_mito_ann > 0 else float("nan")
        print(
            f"  [TOTAL]"
            f"  bg={total_bg:>12,} ({100*total_bg/grand_total:4.1f}%)"
            f"  mito_ann={total_mito_ann:>10,} ({100*total_mito_ann/grand_total:4.1f}%)"
            f"  mito_unann={total_mito_unann:>10,} ({100*total_mito_unann/grand_total:4.1f}%)"
            f"  | cristae={total_cristae:>8,} ({cristae_of_ann:4.1f}% of ann mito)"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="3D UNet for mitochondrial segmentation")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to YAML config file")
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
    parser.add_argument("--ignore_state_value", type=int, default=2, help="During loss computation ignore this state value")
    parser.add_argument("--state_channel", type=int, default=None, help="Use this channel as state channel")
    parser.add_argument("--test_split", type=str, default=None, choices=list(NAMED_TEST_SPLITS.keys()),
                        help="Pin a named test split (e.g. synapsenetv1-testsplit) and exclude those files from training")
    parser.add_argument("--loss_variant", type=str, default="new", choices=["new", "legacy"],
                        help="Loss function to use during training. "
                             "'new' (default): MaskedDiceLoss — excludes unannotated mito (state=2) from Dice. "
                             "'legacy': MaskedDiceLossLegacy wrapper (numerically identical to 'new').")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for torch/numpy/python RNG (identical across A/B arms -> identical init).")
    parser.add_argument("--normalize", action="store_true", default=False,
                        help="Train with normalization instead of standardization.")
    parser.add_argument("--split_strategy", type=str, default="legacy", choices=["legacy", "grouped_stratified"],
                        help="Train/val split strategy. 'legacy': flat random shuffle + ensure_strings. "
                             "'grouped_stratified': group whole specimens (no sibling-crop leakage) and "
                             "stratify val across source x genotype (see synapse.cristae.splits).")
    parser.add_argument("--holdout_test_siblings", action="store_true", default=False,
                        help="Only with grouped_stratified: also drop sibling crops of the pinned-test "
                             "specimens from train/val (leakage-safe test; costly in data).")

    # Parse --config first, apply as defaults, then re-parse so CLI overrides YAML
    cfg_args, _ = parser.parse_known_args()
    if cfg_args.config is not None:
        with open(cfg_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        parser.set_defaults(**cfg)
    # re-parse the FULL argv so explicit CLI flags override the config defaults
    args = parser.parse_args()

    # Deterministic, identical initialization for the loss-isolation A/B: both arms
    # seed the RNGs the same way so the initial UNet weights are byte-identical.
    # (Residual nondeterminism from multi-worker loaders / cuDNN may remain.)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False

    checkpoint_path = args.checkpoint_path
    n_iterations = args.n_iterations
    learning_rate = args.learning_rate
    data_dir = args.data_dir
    # lucchi_data_dir = args.lucchi_data_dir/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae
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

    ndim = 3

    if args.loss_variant == "legacy":
        loss_function = util.MaskedDiceLossLegacy()
    else:  # "new"
        loss_function = util.MaskedDiceLoss()
    metric_function = util.MaskedDiceLossLegacy()
    print(f"[loss] loss_variant={args.loss_variant} -> "
          f"loss={type(loss_function).__name__}, metric={type(metric_function).__name__}")
    gain = 2
    in_channels, out_channels = 2, 2
    scale_factors = [
        [1, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2]
    ]

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
        exclude_strings = [
            "Otof_AVCN03_429C_WT_M.Stim_G3_1_model_combined",  # raw data strange
            "WT20_eb8_AZ1_model_combined",  # poor crisate annotations
            "WT22_eb8_model_combined",  # poor crisate annotations
        ]
        for s in exclude_strings:
            data_paths = explude_string(data_paths, s)
        print("len data paths", len(data_paths))

        if args.split_strategy == "grouped_stratified":
            from synapse.cristae.splits import grouped_stratified_split, summarize_split
            pinned_test = NAMED_TEST_SPLITS[args.test_split] if args.test_split is not None else None
            data = grouped_stratified_split(
                data_paths, val_ratio=0.1, seed=args.seed,
                pinned_test=pinned_test, holdout_test_siblings=args.holdout_test_siblings,
            )
            summarize_split(data, strict_test=args.holdout_test_siblings)
        else:
            pinned_test = None
            if args.test_split is not None:
                pinned_test = NAMED_TEST_SPLITS[args.test_split]
                pinned_test_set = set(pinned_test)
                data_paths = [p for p in data_paths if p not in pinned_test_set]
                print(f"Using pinned test split '{args.test_split}' ({len(pinned_test)} files); {len(data_paths)} remain for train/val")
            random.seed(42)
            random.shuffle(data_paths)
            ensure_strs = ["wichmann", "cooper"] if torch.cuda.is_available() else None
            if pinned_test is not None:
                data = util.split_data_paths_to_dict_with_ensure(
                    data_paths, train_ratio=0.9, val_ratio=0.1, test_ratio=0.0,
                    ensure_strings=ensure_strs
                    )
                data["test"] = pinned_test
            else:
                train_ratio = 0.8
                val_ratio = 0.1
                test_ratio = 0.1
                data = util.split_data_paths_to_dict_with_ensure(
                    data_paths, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio,
                    ensure_strings=ensure_strs
                    )
                print(f"Using dynamic train/val/test ratios: {train_ratio}/{val_ratio}/{test_ratio}")

    end_time = time.time()
    # Calculate execution time in seconds
    execution_time = end_time - start_time
    print(f"Data and ROI preprocessing execution time: {execution_time:.6f} seconds")

    print("Creating 3d UNet with", in_channels, "input channels and", out_channels, "output channels.")
    #UNet3d
    model = util.get_3d_model(
        in_channels=in_channels, out_channels=out_channels, initial_features=initial_features,
        final_activation=final_activation, gain=gain, scale_factors=scale_factors
    )
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = SAVE_DIR
    # load model from checkpoint if exists
    checkpoint_path = tu.resolve_checkpoint(save_dir, experiment_name, args.checkpoint_path)
    if checkpoint_path:
        # Load model weights directly into the freshly-built model. We avoid
        # torch_em.util.load_model here because it reconstructs the FULL trainer
        # (incl. the serialized loss); older cristae checkpoints stored a
        # DiceLoss(ignore_label=..., ignore_state_value=..., state_channel=...)
        # that the current torch_em DiceLoss rejects, which would crash a
        # warm-start. The trainer below builds a fresh optimizer regardless, so
        # only the model weights are needed here.
        ckpt_file = checkpoint_path
        if os.path.isdir(ckpt_file):
            ckpt_file = os.path.join(checkpoint_path, "best.pt")
            if not os.path.exists(ckpt_file):
                ckpt_file = os.path.join(checkpoint_path, "latest.pt")
        ck = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        # Rebuild the model to match the checkpoint architecture exactly (the repo's
        # get_3d_model has drifted from what these checkpoints were trained with —
        # e.g. norm layers), then load weights. This guarantees a clean warm-start.
        model = AnisotropicUNet(**ck["init"]["model_kwargs"])
        model.load_state_dict(ck["model_state"])
        print(f"Warm-started model from {ckpt_file} (arch rebuilt from checkpoint model_kwargs)")
        model.to(device)

    with_channels = True
    with_label_channels = False
    sampler = MinInstanceSampler(p_reject=0.95)
    mito_mask_transform = util.MitoStateMaskTransform(
        mito_channel=args.state_channel, exclude_state_value=float(args.ignore_state_value)
    )
    raw_transform = util.standardize_channel if not args.normalize else util.normalize_channel
    print("Path for this model", os.path.join(SAVE_DIR, experiment_name))
    print("train", len(data["train"]), "val", len(data["val"]), "test", len(data["test"]))
    print("data['train']", data["train"])
    print("data['val']", data["val"])
    print("data['test']", data["test"])
    print("Raw transform:", raw_transform)

    for split in ("train", "val", "test"):
        if data[split]:
            log_dataset_stats(data[split], split)

    if with_rois:
        train_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key="raw_mitos_combined",
            label_paths=data["train"], label_key="labels/cristae",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            rois=rois_dict["train"],
            transform=mito_mask_transform,
        )
        val_loader = torch_em.default_segmentation_loader(
            raw_paths=data["val"], raw_key="raw_mitos_combined",
            label_paths=data["val"], label_key="labels/cristae",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            rois=rois_dict["val"],
            transform=mito_mask_transform,
        )
    else:
        train_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key="raw_mitos_combined",
            label_paths=data["train"], label_key="labels/cristae",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            sampler=sampler,
            raw_transform=raw_transform,
            transform=mito_mask_transform,
        )
        val_loader = torch_em.default_segmentation_loader(
            raw_paths=data["val"], raw_key="raw_mitos_combined",
            label_paths=data["val"], label_key="labels/cristae",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            sampler=sampler,
            raw_transform=raw_transform,
            transform=mito_mask_transform,
        ) if (torch.cuda.is_available() and data["val"]) else None

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
        it = iter(trainer.train_loader)

        for i in range(100):
            image, label = next(it)  # don't recreate iter(...) each time

            # label has shape [B, 4, D, H, W]: [binary, boundary, mask_binary, mask_boundary]
            x = image[0].detach().cpu()   # [2, D, H, W]
            y = label[0].detach().cpu()   # [4, D, H, W]

            # dummy prediction: 2 output channels (binary + boundary)
            n_out = label.size(1) // 2
            pred = torch.rand(label.size(0), n_out, *label.shape[2:])
            p = pred[0].detach().cpu()

            loss_value = loss_function(pred, label)

            # the mask is the second half of the label channels
            valid_mask = y[n_out].to(torch.uint8)  # [D, H, W]

            viewer = napari.Viewer()
            viewer.add_image(x[0].numpy(), name=f"{i}/raw_em", contrast_limits=(x[0].min().item(), x[0].max().item()))
            viewer.add_labels(x[1].numpy().astype(int), name=f"{i}/mito_state")

            viewer.add_image(y[0].numpy(), name=f"{i}/target_cristae")
            viewer.add_image(y[1].numpy(), name=f"{i}/target_boundary")
            viewer.add_image(p[0].numpy(), name=f"{i}/pred_cristae")
            viewer.add_image(p[1].numpy(), name=f"{i}/pred_boundary")

            viewer.add_labels(valid_mask.numpy(), name=f"{i}/valid_mask (1=mito_state==1)")
            print(
                f"sample {i}: loss={loss_value.item():.4f}, "
                f"valid_frac={(valid_mask.float().mean().item()):.3f}"
            )
            viewer.grid.enabled = True
            napari.run()
    else:
        trainer.fit(n_iterations)


if __name__ == "__main__":
    main()
