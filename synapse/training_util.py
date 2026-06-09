"""Shared helpers for the training scripts under ``training/``.

Centralises boilerplate that was duplicated across the trainers:

* :func:`resolve_checkpoint` — the "load best.pt if present, else the given checkpoint,
  else None" idiom (used by ~7 trainers),
* :func:`count_instances_in_files` / :func:`log_dataset_stats` — dataset split logging,
* :func:`raw_transform_fix_white_patches` — normalize after zeroing white EM borders,
* :func:`filter_paths_by_h5_key` — keep only files that contain a given HDF5 key,
* :func:`foreground_fraction_sampler` — pseudo-label foreground sampler for the
  mean-teacher domain-adaptation trainers.

Argparse construction is intentionally left in the individual scripts, since their
defaults (batch size, patch shape, experiment name, ...) deliberately differ.
"""
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch_em

import synapse.util as util


def resolve_checkpoint(
    save_dir: str,
    experiment_name: str,
    cli_checkpoint: Optional[str] = None,
    verbose: bool = True,
) -> Optional[str]:
    """Resolve which checkpoint a trainer should resume from.

    Priority (highest first):
    1. ``cli_checkpoint`` if explicitly given — always honoured, even if a
       ``best.pt`` already exists for this experiment.  This lets you start a
       new A/B arm from a specific seed checkpoint without being silently
       redirected to a stale run.
    2. An existing ``<save_dir>/checkpoints/<experiment_name>/best.pt`` — the
       normal resume-in-place path when no explicit checkpoint is given.
    3. ``None`` — train from scratch.

    Returns the directory ``<save_dir>/checkpoints/<experiment_name>`` when
    resuming in-place (torch-em loads ``best.pt`` from there), or the raw
    ``cli_checkpoint`` path otherwise.
    """
    if cli_checkpoint:
        if verbose:
            print("Loading model from given checkpoint", cli_checkpoint)
        return cli_checkpoint
    best = os.path.join(save_dir, "checkpoints", experiment_name, "best.pt")
    if os.path.exists(best):
        checkpoint_path = os.path.join(save_dir, "checkpoints", experiment_name)
        if verbose:
            print("Checkpoint exists, resuming from", checkpoint_path)
        return checkpoint_path
    return None


def count_instances_in_files(file_paths: Sequence[str], label_key: str) -> int:
    """Count the total number of non-zero label ids across ``file_paths`` (HDF5)."""
    import h5py

    total = 0
    for path in file_paths:
        try:
            with h5py.File(path, "r") as f:
                labels = f[label_key][:]
                unique = np.unique(labels)
                total += int(np.sum(unique > 0))
        except Exception as e:
            print(f"  Warning: could not count instances in {os.path.basename(path)}: {e}")
    return total


def log_dataset_stats(data: Dict[str, List[str]], label_key: Optional[str] = None) -> None:
    """Print the train/val/test split sizes and (optionally) instance counts."""
    n_train, n_val, n_test = len(data["train"]), len(data["val"]), len(data["test"])
    print("\n=== Dataset split ===")
    print(f"  Train: {n_train} files | Val: {n_val} files | Test: {n_test} files | Total: {n_train + n_val + n_test}")
    if label_key is not None:
        train_n = count_instances_in_files(data["train"], label_key)
        val_n = count_instances_in_files(data["val"], label_key)
        print(f"  Train instances: {train_n} | Val instances: {val_n} | Total: {train_n + val_n}")
    print("=====================\n")


def raw_transform_fix_white_patches(x):
    """Zero large white (255) EM regions, then percentile-normalize."""
    x = util.convert_white_patches_to_black(x)
    return torch_em.transform.raw.normalize_percentile(x)


def filter_paths_by_h5_key(paths: Sequence[str], required_key: str) -> List[str]:
    """Keep only the HDF5 ``paths`` that contain ``required_key`` as a dataset key."""
    import synapse.h5_util as h5_util

    return [p for p in paths if required_key in h5_util.get_all_keys_from_h5(p)]


def foreground_fraction_sampler(pseudo_labels, label_filter, threshold=0.75, min_fraction=0.1, p=0.95):
    """Pseudo-label sampler for mean-teacher domain adaptation.

    Accept a patch if the foreground fraction (within ``label_filter``) exceeds
    ``min_fraction``; otherwise keep it only with probability ``1 - p``.
    """
    import random

    sampled_labels = pseudo_labels[label_filter.to(torch.bool)]
    foreground_prediction = (sampled_labels > threshold).sum()
    foreground_fraction = foreground_prediction / label_filter.sum()
    if foreground_fraction > min_fraction:
        return True
    return random.random() > p
