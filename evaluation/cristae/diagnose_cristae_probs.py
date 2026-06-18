"""Diagnose the cristae 06-01 vs 06-04 over-prediction.

For each (model, file) this mirrors the EXACT inference of
`synapse.cristae.segment.run_cristae_segmentation` to obtain the foreground
probability map, then — *within the evaluated region (mito state == 1)* —
computes:

  * threshold-free separability:  average precision (AP) and ROC-AUC of the
    foreground probability vs the cristae GT  -> answers "genuine degradation?"
  * a foreground-threshold sweep (voxel-level precision/recall/Dice)
    -> answers "does a higher threshold recover precision? (calibration)"
  * the operating point at the hard-coded inference threshold 0.5 (sanity-check
    vs the published eval CSVs)

It also reports, for context, how much each model predicts inside *unannotated*
mito (state == 2), which the standard eval masks out.

The inference threshold lives in the separate synapse-net package (hard-coded
`> 0.5`); we never modify it — we threshold the returned probabilities here.

Usage:
  python diagnose_cristae_probs.py \
    --model NAME:CKPT_DIR:best [--model NAME2:CKPT_DIR2:latest ...] \
    --out_dir <dir> [--subset] [--tile 32 256 256] [--cache_dir <dir>]
"""
import argparse
import os

import numpy as np
import pandas as pd
from elf.io import open_file
import torch
import torch_em.util
from sklearn.metrics import average_precision_score, roc_auc_score

from synapse_net.inference.cristae import segment_cristae as _segment_cristae
from torch_em.model.unet import AnisotropicUNet  # noqa: F401 (used via model_kwargs)

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_POINTS_AUC = 3_000_000  # subsample cap for AP/AUC to keep it fast

# 15 test-split files (SYNAPSENETV1_TEST_SPLIT)
ALL_FILES = [
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT22_eb5_model2_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT40_eb10_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/M5_eb1_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae/2026/36859_J1_66K_TS_PS_03_rec_2kb1dawbp_crop_combined.h5",
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae/2026/36194_B4_66K_TS_SC_22_rec_crop_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT21_eb5_model2_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/KO8_eb4_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann_needs_corrections/2026-05-26-dataset/Otof_AVCN07_455L_KO_M.Stim_B3_2_35933_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/M8_eb6_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT20_eb5_model2_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/M1_eb6_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/M2_eb5_model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/Otof_AVCN03_429A_WT_M.Stim_D3_4model_combined.h5",
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae/2026/36194_B4_66K_TS_R01A_SC_01_rec_crop_combined.h5",
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT21_syn4_model2_combined.h5",
]
# Representative subset for a fast first pass: worst collapse, a stable one, two mid.
SUBSET = [
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann_needs_corrections/2026-05-26-dataset/Otof_AVCN07_455L_KO_M.Stim_B3_2_35933_combined.h5",  # 0.737->0.494
    "/mnt/lustre-grete/usr/u12103/mitochondria/cooper/cristae/2026/36194_B4_66K_TS_R01A_SC_01_rec_crop_combined.h5",  # ~0.913 (stable)
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/M8_eb6_model_combined.h5",  # 0.599->0.542
    "/mnt/lustre-grete/usr/u12103/cristae_data/wichmann/inital_data/WT21_eb5_model2_combined.h5",  # 0.773->0.713
]


def fname(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_model_weights(ckpt_dir, name, device):
    """Load just the model weights, bypassing trainer/loss reconstruction.

    `torch_em.util.load_model` rebuilds the full trainer (incl. the loss), which
    fails for the 06-01 checkpoint (its serialized `DiceLoss(ignore_label=...)`
    is incompatible with the current torch_em). We instead rebuild the model from
    the stored `model_kwargs` (identical across both models) and load `model_state`.
    """
    path = ckpt_dir if ckpt_dir.endswith(".pt") else os.path.join(ckpt_dir, f"{name}.pt")
    ck = torch.load(path, map_location=device, weights_only=False)
    mk = dict(ck["init"]["model_kwargs"])
    model = AnisotropicUNet(**mk)
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    print(f"    loaded {path} (epoch={ck.get('epoch')}, iteration={ck.get('iteration')}, "
          f"best_metric={ck.get('best_metric')})", flush=True)
    return model


def load_file(path):
    with open_file(path, "r") as f:
        img = f["raw_mitos_combined"][:]
        gt = f["labels/cristae"][:]
    return img[0], img[1], gt  # raw, state, gt


def predict_fg(raw, state, model, tile, device):
    z, y, x = tile
    tiling = {"tile": {"z": z, "y": y, "x": x},
              "halo": {"z": int(z * 0.25), "y": int(y * 0.25), "x": int(x * 0.25)}}
    mito_proc = (state > 0).astype(np.uint8)
    # mirror run_cristae_segmentation exactly
    seg, pred = _segment_cristae(
        raw, voxel_size=1.44, model=model, scale=None, tiling=tiling, return_predictions=True,
        extra_segmentation=mito_proc, channels_to_standardize=[0], with_channels=True,
        verbose=False,
    )
    return pred[0].astype(np.float32)  # foreground probability (independent of voxel_size/erosion)


def _subsample(gt_bool, fg, cap):
    n = gt_bool.size
    if n <= cap:
        return gt_bool, fg
    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=cap, replace=False)
    return gt_bool[idx], fg[idx]


def analyze(fg, state, gt):
    s1 = state == 1
    s2 = state == 2
    gt1 = (gt[s1] > 0)
    fg1 = fg[s1]

    summary = {
        "n_state1": int(s1.sum()),
        "gt_fg": int(gt1.sum()),
        "n_state2": int(s2.sum()),
        "fg_mean_gtpos_state1": float(fg1[gt1].mean()) if gt1.any() else np.nan,
        "fg_mean_gtneg_state1": float(fg1[~gt1].mean()) if (~gt1).any() else np.nan,
        "frac_fg_gt0p5_state2": float((fg[s2] > 0.5).mean()) if s2.any() else np.nan,
        "mean_fg_state2": float(fg[s2].mean()) if s2.any() else np.nan,
    }
    if gt1.any() and (~gt1).any():
        gsub, fsub = _subsample(gt1, fg1, MAX_POINTS_AUC)
        summary["ap"] = float(average_precision_score(gsub, fsub))
        summary["auc"] = float(roc_auc_score(gsub, fsub))
    else:
        summary["ap"] = np.nan
        summary["auc"] = np.nan

    pos = int(gt1.sum())
    sweep = []
    for thr in THRESHOLDS:
        pred1 = fg1 > thr
        tp = int((pred1 & gt1).sum())
        fp = int((pred1 & ~gt1).sum())
        fn = pos - tp
        eps = 1e-8
        sweep.append({
            "threshold": thr,
            "precision": tp / (tp + fp + eps),
            "recall": tp / (tp + fn + eps),
            "dice": 2 * tp / (2 * tp + fp + fn + eps),
            "tp": tp, "fp": fp, "fn": fn,
        })
    return summary, sweep


def parse_models(specs):
    models = []
    for s in specs:
        parts = s.split(":")
        if len(parts) == 2:
            tag, ckpt = parts; name = "best"
        else:
            tag, ckpt, name = parts[0], parts[1], parts[2]
        models.append((tag, ckpt, name))
    return models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="TAG:CKPT_DIR[:best|latest]; repeatable")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--subset", action="store_true", help="Use the 4-file representative subset")
    ap.add_argument("--tile", type=int, nargs=3, default=[32, 256, 256])
    ap.add_argument("--cache_dir", default=None, help="Cache foreground probs (float16 .npy) for fast re-sweeps")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    files = SUBSET if args.subset else ALL_FILES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = parse_models(args.model)

    summary_rows, sweep_rows = [], []
    for tag, ckpt, name in models:
        print(f"\n=== model {tag} ({name}) from {ckpt} ===", flush=True)
        model = load_model_weights(ckpt, name, device)
        for path in files:
            fn = fname(path)
            cache = os.path.join(args.cache_dir, f"{tag}__{fn}.npy") if args.cache_dir else None
            raw, state, gt = load_file(path)
            if cache and os.path.exists(cache):
                fg = np.load(cache).astype(np.float32)
                print(f"  [{tag}] {fn}: loaded cached fg", flush=True)
            else:
                print(f"  [{tag}] {fn}: predicting (shape {raw.shape}) ...", flush=True)
                fg = predict_fg(raw, state, model, tuple(args.tile), device)
                if cache:
                    np.save(cache, fg.astype(np.float16))
            summary, sweep = analyze(fg, state, gt)
            summary.update({"model": tag, "ckpt_name": name, "file": fn})
            summary_rows.append(summary)
            for sw in sweep:
                sw.update({"model": tag, "ckpt_name": name, "file": fn})
                sweep_rows.append(sw)
            print(f"    AP={summary['ap']:.4f} AUC={summary['auc']:.4f} "
                  f"fg(gt+)={summary['fg_mean_gtpos_state1']:.3f} fg(gt-)={summary['fg_mean_gtneg_state1']:.3f} "
                  f"state2:frac>0.5={summary['frac_fg_gt0p5_state2']:.3f}", flush=True)

    summ = pd.DataFrame(summary_rows)
    sweep = pd.DataFrame(sweep_rows)
    summ.to_csv(os.path.join(args.out_dir, "diag_summary.csv"), index=False)
    sweep.to_csv(os.path.join(args.out_dir, "diag_sweep.csv"), index=False)

    # macro means per model for a quick read
    print("\n==== macro means per model (over files) ====")
    cols = ["ap", "auc", "fg_mean_gtpos_state1", "fg_mean_gtneg_state1", "frac_fg_gt0p5_state2"]
    print(summ.groupby("model")[cols].mean().to_string())
    print(f"\nWrote: {os.path.join(args.out_dir, 'diag_summary.csv')} and diag_sweep.csv")


if __name__ == "__main__":
    main()
