"""Threshold-free AP evaluation for cristae, integrated into the eval pipeline.

Reads, for each segmentation-output H5 in `export_path` (produced by segment_cristae.py with
`save_predictions: true`): the foreground-probability map (`pred/foreground`), the mito-state channel
(`labels/mitochondria`, values 0/1/2) and the cristae GT (`labels/cristae`). Within the evaluated
region (mito state == 1) it computes — reusing `analyze()` from `diagnose_cristae_probs.py` — the
average precision (AP), ROC-AUC, mean foreground prob on true/non cristae, and a threshold sweep.

This reuses the predictions already computed by the segment step (no model re-run). Writes
`cristae_ap_summary.csv` (+ `cristae_ap_sweep.csv`) into `export_path`, next to `cristae_eval_results.csv`.

Usage (config-driven, like segment/evaluate):
    python evaluation/cristae/evaluate_cristae_ap.py -c <eval_config.yaml>
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml
from elf.io import open_file

# Reuse the exact analysis used by the standalone diagnostic (AP/AUC + fg-prob stats + threshold sweep).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_cristae_probs import analyze  # noqa: E402


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to YAML config file. CLI flags override config values.")
    parser.add_argument("-e", "--export_path", default=None,
                        help="Directory with the segmentation-output H5s (also the default output dir).")
    parser.add_argument("-o", "--output_path", default=None,
                        help="Directory for the AP CSVs (defaults to --export_path).")
    parser.add_argument("-k", "--key", default="labels/cristae",
                        help="H5 dataset key for the cristae GT.")
    parser.add_argument("--state_key", default="labels/mitochondria",
                        help="H5 dataset key for the mito-state channel (0=bg, 1=annotated, 2=unannotated).")
    parser.add_argument("--predictions_key", default="pred/foreground",
                        help="H5 dataset key for the foreground-probability map (needs save_predictions=true).")
    return parser


def parse_args():
    parser = build_parser()
    cfg_args, _ = parser.parse_known_args()
    if cfg_args.config is not None:
        with open(cfg_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        parser.set_defaults(**cfg)
    args = parser.parse_args()
    if args.output_path is None:
        args.output_path = args.export_path
    if args.export_path is None:
        parser.error("provide --export_path (or a config with export_path)")
    # config may not define these → fall back to the argparse defaults
    if getattr(args, "key", None) is None:
        args.key = "labels/cristae"
    return args


def _fname(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    args = parse_args()
    state_key = getattr(args, "state_key", "labels/mitochondria")
    pred_key = getattr(args, "predictions_key", "pred/foreground")
    model = os.path.basename(os.path.normpath(args.export_path))

    files = sorted(glob.glob(os.path.join(args.export_path, "*.h5")))
    if not files:
        raise SystemExit(f"No .h5 segmentation outputs found in {args.export_path}")

    summary_rows, sweep_rows = [], []
    for f in files:
        with open_file(f, "r") as h:
            if pred_key not in h:
                raise SystemExit(
                    f"{f}: missing '{pred_key}'. Re-run segment_cristae with `save_predictions: true` "
                    f"so the foreground probability is saved for the AP evaluation."
                )
            fg = h[pred_key][:].astype(np.float32)
            state = h[state_key][:]
            gt = h[args.key][:]
        summary, sweep = analyze(fg, state, gt)
        summary.update({"model": model, "file": _fname(f)})
        summary_rows.append(summary)
        for s in sweep:
            s.update({"model": model, "file": _fname(f)})
            sweep_rows.append(s)
        print(f"  {_fname(f)}: AP={summary['ap']:.4f} AUC={summary['auc']:.4f}", flush=True)

    df = pd.DataFrame(summary_rows)
    avg = {c: (df[c].mean() if np.issubdtype(df[c].dtype, np.number) else "") for c in df.columns}
    avg["file"] = "all-files-averaged"
    avg["model"] = model
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)

    os.makedirs(args.output_path, exist_ok=True)
    out_summary = os.path.join(args.output_path, "cristae_ap_summary.csv")
    df.to_csv(out_summary, index=False)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(args.output_path, "cristae_ap_sweep.csv"), index=False)

    macro = df[df["file"] == "all-files-averaged"].iloc[0]
    print(f"\n[AP] {model}: macro AP={macro['ap']:.4f}  AUC={macro['auc']:.4f}  "
          f"(n={len(files)} files)  ->  {out_summary}", flush=True)


if __name__ == "__main__":
    main()
