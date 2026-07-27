"""Threshold-free AP evaluation for cristae, integrated into the eval pipeline.

Reads, for each segmentation-output H5 in `export_path` (produced by segment_cristae.py with
`save_predictions: true`): the foreground-probability map (`pred/foreground`), the mito-state channel
(`labels/mitochondria`, values 0/1/2) and the cristae GT (`labels/cristae`). Within the evaluated
region (mito state == 1) it computes — reusing `analyze()` from `diagnose_cristae_probs.py` — the
average precision (AP), ROC-AUC, mean foreground prob on true/non cristae, and a threshold sweep.

With `--band_nm` (default 8 and 12 nm) the same metrics are additionally computed **restricted to the
mitochondrial-membrane shell** — the cristae-junction region — and to its complement:

    region = "all"      : every state==1 voxel (unchanged; comparable to previous runs)
    region = "band<t>"  : state==1 voxels within t nm of the membrane   -> junction accuracy
    region = "core<t>"  : the remaining state==1 voxels                 -> bulk accuracy

Junction voxels are a median ~2% of crista voxels (see RESULTS_cristae_junction_audit.md), so they
are invisible in the "all" AP; the band numbers are what a membrane-targeted training change has to
move. The restriction is applied by handing `analyze()` a state array that is 1 only inside the
region, so the metric definition is bit-for-bit the same in every region.

This reuses the predictions already computed by the segment step (no model re-run). Writes
`cristae_ap_summary.csv` (+ `cristae_ap_sweep.csv`) into `export_path`, next to `cristae_eval_results.csv`.

Usage (config-driven, like segment/evaluate):
    python evaluation/cristae/evaluate_cristae_ap.py -c <eval_config.yaml>
    python evaluation/cristae/evaluate_cristae_ap.py -e <export_dir> --band_nm 8 12
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

from synapse.cristae.membrane import DEFAULT_VOXEL_SIZE, membrane_band, membrane_distance  # noqa: E402
from synapse.h5_util import read_voxel_size  # noqa: E402


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
    parser.add_argument("--band_nm", type=float, nargs="+", default=[8.0, 12.0],
                        help="Membrane-shell thicknesses in nm. For each, AP is also computed inside the "
                             "shell (region 'band<t>', the cristae-junction region) and outside it "
                             "(region 'core<t>'). Set `band_nm: []` in the config for the whole-region "
                             "metric only.")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=None,
                        help="(z y x) voxel size in nm, overriding the file's 'voxel_size' attribute. "
                             f"Falls back to {DEFAULT_VOXEL_SIZE} when neither is available.")
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
    if getattr(args, "band_nm", None) is None:
        args.band_nm = []
    return args


def _fname(path):
    return os.path.splitext(os.path.basename(path))[0]


def _voxel_size(path, override):
    """(z, y, x) nm voxel size: explicit override > file attribute > corpus default."""
    if override is not None:
        return tuple(float(v) for v in override), "config"
    try:
        vs = read_voxel_size(path, h5_key="raw", default=None)
    except Exception:
        vs = None
    if vs is not None:
        return tuple(float(v) for v in vs), "file-attr"
    return DEFAULT_VOXEL_SIZE, "default"


def _regions(state, band_nm, voxel_size):
    """[(region_name, state_array)] — the whole annotated-mito region plus membrane band / core.

    The band/core arrays are 0/1 state arrays, so `analyze()` restricts to exactly that region
    without any change to how AP is computed. State==2 statistics are only meaningful for the
    whole region and come out NaN for the band/core rows.
    """
    regions = [("all", state)]
    if not len(band_nm):
        return regions
    s1 = state == 1
    dist = membrane_distance(s1, voxel_size)
    for t in band_nm:
        band = membrane_band(s1, voxel_size, t, distance=dist)
        regions.append((f"band{t:g}", band.astype(np.uint8)))
        regions.append((f"core{t:g}", (s1 & ~band).astype(np.uint8)))
    return regions


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

        vs, vs_source = _voxel_size(f, args.voxel_size)
        line = [f"  {_fname(f)} [vs={vs[0]:g},{vs[1]:g},{vs[2]:g} nm ({vs_source})]"]
        for region, region_state in _regions(state, args.band_nm, vs):
            summary, sweep = analyze(fg, region_state, gt)
            tags = {"model": model, "file": _fname(f), "region": region}
            summary.update(tags)
            summary_rows.append(summary)
            for s in sweep:
                s.update(tags)
                sweep_rows.append(s)
            line.append(f"{region} AP={summary['ap']:.4f}")
        print("  ".join(line), flush=True)

    df = pd.DataFrame(summary_rows)
    # One averaged row per region (macro average over files).
    avgs = []
    for region in df["region"].unique():
        sub = df[df["region"] == region]
        avg = {c: (sub[c].mean() if np.issubdtype(sub[c].dtype, np.number) else "") for c in df.columns}
        avg.update({"file": "all-files-averaged", "model": model, "region": region})
        avgs.append(avg)
    df = pd.concat([df, pd.DataFrame(avgs)], ignore_index=True)

    os.makedirs(args.output_path, exist_ok=True)
    out_summary = os.path.join(args.output_path, "cristae_ap_summary.csv")
    df.to_csv(out_summary, index=False)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(args.output_path, "cristae_ap_sweep.csv"), index=False)

    print(f"\n[AP] {model} (n={len(files)} files)")
    for _, macro in df[df["file"] == "all-files-averaged"].iterrows():
        print(f"  region={macro['region']:<10} macro AP={macro['ap']:.4f}  AUC={macro['auc']:.4f}  "
              f"gt_fg={int(macro['gt_fg']):,}/{int(macro['n_state1']):,} voxels", flush=True)
    print(f"  ->  {out_summary}", flush=True)


if __name__ == "__main__":
    main()
