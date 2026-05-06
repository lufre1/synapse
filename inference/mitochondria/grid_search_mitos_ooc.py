"""
Grid search over watershed post-processing parameters for OOC mitochondria segmentation.

Prediction is computed once per input file and cached on disk as a zarr.
For each parameter combination, only the final segmentation is kept on disk —
intermediates (distance transform, seeds) are deleted after each run.
"""
import argparse
import itertools
import os
import shutil
import time

import torch_em.transform
import yaml
import zarr
from elf.io import open_file
from synapse_net.inference.util import get_prediction
import synapse.io.util as io
import synapse.util as util


DEFAULT_SEED_DISTANCES     = [1, 2, 3]
DEFAULT_BG_PENALTIES       = [1.0, 1.5, 2.5]
DEFAULT_FOREGROUND_THRES   = [0.5, 0.6, 0.7]
DEFAULT_BOUNDARY_THRES     = [0.05, 0.08, 0.12]


def _param_tag(sd, bp, ft, bt):
    def fmt(v):
        return str(v).replace(".", "")
    return f"sd{fmt(sd)}_bp{fmt(bp)}_ft{fmt(ft)}_bt{fmt(bt)}"


def _ensure_prediction(path, key, tile_shape, model_path, preprocess_volem):
    """Return the on-disk zarr prediction array, computing it if not already present."""
    pred_name = os.path.basename(path) + "_pred.zarr"
    pred_path = os.path.join(os.path.dirname(path), pred_name)

    ts = {"z": tile_shape[0], "y": tile_shape[1], "x": tile_shape[2]}
    halo = {k: int(ts[k] * 0.125) for k in ts}
    tiling = {"tile": ts, "halo": halo}
    inner_ts = {k: ts[k] - 2 * halo[k] for k in ts}

    with open_file(path, "r") as f:
        spatial_shape = f[key].shape

    n_out = 2
    expected_shape = (n_out,) + tuple(spatial_shape)
    chunks = (n_out, inner_ts["z"], inner_ts["y"], inner_ts["x"])

    root = zarr.open(pred_path, mode="a")
    pred = root.get("pred")
    pred_ready = pred is not None and pred.shape == expected_shape
    print(f"Prediction already on disk: {pred_ready}  ({pred_path})")

    if not pred_ready:
        pred = root.require_dataset(
            "pred",
            shape=expected_shape,
            chunks=chunks,
            dtype="float32",
            compressor=zarr.Blosc(cname="zstd", clevel=3, shuffle=2),
            overwrite=True,
        )
        with open_file(path, "r") as f:
            image = f[key][...]
        if preprocess_volem:
            image = util.convert_white_patches_to_black(image)
        get_prediction(
            input_volume=image,
            model_path=model_path,
            tiling=tiling,
            preprocess=torch_em.transform.raw.normalize_percentile,
            prediction=pred,
        )
        print(f"Prediction written to {pred_path}")

    return pred, pred_path


def _is_done(out_dir):
    """True if out_dir exists with only the final seg (no intermediates left)."""
    if not os.path.exists(out_dir):
        return False
    root = zarr.open(out_dir, mode="r")
    return "seg" in root and "dist" not in root


def _delete_intermediates(out_dir):
    """Remove dist and seeds from the zarr store, keeping only seg."""
    root = zarr.open(out_dir, mode="a")
    for key in ("dist", "seeds"):
        if key in root:
            del root[key]
        key_dir = os.path.join(out_dir, key)
        if os.path.isdir(key_dir):
            shutil.rmtree(key_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Grid search over OOC mitochondria segmentation post-processing parameters."
    )
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="YAML config (same format as segment_mitos_*.yaml)")
    parser.add_argument("--seed_distance",        "-sd", nargs="+", type=int,
                        default=DEFAULT_SEED_DISTANCES,
                        help="seed_distance values to sweep (default: %(default)s)")
    parser.add_argument("--bg_penalty",           "-bp", nargs="+", type=float,
                        default=DEFAULT_BG_PENALTIES,
                        help="bg_penalty values to sweep (default: %(default)s)")
    parser.add_argument("--foreground_threshold", "-ft", nargs="+", type=float,
                        default=DEFAULT_FOREGROUND_THRES,
                        help="foreground_threshold values to sweep (default: %(default)s)")
    parser.add_argument("--boundary_threshold",   "-bt", nargs="+", type=float,
                        default=DEFAULT_BOUNDARY_THRES,
                        help="boundary_threshold values to sweep (default: %(default)s)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    base_path        = cfg["base_path"]
    file_ext         = cfg.get("file_extension", ".zarr")
    key              = cfg["key"]
    model_path       = cfg["model_path"]
    export_path      = cfg["export_path"]
    tile_shape       = cfg.get("tile_shape", [32, 512, 512])
    min_size         = cfg.get("min_size", 1000)
    area_threshold   = cfg.get("area_threshold", 200)
    n_threads        = cfg.get("n_threads", 8)
    preprocess_volem = cfg.get("preprocess_volem", False)

    os.makedirs(export_path, exist_ok=True)
    paths = io.load_file_paths(base_path, file_ext)
    print(f"Found {len(paths)} input file(s)")

    grid = list(itertools.product(
        args.seed_distance,
        args.bg_penalty,
        args.foreground_threshold,
        args.boundary_threshold,
    ))
    print(f"Grid: {len(grid)} combinations  "
          f"(sd={args.seed_distance}, bp={args.bg_penalty}, "
          f"ft={args.foreground_threshold}, bt={args.boundary_threshold})")

    for path in paths:
        print(f"\n=== {path} ===")
        pred, pred_path = _ensure_prediction(
            path, key, tile_shape, model_path, preprocess_volem
        )

        n_done = 0
        for sd, bp, ft, bt in grid:
            tag     = _param_tag(sd, bp, ft, bt)
            out_dir = os.path.join(export_path, tag + ".zarr")

            if _is_done(out_dir):
                print(f"  [{tag}] already done — skipping")
                n_done += 1
                continue

            print(f"\n  [{tag}]  sd={sd}  bp={bp}  ft={ft}  bt={bt}")
            t0 = time.time()

            util.segment_mitos_ooc_wrapped(
                pred=pred,
                foreground_threshold=ft,
                boundary_threshold=bt,
                seed_distance=sd,
                min_size=min_size,
                area_threshold=area_threshold,
                out_dir=out_dir,
                reuse_computed=False,
                bg_penalty=bp,
                n_threads=n_threads,
            )

            # Keep only the final segmentation — remove intermediates
            _delete_intermediates(out_dir)

            elapsed = time.time() - t0
            print(f"  [{tag}] done in {elapsed:.0f}s  →  {out_dir}")
            n_done += 1

        print(f"\nCompleted {n_done}/{len(grid)} combinations for {path}")


if __name__ == "__main__":
    main()
