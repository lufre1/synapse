import argparse

import yaml

from synapse.cristae.segment import run_cristae_segmentation
from synapse.io.util import get_file_paths


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to YAML config file. CLI flags override config values.")
    parser.add_argument("--base_path", "-b", type=str, default=None,
                        help="Root data directory; H5 files are discovered recursively. "
                             "Ignored when --h5_paths is given.")
    parser.add_argument("--h5_paths", "-p", type=str, nargs="+", default=None,
                        help="Explicit list of H5 files to segment (overrides --base_path globbing). "
                             "Usually supplied via the config file as `h5_paths:`.")
    parser.add_argument("--export_path", "-e", type=str, default=None,
                        help="Root output directory")
    parser.add_argument("--model_path", "-m", type=str, default=None,
                        help="Path to cristae model checkpoint")
    parser.add_argument("--add_missing", "-am", default=False, action="store_true",
                        help="Merge model-found objects back into existing GT labels")
    parser.add_argument("--tile_shape", "-ts", type=int, nargs=3, default=[32, 512, 512],
                        help="Tile shape (z y x)")
    parser.add_argument("--erode_mitos", "-em", action="store_true", default=False,
                        help="Erode mito mask in XY before prediction")
    parser.add_argument("--save_predictions", "-sp", action="store_true", default=False,
                        help="Also write pred/foreground and pred/boundary to the output")
    parser.add_argument("--force", "-f", action="store_true", default=False,
                        help="Overwrite existing output files instead of skipping")
    parser.add_argument("--normalize", "-nz", action="store_true", default=False,
                        help="Percentile-normalize the raw channel instead of standardizing it "
                             "(must match how the model was trained, i.e. train_cristae.py --normalize)")
    return parser


def parse_args():
    parser = build_parser()

    # parse only --config first
    cfg_args, _ = parser.parse_known_args()

    # load config: values become argparse defaults
    if cfg_args.config is not None:
        with open(cfg_args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        parser.set_defaults(**cfg)

    # re-parse the FULL argv so explicit CLI flags override the config defaults
    args = parser.parse_args()

    # enforce required args after config merge
    missing = [n for n in ("model_path", "export_path") if getattr(args, n) is None]
    if missing:
        parser.error("missing required argument(s): "
                     + ", ".join("--" + m for m in missing)
                     + " (provide via config or CLI)")
    if args.h5_paths is None and args.base_path is None:
        parser.error("provide either --h5_paths (or `h5_paths:` in config) or --base_path")

    return args


def main():
    args = parse_args()

    if args.h5_paths is not None:
        h5_paths = list(args.h5_paths)
    else:
        h5_paths = get_file_paths(args.base_path, ext=".h5")
        h5_paths = [p for p in h5_paths if "_combined.h5" in p]
    print(f"Found {len(h5_paths)} files")

    run_cristae_segmentation(
        h5_paths=h5_paths,
        model_path=args.model_path,
        export_path=args.export_path,
        tile_shape=tuple(args.tile_shape),
        erode_mitos=args.erode_mitos,
        add_missing=args.add_missing,
        save_predictions=args.save_predictions,
        base_path=args.base_path,
        force=args.force,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
