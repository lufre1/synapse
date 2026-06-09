import argparse

import yaml

from synapse.cristae.evaluate import run_cristae_evaluation


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Path to YAML config file. CLI flags override config values.")
    parser.add_argument("-e", "--export_path", default=None,
                        help="Segmentation export directory; used as the default for both "
                             "--labels_path and --segmentations_path when those are not given.")
    parser.add_argument("-l", "--labels_path", default=None,
                        help="Path to GT labels file or directory (defaults to --export_path)")
    parser.add_argument("-le", "--labels_ext", default=None,
                        help="Extension of label files when labels_path is a directory")
    parser.add_argument("-k", "--key", default=None,
                        help="H5 dataset key for labels (omit for .tif)")
    parser.add_argument("-s", "--segmentations_path", default=None,
                        help="Path to segmentation file or directory (defaults to --export_path)")
    parser.add_argument("-se", "--segmentations_ext", default=None,
                        help="Extension of segmentation files when segmentations_path is a directory")
    parser.add_argument("-sk", "--segmentations_key", default=None,
                        help="H5 dataset key for segmentation (omit for .tif)")
    parser.add_argument("-d", "--dataset_name", default=None,
                        help="Optional name tag for single-file CSV row")
    parser.add_argument("-o", "--output_path", default=None,
                        help="CSV output file or directory")
    parser.add_argument("-hd", "--compute_hd95", default=False, action="store_true",
                        help="Compute HD95 (slow for large volumes)")
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

    # the segment step writes both GT labels and predictions into export_path,
    # so default both to it when not given explicitly.
    if args.labels_path is None:
        args.labels_path = args.export_path
    if args.segmentations_path is None:
        args.segmentations_path = args.export_path

    if args.labels_path is None or args.segmentations_path is None:
        parser.error("provide --labels_path and --segmentations_path (or --export_path), "
                     "via config or CLI")

    return args


def main():
    args = parse_args()

    run_cristae_evaluation(
        labels_path=args.labels_path,
        segmentations_path=args.segmentations_path,
        label_key=args.key,
        seg_key=args.segmentations_key,
        output_path=args.output_path,
        dataset_name=args.dataset_name,
        labels_ext=args.labels_ext,
        seg_ext=args.segmentations_ext,
        compute_hd95=args.compute_hd95,
    )


if __name__ == "__main__":
    main()
