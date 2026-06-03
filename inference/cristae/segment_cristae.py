import argparse

from synapse.cristae.segment import run_cristae_segmentation
from synapse.io.util import get_file_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b", type=str, required=True,
                        help="Root data directory; H5 files are discovered recursively")
    parser.add_argument("--export_path", "-e", type=str, required=True,
                        help="Root output directory")
    parser.add_argument("--model_path", "-m", type=str, required=True,
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
    args = parser.parse_args()

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
    )


if __name__ == "__main__":
    main()
