#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import imageio.v3 as iio


def main():
    ap = argparse.ArgumentParser(description="Create a multi-panel figure from PNG images.")
    ap.add_argument("-i", "--input", type=str, required=True,
                    help="Input directory containing .png files (or a glob pattern).")
    ap.add_argument("-o", "--output", type=str, default=None,
                    help="Output figure path (.pdf or .png recommended).")
    ap.add_argument("--export_file_format", type=str, default=".png", help="Output file format.")
    ap.add_argument("--n", type=int, default=10, help="Number of images to include.")
    ap.add_argument("--rows", type=int, default=2, help="Number of rows in the grid.")
    ap.add_argument("--cols", type=int, default=5, help="Number of columns in the grid.")
    ap.add_argument("--figsize", type=float, nargs=2, default=(15, 6),
                    metavar=("W", "H"), help="Figure size in inches.")
    ap.add_argument("--dpi", type=int, default=300, help="DPI for raster outputs.")
    ap.add_argument("--title", action="store_true", help="Add each filename as a small title.")
    ap.add_argument("--sort", choices=["name", "mtime"], default="name",
                    help="How to sort images before selecting the first n.")
    ap.add_argument(
        "--pattern", "-p", type=str, default=None,
        help="Only keep files whose path/name contains this substring (e.g. 'axon' or 'fold3')."
    )
    args = ap.parse_args()

    # Support passing either a directory or a glob pattern
    inp = Path(args.input)
    if inp.is_dir():
        paths = list(inp.glob("*.png"))
    else:
        paths = [Path(p) for p in sorted(Path().glob(args.input))]

    if args.sort == "name":
        paths = sorted(paths, key=lambda p: p.name)
    else:  # mtime
        paths = sorted(paths, key=lambda p: p.stat().st_mtime)
    if args.pattern is not None:
        paths = [p for p in paths if args.pattern in str(p)]
    paths = paths[:args.n]
    if len(paths) == 0:
        raise SystemExit("No PNG files found.")

    fig, axes = plt.subplots(args.rows, args.cols, figsize=tuple(args.figsize), constrained_layout=True)

    # axes can be a single Axes if rows=cols=1
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, p in zip(axes_flat, paths):
        im = iio.imread(p)
        ax.imshow(im)
        if args.title:
            ax.set_title(p.stem, fontsize=8)
        ax.axis("off")

    # Hide remaining axes if grid > number of images
    for ax in axes_flat[len(paths):]:
        ax.axis("off")

    if args.output is None:
        out = Path(args.input)
        if not out.is_dir():
            out = out.parent  # if input was a glob pattern
        out = out / f"qualitative_figure{args.export_file_format}"
    else:
        out = Path(args.output)

    out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
    print("Wrote:", out)


if __name__ == "__main__":
    main()