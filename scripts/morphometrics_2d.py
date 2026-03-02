import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import synapse.util as util
# from elf.io import open_file
from tifffile import imread
import numpy as np
import argparse
from skimage.measure import label as cc_label
from skimage.measure import regionprops, perimeter, perimeter_crofton
from skimage.segmentation import relabel_sequential


def _is_binary_label_image(lbl):
    if lbl.dtype == bool:
        return True
    u = np.unique(lbl)
    # allow {0}, {1}, {0,1}
    return np.array_equal(u, [0]) or np.array_equal(u, [1]) or np.array_equal(u, [0, 1])


def morphometrics(raw, label, voxel_size=None, use_crofton=True, relabel=True):
    raw = np.asarray(raw)
    label = np.asarray(label)
    if relabel:
        # ensure instance labels
        if _is_binary_label_image(label):
            label = cc_label(label.astype(bool), connectivity=1)
        else:
            # ensure ints and sequential ids (optional but convenient)
            if not np.issubdtype(label.dtype, np.integer):
                label = label.astype(np.int64, copy=False)
            label, _, _ = relabel_sequential(label)
    voxel_size = np.asarray(voxel_size, dtype=float).ravel()
    if voxel_size.size == 1:
        s = float(voxel_size[0])
        voxel_size = (s, s)
    elif voxel_size.size == 2:
        voxel_size = (voxel_size[0], voxel_size[1])
    else:
        raise ValueError(f"Voxel size must have 1 or 2 values for 2D, got {voxel_size.tolist()}")
    orig_voxel_size = voxel_size
    if voxel_size is None:
        voxel_size = (1.0, 1.0)  # (y_nm, x_nm)
    y_nm, x_nm = map(float, voxel_size)
    px_area_nm2 = y_nm * x_nm
    px_size_nm = (y_nm + x_nm) / 2.0  # isotropic -> effectively y_nm == x_nm

    rows = []
    for r in regionprops(label, intensity_image=raw):
        lab = r.label
        area_px = r.area
        area_nm2 = area_px * px_area_nm2

        mask = (r.image > 0)  # binary mask in the region's bounding box
        perim_px = perimeter_crofton(mask) if use_crofton else perimeter(mask)
        perim_nm = perim_px * px_size_nm

        circ = np.nan
        if perim_nm > 0:
            circ = 4.0 * np.pi * area_nm2 / (perim_nm ** 2)  # C=4πA/P² 

        rows.append({
            "label_id": lab,
            "area_px": area_px,
            "area_nm2": area_nm2 if orig_voxel_size is not None else None,
            "perimeter_px": perim_px,
            "perimeter_nm": perim_nm if orig_voxel_size is not None else None,
            "circularity": circ,
            "mean_intensity": r.mean_intensity,
            "min_intensity": r.min_intensity,
            "max_intensity": r.max_intensity,
        })

    return pd.DataFrame(rows)


def main(args):
    # load paths
    paths = util.get_file_paths(args.path, args.ext)

    if args.raw_pattern is not None:
        raw_paths = [path for path in paths if args.raw_pattern in path]
    if args.label_path is not None:
        paths = util.get_file_paths(args.label_path, args.ext)
    if args.label_pattern is not None:
        label_paths = [path for path in paths if args.label_pattern in path]

    raw_paths.sort()
    label_paths.sort()

    assert len(raw_paths) == len(label_paths), (
        f"Expect equal number of raw and label paths, got {len(raw_paths)} and {len(label_paths)}"
    )
    if args.voxel_size is None:
        print("Warning: voxel size not specified")

    all_rows = []
    for raw_path, label_path in tqdm(zip(raw_paths, label_paths), total=len(raw_paths), desc="Computing morphometrics"):
        if args.verbose:
            print(f"raw and label paths: \n{raw_path}\n{label_path}")
        raw = imread(raw_path)
        label = imread(label_path)
        if args.verbose:
            print("raw and label shapes: ", raw.shape, label.shape)
            import napari
            v = napari.Viewer()
            v.add_image(raw)
            v.add_labels(label)
            napari.run()
        df = morphometrics(
            raw,
            label,
            voxel_size=args.voxel_size,
            use_crofton=True,
            relabel=args.relabel,
        )
        df.insert(0, "label_file", str(os.path.basename(label_path)))
        df.insert(0, "raw_file", str(os.path.basename(raw_path)))
        all_rows.append(df)

    out_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    out = args.output_path
    if out is None:
        out = Path(args.path) / "mito_morphometrics.csv" if args.label_path is None else Path(args.label_path) / "cell_morphometrics.csv"
    else:
        out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print("Wrote:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", "-p", type=str, required=True)
    ap.add_argument("--label_path", "-lpth", type=str, default=None)
    ap.add_argument("--ext", "-e", type=str, default=None)
    ap.add_argument("--output_path", "-o", type=str, default=None)
    ap.add_argument("--raw_pattern", "-rp", type=str, default=None)
    ap.add_argument("--label_pattern", "-lp", type=str, default=None)
    ap.add_argument("--voxel_size", "-vs", type=float, nargs="+", default=None)
    ap.add_argument("--relabel", "-r", action="store_true", help="relabel instance labels")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    main(args)