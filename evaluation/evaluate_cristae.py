import argparse
import os
import re
from typing import Dict, List
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter
from elf.evaluation import matching, symmetric_best_dice_score
from elf.parallel import label
import synapse.io.util as io
from tqdm import tqdm
from elf.io import open_file
import pandas as pd
from scipy import sparse
from scipy.ndimage import binary_erosion, distance_transform_edt


def cut_after_halo(s: str) -> str:
    # match: halo_z<number>_y<number>_x<number>_
    return re.sub(r'^.*halo_z\d+_y\d+_x\d+_', '', s)


# def export(
#     scores,
#     export_path,
#     ds_name=None
# ):

#     # os.makedirs(export_path, exist_ok=True)
#     result_path = export_path
#     print("Evaluation results are saved to:", result_path)
    
#     if os.path.exists(result_path):
#         results = pd.read_csv(result_path)
#     else:
#         results = None
#     basename = os.path.basename(export_path).split(".")[0]
#     res = pd.DataFrame(
#         [[basename if ds_name is None else f"{basename}-{ds_name}"] + scores], columns=["dataset", "f1-score", "precision", "recall", "dice score", "#pred / #actual"]
#     )
#     if results is None:
#         results = res
#     else:
#         results = pd.concat([results, res])
#     results.to_csv(result_path, index=False)

def export(score_dict, export_path, ds_name=None):
    result_path = export_path
    print("Evaluation results are saved to:", result_path)

    basename = os.path.basename(export_path).split(".")[0]
    dataset_name = basename if ds_name is None else f"{basename}-{ds_name}"

    row = {"dataset": dataset_name, **score_dict}
    res = pd.DataFrame([row])

    if os.path.exists(result_path):
        results = pd.read_csv(result_path)

        # 1) Remove duplicated columns caused by previous exports (keep first occurrence)
        results = results.loc[:, ~results.columns.duplicated()]

        # 2) Also drop any old auto-suffixed dataset columns if they exist
        drop_cols = [c for c in results.columns if c.startswith("dataset.")]

        if drop_cols:
            results = results.drop(columns=drop_cols)

        # 3) Align columns
        all_cols = ["dataset"] + sorted((set(results.columns) | set(res.columns)) - {"dataset"})
        results = results.reindex(columns=all_cols)
        res = res.reindex(columns=all_cols)

        results = pd.concat([results, res], ignore_index=True)
    else:
        results = res

    results.to_csv(result_path, index=False)


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if mask.ndim not in (2, 3):
        raise ValueError("mask must be 2D or 3D")
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((3,) * mask.ndim, dtype=bool)
    er = binary_erosion(mask, structure=structure, border_value=0)
    return mask & (~er)


def hd95_binary(labels, seg, spacing=None):
    labels = labels.astype(bool)
    seg = seg.astype(bool)

    if spacing is None:
        spacing = tuple([1.0] * labels.ndim)

    n_lab = int(labels.sum())
    n_seg = int(seg.sum())

    if n_lab == 0 and n_seg == 0:
        return 0.0
    if n_lab == 0 or n_seg == 0:
        return float("nan")

    lab_surf = _surface(labels)
    seg_surf = _surface(seg)

    dt_to_lab = distance_transform_edt(~lab_surf, sampling=spacing)
    dt_to_seg = distance_transform_edt(~seg_surf, sampling=spacing)

    d_seg_to_lab = dt_to_lab[seg_surf]
    d_lab_to_seg = dt_to_seg[lab_surf]
    all_d = np.concatenate([d_seg_to_lab.ravel(), d_lab_to_seg.ravel()])

    return float(np.percentile(all_d, 95))


def evaluate_binary(labels, seg, spacing=None, ignore_mask=None, eps=1e-8):
    labels = labels.astype(bool)
    seg = seg.astype(bool)

    if ignore_mask is not None:
        ignore_mask = ignore_mask.astype(bool)
        # For voxel-wise confusion, evaluate only where ignore_mask is False
        eval_mask = ~ignore_mask
        labels_eval = labels & eval_mask
        seg_eval = seg & eval_mask
    else:
        labels_eval = labels
        seg_eval = seg

    tp = np.logical_and(seg_eval, labels_eval).sum()
    fp = np.logical_and(seg_eval, ~labels_eval).sum()     # within eval region
    fn = np.logical_and(~seg_eval, labels_eval).sum()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    # For HD95, restrict geometry to evaluated region by zeroing outside
    hd95 = hd95_binary(labels_eval, seg_eval, spacing=spacing)

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "hd95": float(hd95) if np.isfinite(hd95) else np.nan,
        "pred_fg": int(seg_eval.sum()),
        "gt_fg": int(labels_eval.sum()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "eval_voxels": int(labels_eval.size) if ignore_mask is None else int((~ignore_mask).sum())
    }

# def evaluate(labels, seg):
#     assert labels.shape == seg.shape
#     stats = matching(seg, labels)
#     sbd = symmetric_best_dice_score(seg, labels)
#     return [stats["f1"], stats["precision"], stats["recall"], sbd]
# def evaluate_binary(labels, seg, eps=1e-8):
#     labels = labels.astype(bool)
#     seg = seg.astype(bool)

#     tp = np.logical_and(seg, labels).sum()
#     fp = np.logical_and(seg, ~labels).sum()
#     fn = np.logical_and(~seg, labels).sum()

#     precision = tp / (tp + fp + eps)
#     recall = tp / (tp + fn + eps)
#     f1 = 2 * precision * recall / (precision + recall + eps)

#     dice = 2 * tp / (seg.sum() + labels.sum() + eps)  # if you want dice instead of SBD
#     ratio_str = f"{int(seg.sum())} / {int(labels.sum())}"  # voxel ratio (pred_fg / gt_fg)

#     return [float(f1), float(precision), float(recall), float(dice), ratio_str]


def main(args):
    print("Evaluating cristae")
    # evaluate single file files
    if os.path.isfile(args.labels_path) and os.path.isfile(args.segmentations_path):
        if ".tif" not in args.labels_path:
            labels = io.load_data_from_file(args.labels_path)[args.key]
        else:
            labels = io.load_data_from_file(args.labels_path)
        if ".tif" not in args.segmentations_path:
            seg = io.load_data_from_file(args.segmentations_path)[args.segmentations_key]
        else:
            seg = io.load_data_from_file(args.segmentations_path)
        scores = evaluate_binary(labels, seg)
        if args.output_path is None:
            output_path = os.path.splitext(args.labels_path)[0] + "_results.csv"
        export(scores, output_path, args.dataset_name)
    else:
        label_paths = io.get_file_paths(args.labels_path, ext=args.labels_ext)
        segmentation_paths = io.get_file_paths(args.segmentations_path, ext=args.segmentations_ext)
        all_scores = []
        for label_path, segmentation_path in tqdm(zip(label_paths, segmentation_paths), desc="Evaluating cristae in files:"):
            print(f"label and segmentation paths: \n{label_path}\n{segmentation_path}\n")
            filename = "cristae_eval_results"
            if args.output_path is None:
                out_dir = os.path.dirname(label_path)
            else:
                # if it's a file path, write next to it; if it's a directory, write into it
                out_dir = os.path.dirname(args.output_path) if os.path.splitext(args.output_path)[1] else args.output_path
            output_path = os.path.join(out_dir, f"{filename}.csv")
            if ".tif" not in label_path:
                labels = io.load_data_from_file(label_path)[args.key]
                mito_states = None
                try:
                    mito_states = io.load_data_from_file(label_path)["raw_mitos_combined"][1]
                except KeyError as e:
                    print(f"KeyError: {e}")
                    pass
                if mito_states is None:
                    try:
                        mito_states = io.load_data_from_file(label_path)["labels/mitochondria"]
                    except KeyError as e:
                        print(f"KeyError: {e}")
                        pass
            else:
                labels = io.load_data_from_file(label_path)
            if ".tif" not in segmentation_path:
                seg = io.load_data_from_file(segmentation_path)[args.segmentations_key]
            else:
                seg = io.load_data_from_file(segmentation_path)
            if mito_states is not None:
                valid_mask = mito_states == 1
                # labels = labels[valid_mask]
                # seg = seg[valid_mask]
            # scores = evaluate_binary(labels, seg)
            scores = evaluate_binary(labels, seg, ignore_mask=np.logical_not(valid_mask), spacing=None)
            all_scores.append(scores)
            export(scores, output_path, cut_after_halo(os.path.basename(label_path.replace("0.", "0")).split(".")[0]))
        if all_scores:
            # choose what you want to average across files
            avg_keys = ["dice", "precision", "recall", "hd95"]

            avg_dict = {}
            for k in avg_keys:
                vals = [d.get(k, np.nan) for d in all_scores]
                vals = np.asarray(vals, dtype=float)
                avg_dict[k] = float(np.nanmean(vals))  # nanmean ignores NaNs (e.g. undefined hd95)

            # optionally also sum counts across files (micro-style bookkeeping)
            sum_keys = ["tp", "fp", "fn", "pred_fg", "gt_fg", "eval_voxels"]
            for k in sum_keys:
                avg_dict[k] = int(np.nansum([d.get(k, 0) for d in all_scores]))

            export(avg_dict, output_path, "all-files-averaged")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labels_path", default="/home/freckmann15/data/mitochondria/volume-em/embl/cutout_1_committed_objects_leonie_2025-08-07.tif")
    parser.add_argument("-le", "--labels_ext", default=None, help="Extension of label files; leave empty for single file")
    parser.add_argument("-k", "--key", default=None, help="Key to dataset; leave empty for tif files")
    parser.add_argument("-s", "--segmentations_path", required=True)
    parser.add_argument("-se", "--segmentations_ext", default=None, help="Extension of segmentation files; leave empty for single file")
    parser.add_argument("-sk", "--segmentations_key", default=None)
    parser.add_argument("-d", "--dataset_name", default=None)
    parser.add_argument("-o", "--output_path", default=None)
    args = parser.parse_args()
    main(args)
