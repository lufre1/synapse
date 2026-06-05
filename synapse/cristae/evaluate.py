import os
import re

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt
from tqdm import tqdm

import synapse.io.util as io
from synapse.evaluation import cut_after_halo as _strip_halo_prefix


def _surface(mask):
    mask = mask.astype(bool)
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
    if not labels.any() and not seg.any():
        return 0.0
    if not labels.any() or not seg.any():
        return float("nan")
    lab_surf = _surface(labels)
    seg_surf = _surface(seg)
    dt_to_lab = distance_transform_edt(~lab_surf, sampling=spacing)
    dt_to_seg = distance_transform_edt(~seg_surf, sampling=spacing)
    all_d = np.concatenate([dt_to_lab[seg_surf].ravel(), dt_to_seg[lab_surf].ravel()])
    return float(np.percentile(all_d, 95))


def evaluate_binary(labels, seg, spacing=None, ignore_mask=None, eps=1e-8, compute_hd95=False):
    """Compute voxel-wise binary segmentation metrics.

    Args:
        labels: Ground truth boolean/integer array.
        seg: Predicted segmentation boolean/integer array.
        spacing: Physical voxel spacing for HD95 (default: isotropic 1.0).
        ignore_mask: Boolean array; True where voxels are excluded from evaluation.
        eps: Epsilon for numerical stability.
        compute_hd95: Whether to compute HD95 (slow for large volumes).

    Returns:
        Dict with dice, precision, recall, hd95, tp, fp, fn, pred_fg, gt_fg, eval_voxels.
    """
    labels = labels.astype(bool)
    seg = seg.astype(bool)
    if ignore_mask is not None:
        eval_mask = ~ignore_mask.astype(bool)
        labels_eval = labels & eval_mask
        seg_eval = seg & eval_mask
    else:
        labels_eval = labels
        seg_eval = seg

    tp = int(np.logical_and(seg_eval, labels_eval).sum())
    fp = int(np.logical_and(seg_eval, ~labels_eval).sum())
    fn = int(np.logical_and(~seg_eval, labels_eval).sum())

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)

    hd95 = np.nan
    if compute_hd95:
        hd95 = hd95_binary(labels_eval, seg_eval, spacing=spacing)

    return {
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "hd95": float(hd95) if np.isfinite(hd95) else np.nan,
        "pred_fg": int(seg_eval.sum()),
        "gt_fg": int(labels_eval.sum()),
        "tp": tp, "fp": fp, "fn": fn,
        "eval_voxels": int(labels_eval.size) if ignore_mask is None else int((~ignore_mask).sum()),
    }


def export_scores(score_dict, export_path, ds_name=None):
    """Append a score row to a CSV file."""
    basename = os.path.basename(export_path).split(".")[0]
    dataset_name = basename if ds_name is None else f"{basename}-{ds_name}"
    res = pd.DataFrame([{"dataset": dataset_name, **score_dict}])

    if os.path.exists(export_path):
        results = pd.read_csv(export_path)
        results = results.loc[:, ~results.columns.duplicated()]
        drop_cols = [c for c in results.columns if c.startswith("dataset.")]
        if drop_cols:
            results = results.drop(columns=drop_cols)
        all_cols = ["dataset"] + sorted((set(results.columns) | set(res.columns)) - {"dataset"})
        results = results.reindex(columns=all_cols)
        res = res.reindex(columns=all_cols)
        results = pd.concat([results, res], ignore_index=True)
    else:
        results = res

    results.to_csv(export_path, index=False)
    print("Evaluation results saved to:", export_path)


def _load_mito_states(label_path):
    """Try to load the mito state channel from known keys."""
    try:
        return io.load_data_from_file(label_path)["raw_mitos_combined"][1]
    except KeyError:
        pass
    try:
        return io.load_data_from_file(label_path)["labels/mitochondria"]
    except KeyError:
        return None


def run_cristae_evaluation(
    labels_path,
    segmentations_path,
    label_key=None,
    seg_key=None,
    output_path=None,
    dataset_name=None,
    labels_ext=None,
    seg_ext=None,
    compute_hd95=False,
):
    """Evaluate cristae segmentation against ground truth.

    Handles both single-file and directory inputs. For H5 files, automatically
    loads the mito state channel and restricts evaluation to annotated mito voxels
    (state == 1).

    Args:
        labels_path: Path to GT file or directory of GT files.
        segmentations_path: Path to segmentation file or directory.
        label_key: H5 dataset key for labels (None for .tif).
        seg_key: H5 dataset key for segmentation (None for .tif).
        output_path: CSV file or directory for results. Defaults to next to labels.
        dataset_name: Optional name tag for the CSV row.
        labels_ext: File extension filter when labels_path is a directory.
        seg_ext: File extension filter when segmentations_path is a directory.
        compute_hd95: Whether to compute HD95.

    Returns:
        List of per-file score dicts.
    """
    is_single = os.path.isfile(labels_path) and os.path.isfile(segmentations_path)

    if is_single:
        labels = io.load_data_from_file(labels_path) if label_key is None else io.load_data_from_file(labels_path)[label_key]
        seg = io.load_data_from_file(segmentations_path) if seg_key is None else io.load_data_from_file(segmentations_path)[seg_key]
        scores = evaluate_binary(labels, seg)
        csv_path = output_path if output_path is not None else os.path.splitext(labels_path)[0] + "_results.csv"
        export_scores(scores, csv_path, dataset_name)
        return [scores]

    label_paths = io.get_file_paths(labels_path, ext=labels_ext or ".h5")
    seg_paths = io.get_file_paths(segmentations_path, ext=seg_ext or ".h5")
    all_scores = []

    for label_path, seg_path in tqdm(zip(label_paths, seg_paths), desc="Evaluating"):
        print(f"label:  {label_path}\nseg:    {seg_path}\n")

        if output_path is None:
            out_dir = os.path.dirname(label_path)
        else:
            out_dir = os.path.dirname(output_path) if os.path.splitext(output_path)[1] else output_path
        csv_path = os.path.join(out_dir, "cristae_eval_results.csv")

        is_tif = label_path.endswith(".tif")
        if is_tif:
            labels = io.load_data_from_file(label_path)
            mito_states = None
        else:
            labels = io.load_data_from_file(label_path)[label_key]
            mito_states = _load_mito_states(label_path)

        seg = io.load_data_from_file(seg_path) if seg_path.endswith(".tif") else io.load_data_from_file(seg_path)[seg_key]

        ignore_mask = None
        if mito_states is not None:
            ignore_mask = mito_states != 1

        scores = evaluate_binary(labels, seg, ignore_mask=ignore_mask, compute_hd95=compute_hd95)
        all_scores.append(scores)

        ds_label = _strip_halo_prefix(
            os.path.basename(label_path.replace("0.", "0")).split(".")[0]
        )
        export_scores(scores, csv_path, ds_label)

    if all_scores:
        avg_keys = ["dice", "precision", "recall", "hd95"]
        avg_dict = {k: float(np.nanmean([d.get(k, np.nan) for d in all_scores])) for k in avg_keys}
        sum_keys = ["tp", "fp", "fn", "pred_fg", "gt_fg", "eval_voxels"]
        for k in sum_keys:
            avg_dict[k] = int(np.nansum([d.get(k, 0) for d in all_scores]))
        export_scores(avg_dict, csv_path, "all-files-averaged")

    return all_scores
