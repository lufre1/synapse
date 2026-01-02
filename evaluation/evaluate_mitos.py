import argparse
import os
from typing import Dict, List
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import Counter
from elf.evaluation import matching, symmetric_best_dice_score
import synapse.io.util as io
from tqdm import tqdm
from elf.io import open_file
import pandas as pd
from scipy import sparse


def export(
    scores,
    export_path,
    ds_name=None
):

    # os.makedirs(export_path, exist_ok=True)
    result_path = export_path
    print("Evaluation results are saved to:", result_path)
    
    if os.path.exists(result_path):
        results = pd.read_csv(result_path)
    else:
        results = None
    basename = os.path.basename(export_path).split(".")[0]
    res = pd.DataFrame(
        [[basename if ds_name is None else f"{basename}-{ds_name}"] + scores], columns=["dataset", "f1-score", "precision", "recall", "SBD score", "#pred / #actual"]
    )
    if results is None:
        results = res
    else:
        results = pd.concat([results, res])
    results.to_csv(result_path, index=False)


# def evaluate(labels, seg):
#     assert labels.shape == seg.shape
#     stats = matching(seg, labels)
#     sbd = symmetric_best_dice_score(seg, labels)
#     return [stats["f1"], stats["precision"], stats["recall"], sbd]
def evaluate(labels, seg):
    assert labels.shape == seg.shape
    stats = matching(segmentation=seg, groundtruth=labels, threshold=0.5, criterion="iou", ignore_label=0)
    sbd = symmetric_best_dice_score(segmentation=seg, groundtruth=labels)
    pred_count = len(set(seg.flatten())) - (1 if 0 in seg else 0)
    actual_count = len(set(labels.flatten())) - (1 if 0 in labels else 0)
    ratio_str = f"{pred_count} / {actual_count}"
    return [stats["f1"], stats["precision"], stats["recall"], sbd, ratio_str]


def main(args):
    print("Evaluating mitos")
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
        scores = evaluate(labels, seg)
        if args.output_path is None:
            output_path = os.path.splitext(args.labels_path)[0] + "_results.csv"
        export(scores, output_path, args.dataset_name)
    else:
        label_paths = io.get_file_paths(args.labels_path, ext=args.labels_ext)
        segmentation_paths = io.get_file_paths(args.segmentations_path, ext=args.segmentations_ext)
        for label_path, segmentation_path in tqdm(zip(label_paths, segmentation_paths), desc="Evaluating mitos in files:"):
            print(f"label and segmentation paths: \n{label_path}\n{segmentation_path}\n")
            filename = "mito_eval_results"
            if args.output_path is not None:
                out_dir = os.path.dirname(args.output_path) if os.path.isfile(args.output_path) else os.path.dirname(label_path) if not args.output_path else args.output_path
            else:
                out_dir = os.path.dirname(label_path)
            output_path = os.path.join(out_dir, f"{filename}.csv")
            if ".tif" not in label_path:
                labels = io.load_data_from_file(label_path)[args.key]
            else:
                labels = io.load_data_from_file(label_path)
            if ".tif" not in segmentation_path:
                seg = io.load_data_from_file(segmentation_path)[args.segmentations_key]
            else:
                seg = io.load_data_from_file(segmentation_path)
            scores = evaluate(labels, seg)
            export(scores, output_path, os.path.basename(label_path.replace("0.", "0")).split(".")[0])


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
