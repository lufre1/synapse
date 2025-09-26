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
    if ".tif" not in args.labels_path:
        labels = io.load_data_from_file(args.labels_path)[args.key]
    else:
        labels = io.load_data_from_file(args.labels_path)
    if ".tif" not in args.segmentations_path:
        seg = io.load_data_from_file(args.segmentations_path)[args.segmentations_key]
    else:
        seg = io.load_data_from_file(args.segmentations_path)
    scores = evaluate(labels, seg)
    export(scores, args.output_path, args.dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--labels_path", default="/home/freckmann15/data/mitochondria/volume-em/embl/cutout_1_committed_objects_leonie_2025-08-07.tif")
    parser.add_argument("-k", "--key", default=None)
    parser.add_argument("-s", "--segmentations_path", required=True)
    parser.add_argument("-sk", "--segmentations_key", default=None)
    parser.add_argument("-d", "--dataset_name", default=None)
    parser.add_argument("-o", "--output_path", default="/home/freckmann15/data/mitochondria/volume-em/embl/paper/cutout_1_eval_segmentation_results.csv")
    args = parser.parse_args()
    main(args)