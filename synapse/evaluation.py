"""Instance-segmentation evaluation helpers shared across the ``evaluation/`` scripts.

These were previously copy-pasted (byte-identical) across ``evaluate_mitos.py``,
``evaluate_mitos_grid.py`` and ``eval_mitos_touching_borders.py``:

* :func:`evaluate_instances` — instance matching (F1/precision/recall) + symmetric best
  Dice + a predicted/actual instance-count ratio string,
* :func:`export_instance_scores` — append a score row to a CSV file,
* :func:`cut_after_halo` — strip the ``halo_z..._y..._x..._`` filename prefix.

For the *binary / semantic* cristae metrics (Dice / HD95) see
:mod:`synapse.cristae.evaluate`.
"""
import os
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from elf.evaluation import matching, symmetric_best_dice_score


def cut_after_halo(s: str) -> str:
    """Strip everything up to and including a ``halo_z<d>_y<d>_x<d>_`` prefix."""
    # match: halo_z<number>_y<number>_x<number>_
    return re.sub(r'^.*halo_z\d+_y\d+_x\d+_', '', s)


def evaluate_instances(labels: np.ndarray, seg: np.ndarray) -> List:
    """Evaluate an instance segmentation against ground-truth instance labels.

    Returns ``[f1, precision, recall, sbd, "<#pred> / <#actual>"]``.
    """
    assert labels.shape == seg.shape
    stats = matching(segmentation=seg, groundtruth=labels, threshold=0.5, criterion="iou", ignore_label=0)
    sbd = symmetric_best_dice_score(segmentation=seg, groundtruth=labels)
    pred_count = len(set(seg.flatten())) - (1 if 0 in seg else 0)
    actual_count = len(set(labels.flatten())) - (1 if 0 in labels else 0)
    ratio_str = f"{pred_count} / {actual_count}"
    return [stats["f1"], stats["precision"], stats["recall"], sbd, ratio_str]


def export_instance_scores(scores: List, export_path: str, ds_name: Optional[str] = None) -> None:
    """Append a row of instance scores to a CSV file at ``export_path``.

    The CSV columns are ``dataset, f1-score, precision, recall, SBD score, #pred / #actual``.
    """
    result_path = export_path
    print("Evaluation results are saved to:", result_path)

    if os.path.exists(result_path):
        results = pd.read_csv(result_path)
    else:
        results = None
    basename = os.path.basename(export_path).split(".")[0]
    res = pd.DataFrame(
        [[basename if ds_name is None else f"{basename}-{ds_name}"] + scores],
        columns=["dataset", "f1-score", "precision", "recall", "SBD score", "#pred / #actual"],
    )
    if results is None:
        results = res
    else:
        results = pd.concat([results, res])
    results.to_csv(result_path, index=False)


def average_instance_scores(all_scores: List[List]) -> List:
    """Mean over the first four numeric columns of a list of score rows.

    Returns ``[mean_f1, mean_precision, mean_recall, mean_sbd, ""]`` — the trailing
    empty string keeps the row aligned with the ``#pred / #actual`` column.
    """
    numeric_scores = [score_list[:4] for score_list in all_scores]
    numeric_array = np.array(numeric_scores, dtype=float)
    avg_scores_numeric = np.mean(numeric_array, axis=0).tolist()
    return avg_scores_numeric + [""]
