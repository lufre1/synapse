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


def evaluate(labels, vesicles):
    assert labels.shape == vesicles.shape
    stats = matching(vesicles, labels)
    sbd = symmetric_best_dice_score(vesicles, labels)
    return [stats["f1"], stats["precision"], stats["recall"], sbd]


def map_instances_to_classes_majority(
    pred_inst: np.ndarray,
    gt_sem:    np.ndarray,
    ignore_label: int = 0,
    min_overlap_frac: float = 0.0,
) -> Dict[int, int]:

    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids != 0]          # keep only real objects
    if pred_ids.size == 0:                     # nothing to map
        return {}

    # ------------------------------------------------------------------
    # 2️⃣  Prepare the output dictionary
    # ------------------------------------------------------------------
    mapping: Dict[int, int] = {pid: -1 for pid in pred_ids}

    # ------------------------------------------------------------------
    # 3️⃣  Loop over each instance and vote for the most common GT label
    # ------------------------------------------------------------------
    for pid in pred_ids:
        # mask of the current instance
        mask = pred_inst == pid

        # extract the GT labels that fall inside the instance,
        # ignoring the background/ignore label
        gt_inside = gt_sem[mask]
        gt_inside = gt_inside[gt_inside != ignore_label]

        if gt_inside.size == 0:          # instance never touches a foreground class
            continue

        # count how many voxels belong to each class
        # ``np.bincount`` works because GT labels are non‑negative integers
        counts = np.bincount(gt_inside)

        # class with the largest count (majority vote)
        best_class = int(np.argmax(counts))
        best_overlap = int(counts[best_class])

        # optional fraction check
        if min_overlap_frac > 0.0:
            instance_size = mask.sum()
            if best_overlap / float(instance_size) < min_overlap_frac:
                continue          # keep mapping[pid] == -1

        mapping[pid] = best_class

    return mapping


def evaluate_class(
    gt_sem:      np.ndarray,   # ground‑truth semantic mask (shape = H×W×…)
    pred_inst:   np.ndarray,   # raw instance map produced by your watershed
    class_id:    int,          # the class you want to evaluate (e.g. vesicles = 3)
    *,
    ignore_label:      int = 0,
    min_overlap_frac:  float = 0.0,
    visualize:         bool = False
) -> List[float]:
    """
    Evaluate only the objects belonging to ``class_id``.

    Returns
    -------
    [f1, precision, recall, sbd]  (all floats)

    The function internally:
        1. maps ``pred_inst`` → semantic map,
        2. extracts binary masks for the requested class,
        3. calls ``matching`` and ``symmetric_best_dice_score``.
    """
    # ------------------------------------------------------------------
    # 2.1  Map instances → semantic classes
    # ------------------------------------------------------------------
    mapping = map_instances_to_classes_majority(
        pred_inst,
        gt_sem,
        ignore_label=ignore_label,
        min_overlap_frac=min_overlap_frac,
    )
    pred_sem = relabel_instances(pred_inst, mapping, ignore_label)

    # ------------------------------------------------------------------
    # 2.2  Build binary masks for the *single* class
    # ------------------------------------------------------------------
    gt_mask = (gt_sem == class_id)
    pred_mask = (pred_sem == class_id)

    # If there are no voxels of this class in either image we return zeros
    if not gt_mask.any() and not pred_mask.any():
        return [0.0, 0.0, 0.0, 0.0]

    # ------------------------------------------------------------------
    # 2.3  Compute the metrics
    # ------------------------------------------------------------------
    # ``matching`` expects (prediction, ground‑truth) – check the signature
    # of your library; if it is (gt, pred) just swap the arguments.
    stats = matching(pred_mask, gt_mask)          # returns dict with f1, precision, recall
    sbd = symmetric_best_dice_score(pred_mask, gt_mask)
    
    if visualize:
        print("stats", stats)
        print("sbd", sbd)
        import napari
        viewer = napari.Viewer()
        viewer.add_labels(gt_mask, name=f"gt_mask-{class_id}")
        viewer.add_labels(gt_sem, name="gt_sem")
        viewer.add_labels(pred_mask, name="pred_mask")
        
        napari.run()

    return [stats["f1"], stats["precision"], stats["recall"], sbd]


# def map_instances_to_classes_hungarian_per_class(pred_inst, gt_sem, ignore_label=0):
#     pred_ids = np.unique(pred_inst)
#     pred_ids = pred_ids[pred_ids != 0]
#     gt_classes = np.unique(gt_sem)
#     gt_classes = gt_classes[gt_classes != ignore_label]

#     pred_index = {pid: i for i, pid in enumerate(pred_ids)}
#     gt_index   = {cid: i for i, cid in enumerate(gt_classes)}

#     flat_pred = pred_inst.ravel()
#     flat_gt   = gt_sem.ravel()
#     mask = (flat_pred != 0) & (flat_gt != ignore_label)
#     flat_pred = flat_pred[mask]
#     flat_gt   = flat_gt[mask]

#     rows = np.vectorize(pred_index.get)(flat_pred)
#     cols = np.vectorize(gt_index.get)(flat_gt)

#     overlap = sparse.coo_matrix(
#         (np.ones_like(rows, dtype=np.int64), (rows, cols)),
#         shape=(len(pred_ids), len(gt_classes)),
#     ).tocsr()

#     mapping = {pid: -1 for pid in pred_ids}
#     # solve a separate assignment for each GT class
#     for c_idx, class_id in enumerate(gt_classes):
#         # extract the column for this class
#         col = overlap.getcol(c_idx).toarray().ravel()
#         # rows (instances) that have any overlap with this class
#         candidate_rows = np.where(col > 0)[0]
#         if candidate_rows.size == 0:
#             continue
#         # cost matrix is just -overlap for those rows
#         cost = -col[candidate_rows][:, None]   # shape (n_candidates, 1)
#         row_ind, _ = linear_sum_assignment(cost)
#         # assign the chosen instance(s) to this class
#         for r in candidate_rows[row_ind]:
#             mapping[pred_ids[r]] = class_id
#     return mapping


# def map_instances_to_classes_sparse(pred_inst, gt_sem, ignore_label=0):
#     pred_ids = np.unique(pred_inst)
#     pred_ids = pred_ids[pred_ids != 0]
#     gt_classes = np.unique(gt_sem)
#     gt_classes = gt_classes[gt_classes != ignore_label]

#     pred_index = {pid: i for i, pid in enumerate(pred_ids)}
#     gt_index   = {cid: i for i, cid in enumerate(gt_classes)}

#     flat_pred = pred_inst.ravel()
#     flat_gt   = gt_sem.ravel()
#     mask = (flat_pred != 0) & (flat_gt != ignore_label)

#     rows = np.vectorize(pred_index.get)(flat_pred[mask])
#     cols = np.vectorize(gt_index.get)(flat_gt[mask])

#     # Build a sparse contingency matrix (counts are summed automatically)
#     overlap_sparse = sparse.coo_matrix(
#         (np.ones_like(rows, dtype=np.int64), (rows, cols)),
#         shape=(len(pred_ids), len(gt_classes)),
#     ).tocsr()

#     # Hungarian on the dense version (cost matrix must be dense)
#     cost = -overlap_sparse.toarray().astype(np.float64)
#     row_ind, col_ind = linear_sum_assignment(cost)

#     mapping = {pid: -1 for pid in pred_ids}
#     for r, c in zip(row_ind, col_ind):
#         if overlap_sparse[r, c] > 0:
#             mapping[pred_ids[r]] = gt_classes[c]
#     return mapping


def relabel_instances(pred_inst: np.ndarray,
                      mapping: dict[int, int],
                      ignore_label: int = 0) -> np.ndarray:
    """
    Convert the instance map into a semantic map (uint8) using the mapping.
    Voxels belonging to unmapped instances become background (ignore_label).
    """
    out = np.full_like(pred_inst, fill_value=ignore_label, dtype=np.uint8)

    for inst_id, class_id in mapping.items():
        if class_id == -1:          # no foreground overlap → keep background
            continue
        out[pred_inst == inst_id] = class_id

    return out


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
        [[basename if ds_name is None else f"{basename}-{ds_name}"] + scores], columns=["dataset", "f1-score", "precision", "recall", "SBD score"]
    )
    if results is None:
        results = res
    else:
        results = pd.concat([results, res])
    results.to_csv(result_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", "-b",  type=str, default="/scratch-grete/usr/nimlufre/cellmap/test_segmentations_microsam-cellmaps-vit_b_em_organelles-bs1-ps256-resized-wocytonucmem-new-seg/", help="Path to the root data directory")
    parser.add_argument("--file_extension", "-fe",  type=str, default=".h5", help="Path to the root data directory")
    parser.add_argument("--pred_key", "-s",  type=str, default="segmentation", help="Path to save the data to")
    parser.add_argument("--gt_key", "-g",  type=str, default="sem_label", help="Path to save the data to")
    parser.add_argument("--force_override", "-fo", action="store_true", default=False, help="Force overwrite of existing files")
    parser.add_argument("--export_path", "-e",  type=str, default=None, help="Path to the root data directory. If None, then base_path is used")
    parser.add_argument("--with_classes", "-c",  action="store_true", default=False, help="Evaluate classes")
    parser.add_argument("--visualize", "-v", action="store_true", default=False, help="Visualize data with napari")
    parser.add_argument("--visualize_classes", "-vc", action="store_true", default=False, help="Visualize classes with napari")
    args = parser.parse_args()

    # Load data
    h5_paths = io.load_file_paths(args.base_path, args.file_extension)

    for path in tqdm(h5_paths):
        with open_file(path) as f:
            pred_inst = f[args.pred_key][:]
            gt_sem = f[args.gt_key][:]
            # breakpoint()
            if not args.with_classes:
                exp_name = os.path.basename(path).split(".")[0] + "_evaluation_test.csv"
                if args.export_path is not None:
                    exp_path = os.path.join(args.export_path, exp_name)
                else:
                    exp_path = os.path.join(args.base_path, exp_name)
                if os.path.exists(exp_path) and not args.force_override:
                    print("File already exists:", exp_path)
                    continue
                else:
                    print("Overriding")
                # mapping = map_instances_to_classes_sparse(pred_inst, gt_sem, ignore_label=0)
                mapping = map_instances_to_classes_majority(pred_inst, gt_sem, ignore_label=0, min_overlap_frac=0.05)
                pred_sem = relabel_instances(pred_inst, mapping, ignore_label=0)
                
                if args.visualize:
                    import napari
                    viewer = napari.Viewer()
                    viewer.add_image(pred_inst)
                    # viewer.add_image(mapping)
                    viewer.add_image(pred_sem)
                    viewer.add_image(gt_sem)
                    napari.run()
                    continue
                
                scores = evaluate(pred_sem, gt_sem)
                export(scores, exp_path)
            
            if args.with_classes:
                exp_name = os.path.basename(path).split(".")[0]
                if args.export_path is not None:
                    exp_path = os.path.join(args.export_path, exp_name)
                else:
                    exp_path = os.path.join(args.base_path, exp_name)
                out_path = exp_path + "-eval-classes.csv"
                if os.path.exists(out_path) and not args.force_override:
                    print("File already exists:", out_path)
                    continue
                elif os.path.exists(out_path) and args.force_override:
                    print("Overriding", out_path)
                    os.remove(out_path)
                for c in np.unique(gt_sem):
                    if c != 0:
                        scores = evaluate_class(
                            gt_sem,
                            pred_inst,
                            class_id=c,
                            ignore_label=0,
                            min_overlap_frac=0.05,
                            visualize=args.visualize_classes
                        )
                        export(scores, out_path, ds_name=f"cls_{c}")


if __name__ == "__main__":
    main()
