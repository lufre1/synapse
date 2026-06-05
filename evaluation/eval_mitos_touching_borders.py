import argparse
import os
from typing import Set
import numpy as np
import synapse.io.util as io
import synapse.evaluation as seval
from tqdm import tqdm
from scipy import ndimage
from typing import Tuple
import napari


def _load_data(label_path, segmentation_path, lk, sk):
    if ".tif" not in label_path:
        labels = io.load_data_from_file(label_path)[lk]
    else:
        labels = io.load_data_from_file(label_path)
    if ".tif" not in segmentation_path:
        seg = io.load_data_from_file(segmentation_path)[sk]
    else:
        seg = io.load_data_from_file(segmentation_path)
    return labels, seg


def _remove_instances_with_touching_borders(
    labels: np.ndarray,
    seg: np.ndarray,
    n_touching_borders: int = 2,
    disregard_z: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove every connected component that touches ``n_touching_borders`` or
    more faces **in its own image** (ground‑truth *or* prediction).

    Parameters
    ----------
    labels, seg : np.ndarray
        Ground‑truth and predicted segmentations (2‑D or 3‑D). 0 is background.
    n_touching_borders : int, optional
        Maximum number of distinct faces an instance may touch.
        Instances that touch *≥* this number are removed.
    disregard_z : bool, optional
        If True the Z‑axis (the first spatial axis in a 3‑D volume) is ignored
        when counting touching faces.

    Returns
    -------
    filtered_labels, filtered_seg : np.ndarray
        Copies of the inputs with the offending instances set to 0.
    """
    # ------------------------------------------------------------------ #
    # 0) sanity checks
    # ------------------------------------------------------------------ #
    if labels.shape != seg.shape:
        raise ValueError("`labels` and `seg` must have the same shape.")
    if n_touching_borders <= 0:
        return labels.copy(), seg.copy()

    # Work on copies – the caller’s arrays stay untouched.
    filtered_labels = labels.copy()
    filtered_seg    = seg.copy()

    ndim  = labels.ndim
    shape = labels.shape

    # ------------------------------------------------------------------ #
    # 1) Helper: how many faces does a bounding box touch?
    # ------------------------------------------------------------------ #
    def _borders_touched(slices: Tuple[slice, ...]) -> int:
        cnt = 0
        for ax, slc in enumerate(slices):
            # Skip Z‑axis when requested (only for 3‑D data).
            if disregard_z and ndim == 3 and ax == 0:
                continue
            if slc.start == 0:
                cnt += 1                     # low face
            if slc.stop == shape[ax]:
                cnt += 1                     # high face
        return cnt

    # ------------------------------------------------------------------ #
    # 2) Helper: return the set of IDs that must be removed from ONE map
    # ------------------------------------------------------------------ #
    def _ids_to_remove(mask: np.ndarray) -> Set[int]:
        """IDs whose component touches > n_touching_borders faces."""
        uniq = np.unique(mask)
        uniq = uniq[uniq != 0]                     # drop background
        if uniq.size == 0:
            return set()

        # Build a dense label image required by ndimage.find_objects.
        dense_map = np.zeros(int(mask.max() + 1), dtype=np.int32)
        dense_map[uniq] = np.arange(1, len(uniq) + 1)

        dense_mask = dense_map[mask]               # 0/1/2/…
        objects = ndimage.find_objects(dense_mask)  # list of slice tuples

        bad_ids = set()
        for orig_id, bbox in zip(uniq, objects):
            if bbox is None:
                continue
            if _borders_touched(bbox) > n_touching_borders:
                bad_ids.add(orig_id)
        return bad_ids

    # ------------------------------------------------------------------ #
    # 3) Determine offending IDs **independently** for each image
    # ------------------------------------------------------------------ #
    ids_remove_labels = _ids_to_remove(labels)
    ids_remove_seg    = _ids_to_remove(seg)

    # ------------------------------------------------------------------ #
    # 4) Zero‑out the offending instances **only in their own image**
    # ------------------------------------------------------------------ #
    if ids_remove_labels:
        # Fast boolean indexing via a lookup table
        lookup = np.zeros(int(filtered_labels.max()) + 1, dtype=bool)
        lookup[list(ids_remove_labels)] = True
        filtered_labels[lookup[filtered_labels]] = 0

    if ids_remove_seg:
        lookup = np.zeros(int(filtered_seg.max()) + 1, dtype=bool)
        lookup[list(ids_remove_seg)] = True
        filtered_seg[lookup[filtered_seg]] = 0

    return filtered_labels, filtered_seg


def main(args):
    print("Evaluating mitos")
    # evaluate single file files
    if os.path.isfile(args.labels_path) and os.path.isfile(args.segmentations_path):
        labels, seg = _load_data(args.labels_path, args.segmentations_path, args.key, args.segmentations_key)
        scores = seval.evaluate_instances(labels, seg)
        if args.output_path is None:
            output_path = os.path.splitext(args.labels_path)[0] + "_results.csv"
        seval.export_instance_scores(scores, output_path, args.dataset_name)
    else:
        label_paths = io.get_file_paths(args.labels_path, ext=args.labels_ext)
        segmentation_paths = io.get_file_paths(args.segmentations_path, ext=args.segmentations_ext)
        all_scores = []
        for label_path, segmentation_path in tqdm(zip(label_paths, segmentation_paths), desc="Evaluating mitos in files:"):
            print(f"label and segmentation paths: \n{label_path}\n{segmentation_path}\n")
            filename = f"mito_eval_results_max_touching_borders{args.max_borders}_disregard_z{args.disregard_z}"
            if args.output_path is not None:
                out_dir = os.path.dirname(args.output_path) if os.path.isfile(args.output_path) else os.path.dirname(label_path) if not args.output_path else args.output_path
            else:
                out_dir = os.path.dirname(segmentation_path)
            output_path = os.path.join(out_dir, f"{filename}.csv")
            labels_orig, seg_orig = _load_data(label_path, segmentation_path, args.key, args.segmentations_key)
            labels, seg = _remove_instances_with_touching_borders(
                labels_orig,
                seg_orig,
                args.max_borders,
                args.disregard_z
                )
            if args.verbose:
                v = napari.Viewer()
                v.title = f"border filtered ({args.max_borders}) labels and segmentations"
                v.add_labels(labels_orig, name="labels")
                v.add_labels(seg_orig, name="segmentations")
                v.add_labels(labels, name="filtered labels")
                v.add_labels(seg, name="filtered segmentations")
                v.grid.enabled = True
                napari.run()
            # continue
            scores = seval.evaluate_instances(labels, seg)
            all_scores.append(scores)
            seval.export_instance_scores(scores, output_path, seval.cut_after_halo(os.path.basename(label_path.replace("0.", "0")).split(".")[0]))
        # Calculate and export average scores across all files
        if all_scores:
            avg_scores = seval.average_instance_scores(all_scores)
            seval.export_instance_scores(avg_scores, output_path, "all-files-averaged")


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
    parser.add_argument("-b", "--max_borders", type=int, default=2, help="Maximum allowed number of touching borders for an instance")
    parser.add_argument("-z", "--disregard_z", default=False, action="store_true", help="Disregard z-dimension when computing touching borders")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
