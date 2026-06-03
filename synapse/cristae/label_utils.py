import numpy as np
from skimage.morphology import binary_erosion, disk
from skimage.segmentation import relabel_sequential
from elf.evaluation.matching import label_overlap, intersection_over_union


def binarize_and_erode_xy(mito_labels, radius_xy):
    """Convert mito instance labels to a binary mask and erode per-slice in XY."""
    mito_bin = (mito_labels > 0)
    if radius_xy <= 0:
        return mito_bin.astype(mito_labels.dtype)
    fp = disk(radius_xy)
    if mito_bin.ndim == 3:
        eroded = np.stack(
            [binary_erosion(mito_bin[z], footprint=fp) for z in range(mito_bin.shape[0])],
            axis=0,
        )
    else:
        eroded = binary_erosion(mito_bin, footprint=fp)
    return eroded.astype(mito_labels.dtype)


def find_additional_objects(ground_truth, segmentation, matching_threshold=0.5):
    """Return segmentation objects that are not sufficiently covered by ground_truth (IoU <= threshold)."""
    ground_truth = relabel_sequential(ground_truth)[0]
    segmentation = relabel_sequential(segmentation)[0]

    overlap, _ = label_overlap(segmentation, ground_truth)
    iou = intersection_over_union(overlap)

    matched_ids = {
        seg_id
        for seg_id in np.unique(segmentation)
        if seg_id != 0 and iou[seg_id, :].max() > matching_threshold
    }

    additional = segmentation.copy()
    for mid in matched_ids:
        additional[additional == mid] = 0

    return relabel_sequential(additional)[0]
