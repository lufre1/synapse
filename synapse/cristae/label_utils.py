import numpy as np
from skimage.morphology import binary_erosion, disk

# Canonical implementation lives in synapse.label_utils; re-exported here for
# backwards compatibility with existing imports of this module.
from synapse.label_utils import find_additional_objects  # noqa: F401


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
