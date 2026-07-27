"""Mitochondrial-membrane geometry for the cristae junction diagnostics.

The mitochondrial membrane is not segmented anywhere in this corpus, so it is approximated
as the *surface* of the annotated-mito mask (``raw_mitos_combined[1] == 1``). Everything the
junction diagnostics need — the inner membrane shell ("band") and the distance of a crista to
the membrane — follows from a single Euclidean distance transform of that mask.

``scipy.ndimage.distance_transform_edt(mask, sampling=voxel_size)`` handles anisotropic voxels
natively, so thicknesses are expressed in **nm** and no morphological footprint has to be built.
This replaces the per-slice-XY erosion in :mod:`synapse.cristae.label_utils`, which cannot express
a nm thickness and is wrong for the (isotropic, 1.74 nm) tomograms in this corpus.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

# All wichmann tomograms; used when a file carries no `voxel_size` attribute.
DEFAULT_VOXEL_SIZE = (1.74, 1.74, 1.74)


def membrane_distance(mito_mask, voxel_size=DEFAULT_VOXEL_SIZE):
    """Distance in nm from each mito-interior voxel to the nearest non-mito voxel.

    Voxels directly adjacent to the outside get the voxel spacing itself (not 0), so a shell of
    ``k`` voxels has distances ``1*vs .. k*vs``. Outside the mask the result is 0.

    Args:
        mito_mask: boolean (or 0/1) array, the annotated-mito mask.
        voxel_size: (z, y, x) spacing in nm.

    Returns:
        float32 array of the same shape.
    """
    mito_mask = np.asarray(mito_mask, dtype=bool)
    if not mito_mask.any():
        return np.zeros(mito_mask.shape, dtype=np.float32)
    return distance_transform_edt(mito_mask, sampling=voxel_size).astype(np.float32)


def membrane_band(mito_mask, voxel_size=DEFAULT_VOXEL_SIZE, thickness_nm=8.0, distance=None):
    """Boolean inner membrane shell: mito interior within ``thickness_nm`` of the membrane.

    Pass a precomputed ``distance`` (from :func:`membrane_distance`) to avoid recomputing the EDT
    when several thicknesses are evaluated on the same mask.
    """
    mito_mask = np.asarray(mito_mask, dtype=bool)
    if distance is None:
        distance = membrane_distance(mito_mask, voxel_size)
    return mito_mask & (distance <= float(thickness_nm))
