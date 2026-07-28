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


def _fast_distance_transform(mask, sampling):
    """EDT for the training hot path: bioimage_cpp if available, else scipy.

    Measured on a padded 32x256x256 training patch (50x274x274): bioimage_cpp 0.175 s vs scipy
    0.907 s, agreeing to 8e-7 (float32 rounding). scipy allocates ~50 B/voxel (an int32 feature
    transform, `np.indices`, a float64 cast and a square); bioimage_cpp returns float32 directly.
    That matters because this runs once per patch in every dataloader worker.

    Single-threaded on purpose: this is called from dataloader workers, which are already
    parallel, so extra threads would oversubscribe the CPUs (and buy ~10% anyway).
    """
    try:
        from bioimage_cpp.distance import distance_transform
    except ImportError:
        return distance_transform_edt(mask, sampling=sampling)
    return distance_transform(mask, sampling=list(sampling), number_of_threads=1)


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


def proximity_band(mito_mask, voxel_size=DEFAULT_VOXEL_SIZE, band_nm=12.0, offset_nm=0.0):
    """Boolean shell ``offset_nm < d <= offset_nm + band_nm`` inside ``mito_mask``.

    Unlike :func:`membrane_band` this is meant to run on a *training patch*, so it corrects for the
    patch borders: a mitochondrion cut by a patch face would otherwise be seen as ending there and
    the EDT would manufacture a membrane band straight through the mito lumen. Padding the mask with
    ``mode="edge"`` before the transform makes a mito that runs off a face continue instead, so no
    boundary is invented; where the face is background, edge replication changes nothing.

    Args:
        mito_mask: boolean (or 0/1) array, the annotated-mito mask.
        voxel_size: (z, y, x) spacing in nm.
        band_nm: thickness of the shell in nm.
        offset_nm: start the shell this far inside the membrane (0 = at the mask surface).

    Returns:
        boolean array of the same shape.
    """
    mito_mask = np.asarray(mito_mask, dtype=bool)
    if not mito_mask.any():
        return np.zeros(mito_mask.shape, dtype=bool)

    lo = float(offset_nm)
    hi = lo + float(band_nm)
    pad = int(np.ceil(hi / float(min(voxel_size)))) + 2
    padded = np.pad(mito_mask, pad, mode="edge")
    distance = _fast_distance_transform(padded, voxel_size)
    crop = tuple(slice(pad, -pad) for _ in range(mito_mask.ndim))
    distance = distance[crop]
    return mito_mask & (distance > lo) & (distance <= hi)


def membrane_proximity_weight(
    mito_state,
    gt_foreground,
    voxel_size=DEFAULT_VOXEL_SIZE,
    band_nm=12.0,
    offset_nm=0.0,
    w_pos=1.0,
    w_neg=1.0,
    annotated_state=1,
    exclude_state_value=2,
):
    """Per-voxel loss weight that emphasises cristae in the mito-membrane proximity band.

    Generalises the binary loss mask used by the cristae training transforms::

        w = 0.0    where state == exclude_state_value   (unannotated mito, ignored)
        w = 1.0    everywhere else                       (today's mask)
        w = w_pos  inside the band AND ground-truth cristae
        w = w_neg  inside the band AND not cristae       (the membrane itself, and the matrix by it)

    The ``w_pos`` / ``w_neg`` split is the point: a single band weight would amplify cristae and the
    membrane equally. ``w_pos`` pushes the model to predict cristae at the junction; ``w_neg``
    separately controls how hard the membrane is pushed down.

    With ``w_pos == w_neg == 1.0`` the result is *exactly* the binary mask, so the default path is
    bit-identical to the unweighted training recipe and costs no distance transform.

    Args:
        mito_state: the semantic mito state channel {0=bg, 1=annotated mito, 2=unannotated mito}.
        gt_foreground: binary cristae ground truth, same shape as ``mito_state``.
        voxel_size: (z, y, x) spacing in nm.
        band_nm: thickness of the proximity band in nm.
        offset_nm: start the band this far inside the membrane.
        w_pos: weight for ground-truth cristae voxels inside the band.
        w_neg: weight for non-cristae voxels inside the band.
        annotated_state: state value marking mitochondria that carry cristae annotations.
        exclude_state_value: state value excluded from the loss.

    Returns:
        float32 array of the same shape as ``mito_state``.
    """
    mito_state = np.asarray(mito_state)
    # 1 where NOT excluded — identical to the inline mask in MitoStateMaskTransform.
    weight = (np.abs(mito_state - float(exclude_state_value)) >= 0.5).astype(np.float32)

    w_pos, w_neg = float(w_pos), float(w_neg)
    if w_pos == 1.0 and w_neg == 1.0:
        return weight

    band = proximity_band(mito_state == annotated_state, voxel_size, band_nm, offset_nm)
    if not band.any():
        return weight

    gt = np.asarray(gt_foreground) > 0
    # Multiply rather than assign so excluded voxels (weight 0) stay excluded.
    if w_pos != 1.0:
        weight[band & gt] *= np.float32(w_pos)
    if w_neg != 1.0:
        weight[band & ~gt] *= np.float32(w_neg)
    return weight
