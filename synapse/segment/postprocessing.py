"""Post-processing for segmentation results.

Size filtering (in-core & out-of-core), connected-component cleaning,
bounding-box computation, and volume rescaling utilities.
"""
import time
import numpy as np
from tqdm import tqdm
from itertools import product
from skimage.measure import label as skimage_label
from scipy.ndimage import sum_labels
from skimage.transform import resize, rescale
from skimage.morphology import remove_small_holes, binary_closing
from scipy.ndimage import binary_closing as scipy_binary_closing
from elf import parallel as parallel
import warnings


def apply_size_filter_ooc(seg, min_size, block_shape, verbose=True, out=None):
    if out is None:
        out = seg  # in-place

    shape = seg.shape
    bz, by, bx = block_shape

    # Pass 0: max label
    max_id = 0
    for zz in range(0, shape[0], bz):
        z1 = min(zz + bz, shape[0])
        blk = np.asarray(seg[zz:z1, :, :], dtype=np.uint64)
        max_id = max(max_id, int(blk.max(initial=0)))
    if verbose:
        print("max label id:", max_id)

    if max_id > np.iinfo(np.int64).max:
        raise ValueError(f"max_id too large for bincount/int64: {max_id}")

    # Pass 1: sizes
    counts = np.zeros(max_id + 1, dtype=np.uint64)
    for zz in tqdm(range(0, shape[0], bz), disable=not verbose, desc="SizeFilter pass1 (count)"):
        z1 = min(zz + bz, shape[0])
        for yy in range(0, shape[1], by):
            y1 = min(yy + by, shape[1])
            for xx in range(0, shape[2], bx):
                x1 = min(xx + bx, shape[2])

                blk_u = np.asarray(seg[zz:z1, yy:y1, xx:x1], dtype=np.uint64)
                blk = blk_u.astype(np.int64, copy=False)  # for bincount
                bc = np.bincount(blk.ravel(), minlength=max_id + 1)

                counts[:bc.shape[0]] += bc.astype(np.uint64, copy=False)

    remove = np.where((counts > 0) & (counts < min_size))[0].astype(np.uint64)
    if verbose:
        print("labels to remove:", len(remove))

    # Pass 2: apply
    for zz in tqdm(range(0, shape[0], bz), disable=not verbose, desc="SizeFilter pass2 (apply)"):
        z1 = min(zz + bz, shape[0])
        for yy in range(0, shape[1], by):
            y1 = min(yy + by, shape[1])
            for xx in range(0, shape[2], bx):
                x1 = min(xx + bx, shape[2])

                blk = np.asarray(out[zz:z1, yy:y1, xx:x1], dtype=np.uint64)
                m = np.isin(blk, remove)
                if m.any():
                    blk[m] = 0
                    out[zz:z1, yy:y1, xx:x1] = blk

    return out


def iterate_blocks(shape, block_shape):
    """Yields chunk bounding boxes for out-of-core block processing."""
    ranges = [range(0, s, b) for s, b in zip(shape, block_shape)]
    for starts in product(*ranges):
        yield tuple(slice(s, min(s + b, dim)) for s, b, dim in zip(starts, block_shape, shape))


def apply_size_filter_ooc_optim(
    segmentation, min_size: int, verbose: bool = False,
    block_shape=(128, 256, 256)
):
    if min_size == 0:
        return segmentation

    t0 = time.time()
    block_shape_ = block_shape[1:] if segmentation.ndim == 2 and len(block_shape) == 3 else block_shape

    # parallel.unique is OOC safe
    ids, sizes = parallel.unique(segmentation, return_counts=True, block_shape=block_shape_, verbose=verbose)

    # Convert to set for O(1) lookup speed in Python
    filter_ids = set(ids[sizes < min_size])

    if not filter_ids:
        return segmentation  # Nothing to filter

    # Block-wise filtering
    for bb in iterate_blocks(segmentation.shape, block_shape):
        chunk = segmentation[bb]

        # np.isin works fine here because 'chunk' is small
        mask = np.isin(chunk, list(filter_ids))
        if np.any(mask):
            chunk[mask] = 0
            segmentation[bb] = chunk  # Write back to out-of-core array

    if verbose:
        print("Size filter in", time.time() - t0, "s")

    return segmentation


def compute_bboxes_ooc(seg, block_shape, verbose=True):
    """
    Compute bounding boxes for each label in `seg` out-of-core.
    Returns: bmin (N,3), bmax (N,3) with N=max_id+1, where label 0 is background.
    bmin[l] = (zmin,ymin,xmin), bmax[l] = (zmax+1,ymax+1,xmax+1)
    """
    shape = seg.shape
    bz, by, bx = block_shape

    # max label (blockwise)
    max_id = 0
    for zz in range(0, shape[0], bz):
        z1 = min(zz + bz, shape[0])
        blk = np.asarray(seg[zz:z1, :, :], dtype=np.uint64)
        max_id = max(max_id, int(blk.max(initial=0)))

    n = max_id + 1
    bmin = np.full((n, 3), np.iinfo(np.int32).max, dtype=np.int32)
    bmax = np.zeros((n, 3), dtype=np.int32)
    seen = np.zeros(n, dtype=bool)

    # accumulate per block
    for zz in tqdm(range(0, shape[0], bz), disable=not verbose, desc="BBox pass"):
        z1 = min(zz + bz, shape[0])
        for yy in range(0, shape[1], by):
            y1 = min(yy + by, shape[1])
            for xx in range(0, shape[2], bx):
                x1 = min(xx + bx, shape[2])

                blk = np.asarray(seg[zz:z1, yy:y1, xx:x1], dtype=np.uint64)
                labels = np.unique(blk)
                labels = labels[labels != 0]
                if labels.size == 0:
                    continue

                # For each label present in this block, update bbox using local coords
                for lab in labels:
                    m = (blk == lab)
                    if not m.any():
                        continue
                    coords = np.argwhere(m)  # local coords (dz,dy,dx)
                    zmin, ymin, xmin = coords.min(axis=0)
                    zmax, ymax, xmax = coords.max(axis=0)

                    # convert to global coords; note +1 for stop
                    gmin = np.array([zz + zmin, yy + ymin, xx + xmin], dtype=np.int32)
                    gmax = np.array([zz + zmax + 1, yy + ymax + 1, xx + xmax + 1], dtype=np.int32)

                    lab = int(lab)
                    seen[lab] = True
                    bmin[lab] = np.minimum(bmin[lab], gmin)
                    bmax[lab] = np.maximum(bmax[lab], gmax)

    return bmin, bmax, seen


def postprocess_seg_3d_ooc(seg, block_shape, area_threshold=1000, iterations=4, iterations_3d=8,
                           verbose=True, overwrite_in_place=True):
    """
    seg: zarr.Array (uint64 labels), modified in-place by default.
    """
    # structure elements as in your code
    structure_element = np.ones((3, 3), dtype=bool)
    structure_3d = np.zeros((1, 3, 3), dtype=bool)
    structure_3d[0] = structure_element

    bmin, bmax, seen = compute_bboxes_ooc(seg, block_shape=block_shape, verbose=verbose)

    # iterate labels (skip 0)
    labels = np.flatnonzero(seen)
    labels = labels[labels != 0]

    for lab in tqdm(labels, disable=not verbose, desc="Postprocess objects"):
        z0, y0, x0 = bmin[lab]
        z1, y1, x1 = bmax[lab]
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue

        bb = (slice(int(z0), int(z1)), slice(int(y0), int(y1)), slice(int(x0), int(x1)))

        # read ROI into memory (must fit RAM!)
        roi = np.asarray(seg[bb], dtype=np.uint64)

        mask = (roi == lab)
        if not mask.any():
            continue

        # same operations as your original
        mask = remove_small_holes(mask, area_threshold=area_threshold)
        mask = np.logical_or(scipy_binary_closing(mask, iterations=iterations), mask)
        mask = np.logical_or(scipy_binary_closing(mask, iterations=iterations_3d, structure=structure_3d), mask)

        # write back only where mask is true
        roi[mask] = lab
        seg[bb] = roi

    return seg


def adjust_size(input_volume, scale=None, is_segmentation=False, orig_shape=None):
    """
    Rescale or resize a 2D/3D volume, using interpolation appropriate for images vs. label maps.

    This function has two modes:

    1) Rescaling (when ``orig_shape is None``):
       - Uses ``skimage.transform.rescale`` with the provided ``scale``.

    2) Resizing to a target shape (when ``orig_shape is not None``):
       - Uses ``skimage.transform.resize`` to match ``orig_shape``.

    For segmentation/label volumes (``is_segmentation=True``), nearest-neighbor interpolation
    is used (``order=0`` and ``anti_aliasing=False``) to avoid creating non-integer labels.
    For intensity images (``is_segmentation=False``), default interpolation is used.

    Parameters
    ----------
    input_volume : np.ndarray
        Input image/volume (2D or 3D). The output is cast back to ``input_volume.dtype``.
    scale : float or sequence of float, optional
        Scale factor(s) passed to ``rescale``. Required when ``orig_shape`` is None.
        Examples: ``0.5`` to downsample by 2, or ``(1, 0.5, 0.5)`` for anisotropic scaling.
    is_segmentation : bool, default=False
        If True, treat ``input_volume`` as a label map and use nearest-neighbor interpolation.
    orig_shape : tuple of int, optional
        Target output shape passed to ``resize``. If provided, ``scale`` is ignored.

    Returns
    -------
    np.ndarray
        Rescaled/resized volume with the same dtype as the input.

    Notes
    -----
    - ``preserve_range=True`` is used to avoid normalization to [0, 1] by scikit-image.
    - For segmentation resizing, nearest-neighbor interpolation preserves label identities.
    """
    if orig_shape is None:
        if is_segmentation:
            input_volume = rescale(
                input_volume, scale, preserve_range=True, order=0, anti_aliasing=False,
            ).astype(input_volume.dtype)
        else:
            input_volume = rescale(input_volume, scale, preserve_range=True).astype(input_volume.dtype)
    else:
        if is_segmentation:
            input_volume = resize(input_volume, orig_shape, preserve_range=True, order=0, anti_aliasing=False).astype(input_volume.dtype)
        else:
            input_volume = resize(input_volume, orig_shape, preserve_range=True).astype(input_volume.dtype)
    return input_volume


def convert_white_patches_to_black(img, min_patch_size=20):
    """Set connected white (255) patches of at least `min_patch_size` voxels to 0.

    Parameters
    ----------
    img : np.ndarray
        3D uint8 EM volume.
    min_patch_size : int, optional
        Minimum number of voxels for a connected white patch to be removed.

    Returns
    -------
    np.ndarray
        Image with large white patches set to 0 (same dtype as input).
    """
    if img.dtype != np.uint8:
        if not (np.issubdtype(img.dtype, np.floating) and img.min() >= 0 and img.max() <= 255):
            warnings.warn("img must be uint8, converting to uint8 from " + str(img.dtype))
        img = img.astype(np.uint8)
    if img.ndim not in (2, 3):
        raise ValueError(f"img must be a 2D or 3D array, but got {img.ndim}D")

    # Binary mask of white voxels
    white_mask = img == 255

    # Label connected components
    labeled = skimage_label(white_mask)
    if labeled.max() == 0:
        return img

    # Component sizes (index 0 is background)
    sizes = np.bincount(labeled.ravel())

    # Labels that meet the size threshold (skip background index 0)
    large_labels = np.where(sizes >= min_patch_size)[0]
    large_labels = large_labels[large_labels != 0]

    if large_labels.size == 0:
        return img

    # Mask of large components
    large_mask = np.isin(labeled, large_labels)

    # Apply mask
    out = img.copy()
    out[large_mask] = 0

    return out


def downsample_to_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Downsample an array to the target shape by selecting voxels evenly (every ith voxel) along each axis.

    Parameters:
    - arr: np.ndarray
        The input array to be downsampled.
    - target_shape: tuple
        The desired shape. Each dimension must be <= the corresponding input dimension.

    Returns:
    - np.ndarray
        The downsampled array with the exact target_shape.
    """
    if len(target_shape) != arr.ndim:
        raise ValueError("Target shape must have the same number of dimensions as input array.")

    for i, (s, t) in enumerate(zip(arr.shape, target_shape)):
        if t > s:
            raise ValueError(f"Target shape {target_shape} must not exceed input shape {arr.shape}")

    slices = tuple(
        np.linspace(0, s - 1, t, dtype=int)
        for s, t in zip(arr.shape, target_shape)
    )

    # Use np.ix_ for advanced indexing in N-D
    mesh = np.ix_(*slices)
    return arr[mesh]


def upsample_data(data, factor, is_segmentation=True, target_size=None):
    if factor is None and target_size is None:
        print("Need factor or target size!")
        return
    if factor:
        out_shape = tuple(dim * factor for dim in data.shape)
    elif target_size:
        out_shape = target_size
    if is_segmentation:
        output = resize(data, out_shape, preserve_range=True, order=0, anti_aliasing=False).astype(data.dtype)
    else:
        output = resize(data, out_shape, preserve_range=True).astype(data.dtype)
    return output


def filter_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Removes small objects and keeps only the largest connected component per label.

    Args:
        segmentation (np.ndarray): Labeled segmentation array.
        min_size (int): Minimum allowed size for connected components.

    Returns:
        np.ndarray: Cleaned segmentation with small components removed.
    """
    unique_labels = np.unique(segmentation)
    cleaned_seg = np.zeros_like(segmentation)

    for lbl in unique_labels:
        if lbl == 0:  # Skip background
            continue

        # Isolate current label
        mask = segmentation == lbl

        # Label connected components within this mask
        labeled_mask = skimage_label(mask)
        num_features = int(labeled_mask.max())

        if num_features == 1:  # If there's only one connected component, keep it
            cleaned_seg[mask] = lbl
            continue

        # Compute sizes of each component
        sizes = sum_labels(mask, labeled_mask, index=np.arange(1, num_features + 1))

        # Find the largest component
        largest_component = np.argmax(sizes) + 1  # +1 because labels start at 1

        # Keep only the largest component
        cleaned_seg[labeled_mask == largest_component] = lbl

    return cleaned_seg


def filter_small_objects(segmentation: np.ndarray, min_size: int) -> np.ndarray:
    """Removes small connected components from a labeled segmentation.

    Args:
        segmentation (np.ndarray): Labeled segmentation array.
        min_size (int): Minimum allowed size for connected components.

    Returns:
        np.ndarray: Cleaned segmentation with small components removed.
    """
    # Label connected components
    labeled_array = skimage_label(segmentation)
    num_features = int(labeled_array.max())

    # Compute size of each component
    sizes = sum_labels(segmentation > 0, labeled_array, index=np.arange(1, num_features + 1))

    # Find small components
    small_components = np.where(sizes < min_size)[0] + 1  # +1 because labels start at 1

    # Remove small components
    mask = np.isin(labeled_array, small_components)
    segmentation[mask] = 0  # Set small components to background (0)

    return segmentation


def refine_seg(seg,
               min_size=50000*1,
               area_threshold=1000 * 2,
               block_shape=(128, 256, 256),
               halo=(48, 48, 48)
               ):
    from synapse_net.inference.util import _postprocess_seg_3d
    seg = filter_segmentation(seg)
    seg = filter_small_objects(seg, min_size)
    seg = _postprocess_seg_3d(seg, area_threshold=area_threshold, iterations=4, iterations_3d=8)
    return seg
