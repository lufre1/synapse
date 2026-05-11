import fnmatch
import os
from glob import glob
import time
import warnings
import h5py
import mrcfile
import tifffile
import imageio.v3 as iio
import z5py
import zarr
from elf.io import open_file
import elf.parallel as parallel
import elf.wrapper as wrapper
from elf.wrapper.base import MultiTransformationWrapper
from tqdm import tqdm
import napari
import torch
import torch_em
from torch_em.util.prediction import predict_with_halo
import torch.nn as nn
import numpy as np
import yaml
import random
from skimage.measure import regionprops
from scipy.ndimage import sum_labels
from skimage.measure import label
from skimage.transform import resize, rescale
from skimage.morphology import remove_small_holes, binary_closing
from scipy.ndimage import binary_closing as scipy_binary_closing
# from synapse_net.file_utils import read_ome_zarr
from synapse.h5_util import read_data, read_voxel_size
from torch_em.model import AnisotropicUNet
# used for combined_datasets
from typing import Dict, List, Union, Tuple, Optional, Any
from numpy.typing import ArrayLike
import nifty.tools as _nt
from itertools import product

# Define the data path and filename
# data_path = "/scratch-grete/projects/nim00007/data/mitochondria/moebius/em_tomograms_v1/170-PLP-wt/170_2_rec.h5"
# data_format = "*.h5"


def export_ooc_to_h5(ooc_array, h5_file, dataset_name, exp_scale=1, chunk_shape=(128, 256, 256)):
    """Safely writes an out-of-core array to an HDF5 file chunk-by-chunk."""
    shape = ooc_array.shape
    dtype = ooc_array.dtype
    
    # Calculate scaled dimensions
    if exp_scale != 1:
        out_shape = tuple(max(1, s // exp_scale + (1 if s % exp_scale else 0)) for s in shape)
        out_chunk_shape = tuple(max(1, c // exp_scale) for c in chunk_shape)
    else:
        out_shape = shape
        out_chunk_shape = chunk_shape
        
    # Pre-allocate the HDF5 dataset on disk
    ds = h5_file.create_dataset(
        dataset_name, 
        shape=out_shape, 
        chunks=out_chunk_shape, 
        dtype=dtype, 
        compression="gzip"
    )
    
    # Iterate through the out-of-core array in blocks
    ranges = [range(0, s, c) for s, c in zip(shape, chunk_shape)]
    for starts in product(*ranges):
        # 1. Define the input slice
        bb_in = tuple(slice(s, min(s + c, dim)) for s, c, dim in zip(starts, chunk_shape, shape))
        
        # 2. Read exactly ONE chunk into system RAM
        chunk_data = ooc_array[bb_in]
        
        # 3. Apply downscaling to the chunk if requested
        if exp_scale != 1:
            chunk_data = chunk_data[::exp_scale, ::exp_scale, ::exp_scale]
            
        # 4. Define the output slice and write to disk
        bb_out = tuple(slice(s.start // exp_scale, s.start // exp_scale + chunk_data.shape[i]) for i, s in enumerate(bb_in))
        ds[bb_out] = chunk_data

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


def read_voxel_size_h5(file_path, dataset_name="raw"):
    # deprecated, try use read_voxel_size from synapse.h5_util instead
    voxel_size = None
    key = [k for k in get_all_datasets(file_path) if dataset_name in k]
    if len(key) != 1:
        print(f"Warning: Could not find dataset {dataset_name} in {file_path}.")
        return voxel_size
    try:
        with h5py.File(file_path, "r") as f:
            voxel_size = f[key[0]].attrs["voxel_size"]
    except KeyError:
        print(f"Warning: Could not find voxel_size attribute in {file_path} of dataset {dataset_name}.")
    return voxel_size


def get_3d_model(
    out_channels: int,
    in_channels: int = 1,
    scale_factors: Tuple[Tuple[int, int, int]] = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    initial_features: int = 32,
    final_activation: str = "Sigmoid",
    norm: str = None,
    gain: int = 2,
) -> torch.nn.Module:
    """Get the U-Net model for 3D segmentation tasks.

    Args:
        out_channels: The number of output channels of the network.
        scale_factors: The downscaling factors for each level of the U-Net encoder.
        initial_features: The number of features in the first level of the U-Net.
            The number of features increases by a factor of two in each level.
        final_activation: The activation applied to the last output layer.

    Returns:
        The U-Net.
    """
    model = AnisotropicUNet(
        scale_factors=scale_factors,
        in_channels=in_channels,
        out_channels=out_channels,
        initial_features=initial_features,
        gain=gain,
        final_activation=final_activation,
        norm=norm
    )
    return model


def get_prediction_torch_em(
    input_volume: ArrayLike,  # [z, y, x]
    tiling: Dict[str, Dict[str, int]],  # {"tile": {"z": int, ...}, "halo": {"z": int, ...}}
    model_path: Optional[str] = None,
    model: Optional[torch.nn.Module] = None,
    verbose: bool = True,
    with_channels: bool = False,
    mask: Optional[ArrayLike] = None,
    prediction: Optional[ArrayLike] = None,
    devices: Optional[List[str]] = None,
    preprocess: Optional[callable] = None,
    grid_shift: Optional[Tuple[float, ...]] = None
) -> np.ndarray:
    """Run prediction using torch-em on a given volume.

    Args:
        input_volume: The input volume to predict on.
        model_path: The path to the model checkpoint if 'model' is not provided.
        model: Pre-loaded model. Either model_path or model is required.
        tiling: The tiling configuration for the prediction.
        verbose: Whether to print timing information.
        with_channels: Whether to predict with channels.
        mask: Optional binary mask. If given, the prediction will only be run in
            the foreground region of the mask.
        prediction: An array like object for writing the prediction.
            If not given, the prediction will be computed in moemory.
        devices: The devices for running prediction. If not given will use the GPU
            if available, otherwise the CPU.

    Returns:
        The predicted volume.
    """
    # get block_shape and halo
    block_shape = [tiling["tile"]["z"], tiling["tile"]["x"], tiling["tile"]["y"]]
    halo = [tiling["halo"]["z"], tiling["halo"]["x"], tiling["halo"]["y"]]

    t0 = time.time()
    if devices is None:
        devices = ["cuda" if torch.cuda.is_available() else "cpu"]

    # Suppress warning when loading the model.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model is None:
            if os.path.isdir(model_path):  # Load the model from a torch_em checkpoint.
                model = torch_em.util.load_model(checkpoint=model_path, device=devices[0])
            else:  # Load the model directly from a serialized pytorch model.
                model = torch.load(model_path, weights_only=False)

    # Run prediction with the model.
    with torch.no_grad():

        # Deal with 2D segmentation case
        if len(input_volume.shape) == 2:
            block_shape = [block_shape[1], block_shape[2]]
            halo = [halo[1], halo[2]]

        if mask is not None:
            if verbose:
                print("Run prediction with mask.")
            mask = mask.astype("bool")

        if preprocess is None:
            preprocess = None if isinstance(input_volume, np.ndarray) else torch_em.transform.raw.standardize
        else:
            preprocess = preprocess
        prediction = predict_with_halo(
            input_volume, model, gpu_ids=devices,
            block_shape=block_shape, halo=halo,
            preprocess=preprocess, with_channels=with_channels, mask=mask,
            output=prediction,
            grid_shift=grid_shift
        )
    if verbose:
        print("Prediction time in", time.time() - t0, "s")
    return prediction


def segment_axons(
    foreground: np.ndarray,
    boundary: np.ndarray=None,
    block_shape=(128, 256, 256),
    halo=(32, 48, 48),
    seed_distance=4,
    boundary_threshold=0.15,
    foreground_threshold=0.5,
    min_size=2000,
    area_threshold=500,
    dist=None,
    post_iter=4,
    post_iter3d=8,
    verbose=False,
    return_binary=False
):
    if verbose:
        print("Run axon segmentation.")
        print("foreground shape:", foreground.shape)
    if boundary is not None:
        return segment_mitos(
            foreground=foreground,
            boundary=boundary,
            foreground_threshold=foreground_threshold,
            boundary_threshold=boundary_threshold,
            seed_distance=seed_distance,
            min_size=min_size,
            area_threshold=area_threshold,
            post_iter3d=post_iter3d
        )["segmentation"]
    # get the segmentation via seeded watershed
    t0 = time.time()
    seg = parallel.label(foreground > foreground_threshold, block_shape=block_shape, verbose=verbose)
    if verbose:
        print("Compute connected components in", time.time() - t0, "s")

    # size filter
    t0 = time.time()
    ids, sizes = parallel.unique(seg, return_counts=True, block_shape=block_shape, verbose=verbose)
    filter_ids = ids[sizes < min_size]
    seg[np.isin(seg, filter_ids)] = 0
    if verbose:
        print("Size filter in", time.time() - t0, "s")
    if return_binary:
        seg = np.where(seg > 0, 1, 0)
    return seg


def segment_axons_ooc(
    pred,
    out_path=None,                # zarr directory for outputs
    out_key="seg",
    block_shape=(128, 256, 256),
    foreground_threshold=0.5,
    min_size=2000,
    compressor=None,
    verbose=False,
    return_binary=False,
):
    if out_path is None:
        raise ValueError("out_path is required for out-of-core output")

    # --- SAFELY HANDLE 4D vs 3D INPUT ---
    if pred.ndim == 4:
        if verbose: 
            print(f"Detected 4D input with shape {pred.shape}. Lazily wrapping channel 0.")
        fg_3d = ZarrChannelWrapper(pred, channel=0)
        shape = fg_3d.shape
    elif pred.ndim == 3:
        fg_3d = pred
        shape = pred.shape
    else:
        raise ValueError(f"Expected 3D or 4D prediction input, got {pred.ndim}D")

    root = zarr.open(out_path, mode="a")

    # Define out-of-core datasets
    mask = root.require_dataset(
        "mask", shape=shape, chunks=block_shape, dtype="uint8",
        overwrite=True, compressor=compressor
    )
    seg = root.require_dataset(
        out_key, shape=shape, chunks=block_shape, dtype="uint64",
        overwrite=True, compressor=compressor
    )

    # 1) Build mask blockwise (OOC)
    if verbose: print("Thresholding foreground out-of-core...")
    bz, by, bx = block_shape
    Z, Y, X = shape
    for z0 in range(0, Z, bz):
        z1 = min(z0 + bz, Z)
        for y0 in range(0, Y, by):
            y1 = min(y0 + by, Y)
            for x0 in range(0, X, bx):
                x1 = min(x0 + bx, X)
                # Safely load only a 3D block via the wrapper
                blk = fg_3d[z0:z1, y0:y1, x0:x1]  
                mask[z0:z1, y0:y1, x0:x1] = (blk > foreground_threshold).astype(np.uint8)

    # 2) Blockwise connected components
    if verbose: print("Computing connected components out-of-core...")
    parallel.label(mask, block_shape=block_shape, out=seg, verbose=verbose)
    del root["mask"]

    # 3) Size filter OOC
    if min_size and min_size > 0:
        if verbose: print(f"Applying size filter (min_size={min_size})...")
        seg = apply_size_filter_ooc(seg, min_size=min_size, block_shape=block_shape, out=seg, verbose=verbose)

    if return_binary:
        if verbose: print("Converting back to binary out-of-core...")
        # Convert to binary OOC
        for z0 in range(0, Z, bz):
            z1 = min(z0 + bz, Z)
            for y0 in range(0, Y, by):
                y1 = min(y0 + by, Y)
                for x0 in range(0, X, bx):
                    x1 = min(x0 + bx, X)
                    blk = seg[z0:z1, y0:y1, x0:x1]
                    seg[z0:z1, y0:y1, x0:x1] = (blk > 0).astype(np.uint8)

    return seg



def segment_mitos(
    foreground: np.ndarray,
    boundary: np.ndarray,
    block_shape=(128, 256, 256),
    halo=(32, 48, 48),
    seed_distance=4,
    boundary_threshold=0.15,
    foreground_threshold=0.85,
    min_size=2000,
    area_threshold=500,
    dist=None,
    post_iter=4,
    post_iter3d=8
):
    """Return a dict with segmentation, seeds, distance map and h‑map."""
    # ------------------------------------------------------------------
    #  The code you already had – no changes required
    # ------------------------------------------------------------------
    from synapse_net.inference.util import apply_size_filter, _postprocess_seg_3d
    boundaries = boundary
    if dist is None:
        dist = parallel.distance_transform(
            boundaries < boundary_threshold, halo=halo, verbose=True, block_shape=block_shape
        )
    # hmap = (dist.max() - dist) / (dist.max() + 1e-6)  # inverse
    # hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-12)  # normalise
    # barrier_mask = np.logical_and(boundaries > boundary_threshold, foreground < foreground_threshold)
    # hmap[barrier_mask] = (hmap + boundaries).max()
    hmap = (dist.max() - dist) / dist.max()
    hmap[np.logical_and(boundaries > boundary_threshold, foreground < boundary_threshold)] = (hmap + boundaries).max()

    seeds = np.logical_and(foreground > foreground_threshold, dist > seed_distance)
    # seeds = parallel.label(seeds, block_shape=block_shape, verbose=True, connectivity=1)
    seeds = label(seeds, connectivity=2)
    seeds = apply_size_filter(seeds, min_size, verbose=True, block_shape=block_shape)

    # mask = (foreground + boundaries) > 0.5
    mask = (foreground + np.where(boundaries < boundary_threshold, boundaries, 0)) > 0.5  # take overlap
    # mask = foreground > foreground_threshold
    # mask = np.logical_or((foreground > foreground_threshold), (boundary > boundary_threshold))  # (boundaries > (1-boundary_threshold)))

    seg = np.zeros_like(seeds)
    seg = parallel.seeded_watershed(
        hmap, seeds, block_shape=block_shape, out=seg, verbose=True, halo=halo, mask=mask
    )
    seg = apply_size_filter(seg, min_size, verbose=True, block_shape=block_shape)
    seg = _postprocess_seg_3d(seg, area_threshold=area_threshold, iterations=post_iter, iterations_3d=post_iter3d)

    return {
        "segmentation": seg.astype(np.uint8),
        "seeds": seeds.astype(np.uint8),
        "dist": dist.astype(np.float32),
        "hmap": hmap.astype(np.float32),
        "mask": mask.astype(np.uint8),
    }


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


from itertools import product

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
        return segmentation # Nothing to filter

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

class ZarrChannelWrapper:
    """Lazily extracts a single channel from a multi-channel Zarr array."""
    def __init__(self, zarr_array, channel):
        self.zarr_array = zarr_array
        self.channel = channel
        self.shape = zarr_array.shape[1:]
        self.ndim = len(self.shape)
        self.dtype = zarr_array.dtype

    def __getitem__(self, key):
        # When a block is requested (e.g., key = (slice(0, 128), slice(0, 256), ...))
        # we prepend the channel index so we only load that specific block for this channel.
        if isinstance(key, tuple):
            return self.zarr_array[(self.channel,) + key]
        else:
            return self.zarr_array[(self.channel, key)]


def segment_mitos_ooc_wrapped(
    pred, out_dir, min_size=250,
    verbose=True,
    block_shape=(128, 256, 256),
    halo=(48, 48, 48),
    seed_distance=6,
    boundary_threshold=0.25,
    foreground_threshold=0.5,
    area_threshold=5000,   # kept for signature compatibility (not used here)
    reuse_computed=False,
    bg_penalty=2.0,        # height-map penalty for barrier voxels; lower (e.g. 1.2) reduces fragmentation
    n_threads=8,           # cap threads to match SLURM CPU allocation; multiprocessing.cpu_count() can return full node count
):
    # pred is (C,Z,Y,X)
    shape = pred.shape[1:]
    fg, bd = ZarrChannelWrapper(pred, 0), ZarrChannelWrapper(pred, 1)

    store = zarr.DirectoryStore(out_dir)
    root = zarr.group(store=store)  #, overwrite=not reuse_computed)

    def needs(name):
        return (not reuse_computed) or (name not in root)

    # --- dist (persisted) ---
    dist = root.get("dist")

    if dist is None or needs("dist"):
        dist = root.require_dataset("dist", shape=shape, chunks=block_shape, dtype=np.float32)
        if verbose: print("Computing dist (distance transform)...")
        t0 = time.time()

        # virtual boolean boundaries-threshold volume (no dataset written)
        boundaries_thresh = wrapper.SimpleTransformationWrapper(
            bd, lambda x: x < boundary_threshold
        )

        parallel.distance_transform(
            boundaries_thresh, halo=halo, verbose=verbose,
            block_shape=block_shape, distances=dist, n_threads=n_threads,
        )
        if verbose: print("dist in", time.time() - t0, "s")
    elif verbose:
        print("Reusing existing dist...")

    # --- seeds (persisted labeled CC) ---
    seeds = root.get("seeds")

    if seeds is None or needs("seeds"):
        seeds = root.require_dataset("seeds", shape=shape, chunks=block_shape, dtype=np.uint32)
        if verbose: print("Computing seeds (mask) + CC labeling...")
        t0 = time.time()

        # virtual seed mask (no dataset written)
        seed_mask = MultiTransformationWrapper(
            lambda f, d: np.logical_and(f > foreground_threshold, d > seed_distance),
            fg, dist
        )

        # label writes out-of-core into `seeds`
        parallel.label(seed_mask, block_shape=block_shape, verbose=verbose, out=seeds, n_threads=n_threads)
        if verbose: print("seeds in", time.time() - t0, "s")
    elif verbose:
        print("Reusing existing seeds...")

    # --- compute dist_max (blockwise reduction) ---
    if verbose: print("Computing dist_max...")
    t0 = time.time()
    dist_max = float(max(np.max(dist[bb]) for bb in iterate_blocks(shape, block_shape)))
    if verbose: print("dist_max =", dist_max, "in", time.time() - t0, "s")

    # --- hmap + mask as virtual volumes (no datasets written) ---
    def hmap_tf(d_chunk, index):
        # index is the spatial block slice tuple (z,y,x)
        f_chunk = fg[index]
        b_chunk = bd[index]
        h = (dist_max - d_chunk) / dist_max
        bg = np.logical_and(b_chunk > boundary_threshold, f_chunk < boundary_threshold)
        h[bg] = bg_penalty
        return h.astype(np.float32, copy=False)

    hmap = wrapper.TransformationWrapper(dist, hmap_tf)

    mask = MultiTransformationWrapper(
        lambda f, b: (f + b) > 0.5,
        fg, bd
    )

    # --- watershed (persisted) ---
    seg = root.get("seg")

    if seg is None or needs("seg"):
        seg = root.require_dataset("seg", shape=shape, chunks=block_shape, dtype=np.uint32)
        if verbose: print("Computing watershed...")
        t0 = time.time()

        parallel.seeded_watershed(
            hmap, seeds,
            block_shape=block_shape, halo=halo,
            out=seg, mask=mask,
            verbose=verbose, n_threads=n_threads,
        )
        if verbose: print("watershed in", time.time() - t0, "s")

        if verbose: print("Applying size filter...")
        seg = apply_size_filter_ooc_optim(seg, min_size, verbose=verbose, block_shape=block_shape)
    elif verbose:
        print("Reusing existing segmentation...")

    return {"segmentation": seg}



def segment_mitos_ooc_optimized(pred, out_dir, min_size=250,
    verbose=True,
    block_shape=(128, 256, 256),
    halo=(48, 48, 48),
    seed_distance=6,
    boundary_threshold=0.25,
    area_threshold=5000,
    reuse_computed=False,
):
    # WARNING: This approach uses dataset existence as the proxy for completion!!
    # FIX: pred is (c, z, y, x). We need (z, y, x) for the out-of-core shapes.
    shape = pred.shape[1:] 
    
    # Initialize out-of-core storage (e.g., Zarr directory store)
    store = zarr.DirectoryStore(out_dir)
    # Only wipe the directory if we explicitly don't want to reuse data
    root = zarr.group(store=store, overwrite=not reuse_computed)
    
    # Helper function to check if a step needs to be computed
    def needs_compute(name):
        return (not reuse_computed) or (name not in root)

    # 1. Blockwise Thresholding
    compute_thresh = needs_compute('boundaries_thresh')
    boundaries_thresh = root.require_dataset('boundaries_thresh', shape=shape, chunks=block_shape, dtype=bool)
    if compute_thresh:
        if verbose: print("Computing boundaries_thresh...")
        for bb in iterate_blocks(shape, block_shape):
            boundaries_thresh[bb] = pred[(1,) + bb] < boundary_threshold
    elif verbose:
        print("Reusing existing boundaries_thresh...")

    # 1.5 Distance Transform
    compute_dist = needs_compute('dist')
    dist = root.require_dataset('dist', shape=shape, chunks=block_shape, dtype=np.float32)
    if compute_dist:
        t0 = time.time()
        parallel.distance_transform(
            boundaries_thresh, halo=halo, verbose=verbose, block_shape=block_shape, distances=dist
        )
        if verbose: print("Compute distance transform in", time.time() - t0, "s")
    elif verbose:
        print("Reusing existing dist...")

    # 2. Blockwise Seed Generation
    compute_seeds = needs_compute('seeds')
    seeds = root.require_dataset('seeds', shape=shape, chunks=block_shape, dtype=np.uint32)
    if compute_seeds:
        if verbose: print("Computing seeds and connected components...")
        for bb in iterate_blocks(shape, block_shape):
            seeds[bb] = np.logical_and(pred[(0,) + bb] > 0.5, dist[bb] > seed_distance)

        t0 = time.time()
        parallel.label(seeds, block_shape=block_shape, verbose=verbose, out=seeds) 
        if verbose: print("Compute connected components in", time.time() - t0, "s")
    elif verbose:
        print("Reusing existing seeds...")

    # 3. Blockwise HMAP and Mask creation
    compute_hmap_mask = needs_compute('hmap') or needs_compute('mask')
    hmap = root.require_dataset('hmap', shape=shape, chunks=block_shape, dtype=np.float32)
    mask = root.require_dataset('mask', shape=shape, chunks=block_shape, dtype=bool)
    if compute_hmap_mask:
        if verbose: print("Computing hmap and mask...")
        t0 = time.time()
        dist_max = np.max([np.max(dist[bb]) for bb in iterate_blocks(shape, block_shape)]) 
        
        bg_penalty = 2.0

        for bb in iterate_blocks(shape, block_shape):
            f_chunk = pred[(0,) + bb]
            b_chunk = pred[(1,) + bb]
            
            h_chunk = (dist_max - dist[bb]) / dist_max
            bg_mask = np.logical_and(b_chunk > boundary_threshold, f_chunk < boundary_threshold)
            h_chunk[bg_mask] = bg_penalty
            
            hmap[bb] = h_chunk
            mask[bb] = (f_chunk + b_chunk) > 0.5
        if verbose: print("Compute hmap and mask in", time.time() - t0, "s")
    elif verbose:
        print("Reusing existing hmap and mask...")

    # 4. Watershed
    compute_seg = needs_compute('seg')
    seg = root.require_dataset('seg', shape=shape, chunks=block_shape, dtype=np.uint32)
    if compute_seg:
        if verbose: print("Computing watershed...")
        t0 = time.time()
        seg = parallel.seeded_watershed(
            hmap, seeds, block_shape=block_shape,
            out=seg, mask=mask, verbose=verbose, halo=halo,
        )
        if verbose: print("Compute watershed in", time.time() - t0, "s")

        # Pass the out-of-core seg object to your filters
        # Note: Size filter is combined with watershed creation block so it doesn't run 
        # twice on an already-completed segmentation array
        if verbose: print("Applying size filter...")
        seg = apply_size_filter_ooc_optim(seg, min_size, verbose=verbose, block_shape=block_shape)
    elif verbose:
        print("Reusing existing segmentation...")
    
    return {"segmentation": seg}


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
        warnings.warn("img must be uint8, converting to uint8 from " + str(img.dtype))
        img = img.astype(np.uint8)
    if img.ndim not in (2, 3):
        raise ValueError(f"img must be a 2D or 3D array, but got {img.ndim}D")

    # Binary mask of white voxels
    white_mask = img == 255

    # Label connected components
    labeled = label(white_mask)
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


def get_file_paths(path, ext=".h5", reverse=False):
    ext = ext if ext.startswith(".") else f".{ext}"
    ext_l = ext.lower()

    # If `path` is a file, just check its extension (case-insensitive)
    if os.path.isfile(path):
        return [path] if path.lower().endswith(ext_l) else []

    # Otherwise search recursively and filter case-insensitively
    candidates = glob(os.path.join(path, "**", "*"), recursive=True)
    paths = [p for p in candidates if os.path.isfile(p) and p.lower().endswith(ext_l)]
    return sorted(paths, reverse=reverse)


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


def export_data(export_path: str, data, voxel_size=None):
    """Export data to the specified path, determining format from the file extension.
    
    Args:
        data (np.ndarray | dict): The data to save. For HDF5/Zarr, a dict of named datasets is required.
        export_path (str): The file path where the data should be saved.
        voxel_size (tuple | list | np.ndarray, optional): The voxel dimensions.
    
    Raises:
        ValueError: If the file format is unsupported or if data format does not match the expected type.
    """

    path, ext = os.path.split(export_path)
    ext = ext.lower().split(".")[-1]

    if ext == "tif":
        if isinstance(data, dict):
            for key, value in data.items():
                out_name = export_path.replace(".tif", f"_{key}.tif".replace("/", "_"))
                tifffile.imwrite(out_name, value, dtype=value.dtype, compression="zlib")
        elif not isinstance(data, np.ndarray):
            raise ValueError("For .tif format, data must be a NumPy array or a dict of named NumPy arrays.")

    elif ext in {"mrc", "rec"}:
        if not isinstance(data, np.ndarray):
            raise ValueError("For .mrc and .rec formats, data must be a NumPy array.")
        with mrcfile.new(export_path, overwrite=True) as mrc:
            mrc.set_data(data.astype(data.dtype))

    elif ext == "zarr":
        if not isinstance(data, dict):
            raise ValueError("For .zarr format, data must be a dictionary with dataset names as keys.")
        root = zarr.open(export_path, mode="w")
        for key, value in data.items():
            root.create_dataset(key, data=value.astype(data.dtype))

    elif ext in {"h5", "hdf5"}:
        if not isinstance(data, dict):
            raise ValueError("For .h5 and .hdf5 formats, data must be a dictionary with dataset names as keys.")
        
        with h5py.File(export_path, "w") as f:
            if voxel_size is not None:
                voxel_size_array = voxel_size if isinstance(voxel_size, np.ndarray) else np.array(voxel_size, dtype=np.float32)
                f.attrs.create(name='voxel_size', data=voxel_size_array)
                # Add explicit indication of the dimension order
                f.attrs.create(name='voxel_size_order', data='z, y, x')
            
            for key, value in data.items():
                ds = f.create_dataset(name=key, data=value, dtype=value.dtype, compression="gzip")
                
                if "raw" in key and voxel_size is not None:
                    ds.attrs.create(name='voxel_size', data=voxel_size_array)
                    # Add it to the dataset attributes as well
                    ds.attrs.create(name='voxel_size_order', data='z, y, x')
    
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    print(f"Data successfully exported to {export_path}")


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
        labeled_mask, num_features = label(mask)

        if num_features == 1:  # If there's only one connected component, keep it
            cleaned_seg[mask] = lbl
            continue

        # Compute sizes of each component
        # sizes = sum_labels(np.array(mask, dtype=np.uint8), np.array(labeled_mask, dtype=np.uint8), index=np.arange(1, num_features + 1))
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
    labeled_array, num_features = label(segmentation)
    
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
    seg = filter_segmentation(seg)
    seg = filter_small_objects(seg, min_size)
    seg = _postprocess_seg_3d(seg, area_threshold=area_threshold, iterations=4, iterations_3d=8)
    return seg


def find_label_file(raw_path: str, label_paths: list) -> str:
    """
    Find the corresponding label file for a given raw file.

    Args:
        raw_path (str): The path to the raw file.
        label_paths (list): A list of label file paths.

    Returns:
        str: The path to the matching label file, or None if no match is found.
    """
    raw_base = os.path.splitext(os.path.basename(raw_path))[0]  # Remove extension
    # if "raw" in raw_base:
    #     raw_base = raw_base.replace("_raw", "")

    for label_path in label_paths:
        label_base = os.path.splitext(os.path.basename(label_path))[0]  # Remove extension
        if raw_base in label_base:  # Ensure raw name is contained in label name
            return label_path

    return None  # No match found


def export_to_h5(data, export_path):
    with h5py.File(export_path, mode='a') as h5f:
        for key in data.keys():
            if key in h5f:
                print(f"Skipping {key} as it already exists in {export_path}")
                continue
            h5f.create_dataset(key, data=data[key], compression="gzip")
    print("exported to", export_path)


def _extract_zdim_and_save_h5(data, save_dir, start_z, end_z, prefix="cropped"):
    """
    Crops mitochondria and raw data based on given z-dimension range, then saves them to HDF5.

    Parameters:
    - data (dict): Dictionary with labeled "mitochondria" data and "raw" image data.
    - save_dir (str): Directory to save the extracted regions.
    - start_z (int): Starting slice of the z-dimension.
    - end_z (int): Ending slice of the z-dimension.
    - prefix (str): Prefix for saved files.

    Returns:
    - None (saves files to `save_dir`)
    """
    os.makedirs(save_dir, exist_ok=True)
    export_data = {}

    if "raw" not in data:
        raise ValueError("No 'raw' key found in the dataset.")

    for key, value in data.items():

        # Crop the mitochondria and raw data along the given z range
        cropped = value[start_z:end_z]  # Cropped 
        #raw_cropped = data["raw"][start_z:end_z]  # Corresponding raw data

        # Downscale y and x dimensions
        max_y, max_x = cropped.shape[1], cropped.shape[2]
        cropped = cropped[:, :max_y, :max_x]
        #raw_cropped = raw_cropped[:, :max_y, :max_x]

        # Prepare data for export
        export_data[key] = cropped

    # Save to HDF5
    export_path = os.path.join(save_dir, f"{prefix}_z{start_z}-{end_z}-mito.h5")
    export_to_h5(export_data, export_path)


def export_mrc(filename: str, data: np.ndarray, voxel_size: tuple[float, float, float]):
    """
    Export a 3D NumPy array to an .mrc file with a specified voxel size.

    Args:
        filename (str): Output .mrc file path.
        data (np.ndarray): 3D NumPy array (Z, Y, X).
        voxel_size (tuple[float, float, float]): Voxel size in (Z, Y, X) order.
    """
    assert data.ndim == 3, "Input data must be a 3D array (Z, Y, X)."
    assert len(voxel_size) == 3, "Voxel size must be a tuple of (Z, Y, X)."

    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))  # Ensure data is in correct dtype
        mrc.voxel_size = voxel_size  # Set voxel size metadata
        mrc.update_header_from_data()  # Ensure header consistency
        mrc.header.d = np.array(voxel_size, dtype=np.float32)  # Alternative metadata setting
        mrc.flush()  # Write changes to disk

    print(f"Saved {filename} with voxel size {voxel_size}")


def standardize_channel(raw, channel=0):

    if raw.ndim != 4:
        raise ValueError(f"Expected a 4D input (C, Z, Y, X), got shape {raw.shape}")

    if not (0 <= channel < raw.shape[0]):
        raise ValueError(f"Invalid channel index {channel}, must be in range [0, {raw.shape[0]-1}]")

    raw_norm = np.float32(raw)
    raw_norm[channel] = torch_em.transform.raw.standardize(raw[channel])

    return raw_norm


class MitoStateMaskTransform:
    """Joint (raw, label) transform for cristae training.

    After `label_transform` has produced labels of shape [n_ch, D, H, W], this
    appends n_ch mask channels encoding voxels where loss should be computed.
    Voxels where `raw[mito_channel] == exclude_state_value` are masked out
    (mask=0); all other voxels — including background — remain active (mask=1).

    This prevents the network from being penalised for predictions inside
    mitochondria that carry no cristae annotations (typically label 2), while
    still letting it learn the no-cristae signal from background voxels.

    The result has shape [2*n_ch, D, H, W], compatible with `MaskedDiceLoss`.
    The mito-state channel is left as integer-valued floats by
    `standardize_channel` (which only normalises channel 0), so the equality
    check is safe.
    """

    def __init__(self, mito_channel: int = 1, exclude_state_value: float = 2.0):
        self.mito_channel = mito_channel
        self.exclude_state_value = exclude_state_value

    def __call__(self, raw: np.ndarray, labels: np.ndarray):
        mito_state = raw[self.mito_channel]                                        # [D, H, W]
        mask = (np.abs(mito_state - self.exclude_state_value) >= 0.5).astype(np.float32)  # 1 where NOT excluded
        masks = np.stack([mask] * labels.shape[0], axis=0)                         # [n_ch, D, H, W]
        labels = np.concatenate([labels, masks], axis=0)                           # [2*n_ch, D, H, W]
        return raw, labels


class MaskedDiceLoss(nn.Module):
    """Dice loss that reads a per-channel binary mask from the second half of
    the target tensor (as produced by `MitoStateMaskTransform`).

    target shape: [B, 2*n_ch, ...] – first n_ch channels are the actual
    targets, second n_ch channels are the spatial masks (1 = compute loss,
    0 = ignore).  Masking is applied via element-wise multiplication so that
    zeroed-out voxels contribute 0 to both the numerator and denominator of
    the Dice score.
    """

    def __init__(self, **dice_kwargs):
        super().__init__()
        self._dice = torch_em.loss.DiceLoss(**dice_kwargs)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_pred_ch = prediction.size(1)
        assert target.size(1) == 2 * n_pred_ch, (
            f"MaskedDiceLoss expects target with {2 * n_pred_ch} channels, got {target.size(1)}"
        )
        mask = target[:, n_pred_ch:]   # [B, n_ch, ...]
        target = target[:, :n_pred_ch] # [B, n_ch, ...]
        return self._dice(prediction * mask, target * mask)


def normalize_percentile_with_channel(raw, lower=1, upper=99, channel=0):
    """
    Normalize a specific channel of a multi-channel array using percentile normalization.

    Args:
        raw (np.ndarray): Input array of shape (C, Z, Y, X).
        lower (float): Lower percentile for normalization.
        upper (float): Upper percentile for normalization.
        channel (int, optional): The channel index to normalize. Defaults to 0.

    Returns:
        np.ndarray: Normalized array with shape (C, Z, Y, X).
    """
    if raw.ndim != 4:
        raise ValueError(f"Expected a 4D input (C, Z, Y, X), got shape {raw.shape}")

    if not (0 <= channel < raw.shape[0]):
        raise ValueError(f"Invalid channel index {channel}, must be in range [0, {raw.shape[0]-1}]")

    raw_norm = np.float32(raw)
    raw_norm[channel] = torch_em.transform.raw.normalize_percentile(raw[channel], lower=lower, upper=upper)

    return raw_norm


def get_all_datasets(file_path):
    dataset_names = []

    def visit_func(name, obj):
        if isinstance(obj, h5py.Dataset):
            dataset_names.append(name)

    with h5py.File(file_path, 'r') as hdf5_file:
        hdf5_file.visititems(visit_func)

    return dataset_names


def get_filename_and_inter_dirs(file_path, base_path):
    # Extract the base name (filename with extension)
    base_name = os.path.basename(file_path)
    # Split the base name into name and extension to get the filename
    file_name = os.path.splitext(base_name)[0]
    # Get the relative path of file_path from base_path
    relative_path = os.path.relpath(file_path, base_path)
    # Get the intermediate directories by removing the filename from the relative path
    inter_dirs = os.path.dirname(relative_path)
    return file_name, inter_dirs


def create_directories_if_not_exists(base_path, inter_dirs):
    # Construct the full path from base_path and inter_dirs
    full_path = os.path.join(base_path, inter_dirs)
    
    # Check if the path exists
    if not os.path.exists(full_path):
        # If it doesn't exist, create the directories
        os.makedirs(full_path)
        print(f"\nCreated directories: {full_path}")
    else:
        print(f"\nDirectories already exist: {full_path}")


def get_wichmann_data():
    data = [
        "mitos_and_cristae/Otof-KO_M6/KO8_eb2_model.h5",
        "mitos_and_cristae/Otof-KO_M6/KO9_eb11_model.h5",
        "mitos_and_cristae/Otof-KO_M6/KO9_eb13_model.h5",
        "mitos_and_cristae/Otof-KO_M6/KO9_eb4_model.h5",
        "mitos_and_cristae/Otof-KO_M6/KO9_eb6_model.h5",
        "mitos_and_cristae/Otof-KO_M6/KO9_eb9_model.h5",
        "mitos_and_cristae/Otof-KO_M6/M10_eb8_model.h5",
        "mitos_and_cristae/Otof-KO_P10/M1_eb8_model.h5",
        "mitos_and_cristae/Otof-KO_P10/M2_eb10_model.h5",
        "mitos_and_cristae/Otof-KO_P10/M2_eb1_model.h5",
        "mitos_and_cristae/Otof-KO_P10/M2_eb8_model.h5",
        "mitos_and_cristae/Otof-KO_P22/M5_eb3_model.h5",
        "mitos_and_cristae/Otof-KO_P22/M6_eb2_model.h5",
        "mitos_and_cristae/Otof-KO_P22/M7_eb15_model.h5",
        "mitos_and_cristae/Otof-WT_M6/WT40_eb10_model.h5",
        "mitos_and_cristae/Otof-WT_M6/WT40_eb3_model.h5",
        "mitos_and_cristae/Otof-WT_M6/WT40_eb8_model.h5",
        "mitos_and_cristae/Otof-WT_M6/WT41_eb4_model.h5",
        "mitos_and_cristae/Otof-WT_P10/WT13_syn1_model2.h5",
        "mitos_and_cristae/Otof-WT_P10/WT13_syn4_model2.h5",
        "mitos_in_endbuld/Otof_AVCN03_429C_WT_M.Stim_G3_3_model.h5",
        "mitos_in_endbuld/Otof_AVCN03_429C_WT_M.Stim_G3_4_model.h5",
        "mitos_in_endbuld/Otof_AVCN03_429C_WT_M.Stim_G3_5_model.h5",
        "mitos_in_endbuld/Otof_AVCN03_429D_WT_Rest_G3_4_model.h5",
        "mitos_in_endbuld/Otof_AVCN03_429D_WT_Rest_H5_1_35461_model.h5",
        "mitos_in_endbuld/Otof_AVCN03_429D_WT_Rest_H5_3_35461_model.h5",
        "mitos_in_endbuld/Otof_AVCN03_429D_WT_Rest_H5_4_35461_model.h5",
    ]
    for i in range(len(data)):
        # data[i] = "/scratch-grete/projects/nim00007/data/mitochondria/wichmann/extracted/" + data[i]
        data[i] = "/home/freckmann15/data/mitochondria/wichmann/extracted/" + data[i]
    return data


class CombinedDatasets(torch.utils.data.Dataset):
    def __init__(
        self,
        raw_path: Union[List[Any], str, os.PathLike],
        raw_key: str,
        raw2_key: str,
        label_path: Union[List[Any], str, os.PathLike],
        label_key: str,
        patch_shape: Tuple[int, ...],
        raw_transform=None,
        raw2_transform=None,
        label_transform=None,
        label_transform2=None,
        transform=None,
        roi: Optional[dict] = None,
        dtype: torch.dtype = torch.float32,
        label_dtype: torch.dtype = torch.float32,
        n_samples: Optional[int] = None,
        sampler=None,
        ndim: Optional[int] = None,
        with_channels: bool = False,
        with_label_channels: bool = False,
        with_padding: bool = True,
    ):
        self.ds1 = torch_em.data.SegmentationDataset(
            raw_path,
            raw_key,
            label_path,
            label_key,
            patch_shape=patch_shape,
            raw_transform=raw_transform,
            label_transform=label_transform,
            label_transform2=label_transform2,
            transform=transform,
            roi=roi,
            n_samples=n_samples,
            sampler=sampler,
            ndim=ndim,
            dtype=dtype,
            label_dtype=label_dtype,
            with_channels=with_channels,
            with_label_channels=with_label_channels,
            with_padding=with_padding,
        )
        # Additional raw data key for the second dataset
        self.ds2 = torch_em.data.SegmentationDataset(
            raw_path,
            raw2_key,
            label_path,
            label_key,
            patch_shape=patch_shape,
            raw_transform=raw2_transform,
            transform=transform,
            roi=roi,
            n_samples=n_samples,
            sampler=sampler,
            ndim=ndim,
            dtype=dtype,
            label_dtype=label_dtype,
            with_channels=with_channels,
            with_label_channels=with_label_channels,
            with_padding=with_padding,
        )

    def __len__(self):
        return len(self.ds1)

    def __getitem__(self, index):
        data1 = self.ds1.super().__getitem__(index)
        raw1, labels = data1[0], data1[1]

        data2 = self.ds2.super().__getitem__(index)
        raw2 = data2[0]

        # Return the combined result (raw1, raw2, labels)
        return (raw1, raw2), labels


# not in use atm
def get_loaders(
        data, patch_shape, ndim=3, batch_size=1, n_workers=16, 
        label_transform=None, with_channels=True, with_label_channels=True, 
        rois_dict=None):
    """
    Generates data loaders for training and validation using the given data, patch shape, and other parameters.

    Args:
        data (dict): A dictionary containing the paths to the training and validation data.
        patch_shape (tuple): The shape of the patches to be extracted from the data.
        ndim (int, optional): The number of dimensions of the data. Defaults to 3.
        batch_size (int, optional): The batch size for the data loaders. Defaults to 1.
        n_workers (int, optional): The number of workers for data loading. Defaults to 16.
        label_transform (callable, optional): A callable that transforms the labels. Defaults to None.
        with_channels (bool, optional): Whether to include the channels in the data. Defaults to True.
        with_label_channels (bool, optional): Whether to include the label channels in the data. Defaults to True.
        rois_dict (dict, optional): A dictionary containing the regions of interest (ROIs) for training and validation. Defaults to None.

    Returns:
        tuple: A tuple containing the training data loader and the validation data loader.
    """
    if rois_dict is not None:
        train_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key="raw",
            label_paths=data["train"], label_key="labels/mitochondria",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            rois=rois_dict["train"]
        )
        val_loader = torch_em.default_segmentation_loader(
            raw_paths=data["val"], raw_key="raw",
            label_paths=data["val"], label_key="labels/mitochondria",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
            rois=rois_dict["val"]
        )
    else:
        train_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key="raw",
            label_paths=data["train"], label_key="labels/mitochondria",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
        )
        val_loader = torch_em.default_segmentation_loader(
            raw_paths=data["train"], raw_key="raw",
            label_paths=data["val"], label_key="labels/mitochondria",
            patch_shape=patch_shape, ndim=ndim, batch_size=batch_size,
            label_transform=label_transform, num_workers=n_workers,
            with_channels=with_channels, with_label_channels=with_label_channels,
        )
    
    return train_loader, val_loader


def remove_prefix_from_keys(state_dict, prefix="_orig_mod."):
    """
    Removes the specified prefix from the beginning of all keys in a dictionary.

    Args:
        state_dict (dict): The dictionary containing keys with the prefix to remove.
        prefix (str): The string prefix to remove from the beginning of keys.

    Returns:
        dict: A new dictionary with the prefix removed from all keys.
    """
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            # Remove the prefix and store the value in the new dictionary with the modified key
            new_key = key[len(prefix):]
            filtered_state_dict[new_key] = value
        else:
            # If the key doesn't start with the prefix, keep it as is
            filtered_state_dict[key] = value
    return filtered_state_dict


def get_rois_coordinates_skimage(file, label_key, min_shape, euler_threshold=None, min_amount_pixels=None):
    """
    Calculates the average coordinates for each unique label in a 3D label image using skimage.regionprops.

    Args:
        file (h5py.File): Handle to the open HDF5 file.
        label_key (str): Key for the label data within the HDF5 file.
        min_shape (tuple): A tuple representing the minimum size for each dimension of the ROI.

    Returns:
        dict: A dictionary mapping unique labels to lists of average coordinates
            for each dimension, or None if no labels are found.
    """

    label_data = file[label_key]
    label_shape = label_data.shape

    # Ensure data type is suitable for regionprops (usually uint labels)
    # if label_data.dtype != np.uint:
    #     label_data = label_data.astype(np.uint).value

    # Find connected regions (objects) using regionprops
    regions = regionprops(label_data)

    # Check if any regions were found
    if not regions:
        return None

    label_extents = {}
    for region in regions:
        if euler_threshold is not None:
            if region.euler_number != euler_threshold:
                continue
        if min_amount_pixels is not None:
            if region["area"] < min_amount_pixels:
                continue
        
        # # Extract relevant information for ROI calculation
        label = region.label  # Get the label value
        min_coords = region.bbox[:3]  # Minimum coordinates (excluding intensity channel)
        max_coords = region.bbox[3:6]  # Maximum coordinates (excluding intensity channel)

        # Clip coordinates and create ROI extent (similar to previous approach)
        clipped_min_coords = np.clip(min_coords, 0, label_shape[0] - min_shape[0])
        clipped_max_coords = np.clip(max_coords, min_shape[1], label_shape[1])
        roi_extent = tuple(slice(min_val, min_val + min_shape[dim]) for dim, (min_val, max_val) in enumerate(zip(clipped_min_coords, clipped_max_coords)))

        # Check for labels within the ROI extent (new part)
        roi_data = file[label_key][roi_extent]
        amount_label_pixels = np.count_nonzero(roi_data)
        if amount_label_pixels < 100:  # Check for any non-zero values (labels)
            continue  # Skip this ROI if no labels present
        if min_amount_pixels is not None:
            if amount_label_pixels < min_amount_pixels:
                continue

        label_extents[label] = roi_extent

    return label_extents


def _norm_pattern(pat: str) -> str:
    """
    Turn any of the following into a proper glob pattern:
        "tif"   → "*.tif"
        ".tif"  → "*.tif"
        "*.tif" → "*.tif"
    """
    pat = pat.strip()
    if pat.startswith("*"):
        return pat          # already a glob
    if pat.startswith("."):
        pat = pat[1:]       # drop leading dot
    return f"*.{pat}"       # add the leading "*."


def get_data_paths(data_dir, data_format="*.h5"):
    data_format = _norm_pattern(data_format)
    # check if data_dir is file
    if os.path.isfile(data_dir) or fnmatch.fnmatch(data_dir, data_format):
        return [data_dir]
    data_paths = glob(os.path.join(data_dir, "**", data_format), recursive=True)
    return data_paths


def get_data_paths_and_rois(data_dir, min_shape,
                            data_format="*.h5",
                            image_key="raw",
                            label_key_mito="labels/mitochondria",
                            label_key_cristae="labels/cristae",
                            with_thresholds=True):
    """
    Retrieves all HDF5 data paths, their corresponding image and label data keys,
    and extracts Regions of Interest (ROIs) for labels.

    Args:
        data_dir (str): Path to the directory containing HDF5 files.
        data_format (str, optional): File format to search for (default: "*.h5").
        image_key (str, optional): Key for image data within the HDF5 file (default: "raw").
        label_key_mito (str, optional): Key for the first label data (default: "labels/mitochondria").
        label_key_cristae (str, optional): Key for the second label data (default: "labels/cristae").
        roi_halo (tuple, optional): A fixed tuple representing the halo radius for ROIs in each dimension (default: (2, 3, 1)).

    Returns:
        tuple: A tuple containing three lists:
            - data_paths: List of paths to all HDF5 files in the directory and subdirectories.
            - rois_list: List containing ROIs for each valid HDF5 file.
                - Each ROI is a list of tuples representing slices for each dimension.
    """

    data_paths = glob(os.path.join(data_dir, "**", data_format), recursive=True)
    rois_list = []
    new_data_paths = [] # one data path for each ROI

    for data_path in data_paths:
        try:
            # Open the HDF5 file in read-only mode
            with h5py.File(data_path, "r") as f:
                # Check for existence of image and label datasets (considering key flexibility)
                if image_key not in f:
                    print(f"Warning: Key(s) missing in {data_path}. Skipping {image_key}")
                    continue

                #label_data_mito = f[label_key_mito][()] if label_key_mito is not None else None

                # Extract ROIs (assuming ndim of label data is the same as image data)
                if with_thresholds:
                    rois = get_rois_coordinates_skimage(f, label_key_mito, min_shape, min_amount_pixels=100) # euler_threshold=1,
                else:
                    rois = get_rois_coordinates_skimage(f, label_key_mito, min_shape, euler_threshold=None, min_amount_pixels=None)
                for label_id, roi in rois.items():
                    rois_list.append(roi)
                    new_data_paths.append(data_path)
        except OSError:
            print(f"Error accessing file: {data_path}. Skipping...")

    return new_data_paths, rois_list

def split_data_paths_to_dict_with_ensure(data_paths, ensure_strings=None, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
    """
    Splits data paths into training, validation, and testing sets without shuffling.
    Ensures that at least one file is present in the validation split for each string
    in ensure_strings (if provided).

    Args:
        data_paths (list): List of paths to all files.
        ensure_strings (tuple/list of str, optional): Strings that must be present in 
            at least one file path in the validation split. Substrings of file paths are matched.
        train_ratio (float, optional): Proportion of data for training (0.0-1.0) (default: 0.8).
        val_ratio (float, optional): Proportion of data for validation (0.0-1.0) (default: 0.1).
        test_ratio (float, optional): Proportion of data for testing (0.0-1.0) (default: 0.1).

    Returns:
        dict: Dictionary containing "train", "val", and "test" keys with data paths.

    Raises:
        ValueError: If the sum of ratios exceeds 1 or if ensure_strings is provided but
                   validation set would be empty or doesn't contain required strings.
    """
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Sum of train, validation, and test ratios must equal 1.0.")
    
    num_data = len(data_paths)
    
    # Validate ensure_strings parameter
    if ensure_strings is not None:
        if not isinstance(ensure_strings, (tuple, list)):
            raise ValueError("ensure_strings must be a tuple or list of strings")
        if not all(isinstance(s, str) for s in ensure_strings):
            raise ValueError("All elements in ensure_strings must be strings")
    
    train_size = int(num_data * train_ratio)
    val_size = int(num_data * val_ratio)
    test_size = int(num_data * test_ratio)
    remaining = num_data - (train_size + val_size + test_size)
    if remaining > 0:
        train_size += remaining

    # Split data paths
    data_split = {
        "train": data_paths[:train_size],
        "val": data_paths[train_size:train_size+val_size],
        "test": data_paths[train_size+val_size:]
    }
    
    # Ensure validation set contains at least one file for each required string
    if ensure_strings is not None:
        # Check if validation set already satisfies the requirement
        val_files = data_split["val"]
        strings_found = []
        
        for string in ensure_strings:
            found_in_val = any(string in file_path for file_path in val_files)
            if found_in_val:
                strings_found.append(string)
        
        # If not all strings are found, we need to adjust the splits
        if len(strings_found) < len(ensure_strings):
            # Find files that contain the missing strings
            missing_strings = [s for s in ensure_strings if s not in strings_found]
            
            # Try to move files from test set to validation set
            test_files = data_split["test"]
            files_to_move = []
            
            for string in missing_strings:
                for file_path in test_files:
                    if string in file_path and file_path not in files_to_move:
                        files_to_move.append(file_path)
                        break
            
            # If still not enough files, try moving from train set
            if len(files_to_move) < len(missing_strings):
                train_files = data_split["train"]
                for string in missing_strings:
                    if len(files_to_move) >= len(missing_strings):
                        break
                    for file_path in train_files:
                        if string in file_path and file_path not in files_to_move:
                            files_to_move.append(file_path)
                            break
            
            # If we still don't have enough files, raise an error
            if len(files_to_move) < len(missing_strings):
                raise ValueError(f"Cannot ensure at least one file containing each of the strings "
                               f"{ensure_strings} in validation set. Consider adjusting ratios.")
            
            # Move files from test/train to validation
            for file_path in files_to_move:
                # Remove from original set
                if file_path in test_files:
                    test_files.remove(file_path)
                    data_split["test"] = test_files
                elif file_path in data_split["train"]:
                    data_split["train"].remove(file_path)
                
                # Add to validation set
                data_split["val"].append(file_path)
            
            # Re-sort to maintain order
            data_split["val"] = sorted(data_split["val"])
    
    return data_split


def split_data_paths_to_dict(data_paths, rois_list, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0):
    """
    Splits data paths and ROIs into training, validation, and testing sets without shuffling.

    Args:
        data_paths (list): List of paths to all HDF5 files.
        rois_dict (dict): Dictionary mapping data paths (or indices) to corresponding ROIs.
        train_ratio (float, optional): Proportion of data for training (0.0-1.0) (default: 0.8).
        val_ratio (float, optional): Proportion of data for validation (0.0-1.0) (default: 0.1).
        test_ratio (float, optional): Proportion of data for testing (0.0-1.0) (default: 0.1).

    Returns:
        tuple: A tuple containing two dictionaries:
            - data_split: Dictionary containing "train", "val", and "test" keys with data paths.
            - rois_split: Dictionary containing "train", "val", and "test" keys with corresponding ROIs.

    Raises:
        ValueError: If the sum of ratios exceeds 1 or the length of data paths and number of ROIs don't match.
    """
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Sum of train, validation, and test ratios must equal 1.0.")
    num_data = len(data_paths)
    if rois_list is not None:
        if len(rois_list) != num_data:
            raise ValueError(f"Length of data paths and number of ROIs in the dictionary must match: len rois {len(rois_list)}, len data_paths {len(data_paths)}")

    train_size = int(num_data * train_ratio)
    val_size = int(num_data * val_ratio)
    test_size = int(num_data * test_ratio)
    remaining = num_data - (train_size + val_size + test_size)
    if remaining > 0:
        train_size += remaining

    data_split = {
        "train": data_paths[:train_size],
        "val": data_paths[train_size:train_size+val_size],
        "test": data_paths[train_size+val_size:]
    }

    if rois_list is not None:
        rois_split = {
            "train": rois_list[:train_size],
            "val": rois_list[train_size:train_size+val_size],
            "test": rois_list[train_size+val_size:]
        }

        return data_split, rois_split
    else:
        return data_split


def get_filename_from_path(path):
    """
    Extracts the filename from a given path by splitting on the last '/'.

    Args:
        path: The path string.

    Returns:
        The filename portion of the path.
    """
    return path.split("/")[-1]


def split_data_paths(data_paths, key_dicts, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    """
    Splits data paths and key information into training, validation, and testing sets.

    Args:
        data_paths (list): List of paths to all HDF5 files.
        key_dicts (list): List of dictionaries containing image and label data keys for each HDF5 file.
        train_ratio (float, optional): Proportion of data for training (0.0-1.0) (default: 0.8).
        val_ratio (float, optional): Proportion of data for validation (0.0-1.0) (default: 0.1).
        test_ratio (float, optional): Proportion of data for testing (0.0-1.0) (default: 0.1).
        seed (int, optional): Random seed for shuffling data paths (default: None).

    Returns:
        tuple: A tuple containing three dictionaries:
            - train_data: Dictionary containing "data_paths" and "key_dicts" for training data.
            - val_data: Dictionary containing "data_paths" and "key_dicts" for validation data (if applicable).
            - test_data: Dictionary containing "data_paths" and "key_dicts" for testing data.

    Raises:
        ValueError: If the sum of ratios exceeds 1.
    """

    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Sum of train, validation, and test ratios must equal 1.0.")

    if seed is not None:
        random.seed(seed)
    random.shuffle(data_paths)
    random.shuffle(key_dicts)

    num_data = len(data_paths)
    train_size = int(num_data * train_ratio)
    val_size = int(num_data * val_ratio)  # Optional validation set
    test_size = num_data - train_size - val_size

    train_data = {
        "data_paths": data_paths[:train_size],
        "key_dicts": key_dicts[:train_size]
    }
    val_data = {"data_paths": [], "key_dicts": []}  # Optional validation set
    test_data = {
        "data_paths": data_paths[train_size+val_size:],
        "key_dicts": key_dicts[train_size+val_size:]
    }

    if val_size > 0:
        val_data = {
            "data_paths": data_paths[train_size:train_size+val_size],
            "key_dicts": key_dicts[train_size:train_size+val_size]
        }

    return train_data, val_data, test_data


def get_data_paths_and_keys(data_dir, data_format="*.h5", image_key="raw", label_key="labels/mitochondria"):
    """
    Retrieves all HDF5 data paths and their corresponding image and label data keys.

    Args:
        data_dir (str): Path to the directory containing HDF5 files.
        data_format (str, optional): File format to search for (default: "*.h5").
        image_key (str, optional): Key for image data within the HDF5 file (default: "raw").
        label_key (str, optional): Key for label data within the HDF5 file (default: "labels/mitochondria").

    Returns:
        tuple: A tuple containing two lists:
            - data_paths: List of paths to all HDF5 files in the directory and subdirectories.
            - key_dicts: List of dictionaries containing image and label data keys for each HDF5 file.
                - Each dictionary has keys: "image_key" and "label_key".
    """

    data_paths = glob(os.path.join(data_dir, "**", data_format), recursive=True)
    key_dicts = []

    for data_path in data_paths:
        try:
            # Open the HDF5 file in read-only mode
            with h5py.File(data_path, "r") as f:
                # Check for existence of image and label datasets (considering key flexibility)
                if image_key not in f or (label_key is not None and label_key not in f):
                    print(f"Warning: Key(s) missing in {data_path}. Skipping...")
                    continue
                key_dicts.append({"image_key": image_key, "label_key": label_key})
        except OSError:
            print(f"Error accessing file: {data_path}. Skipping...")

    return data_paths, key_dicts


def create_directory(directory):
    """
    Creates a directory if it doesn't already exist.

    Args:
        directory (str): The path to the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def visualize_data_napari(data):
    """
    Visualizes the 3D raw data and all labels using napari.

    Args:
        data (dict): Dictionary containing loaded raw data ("raw" key) 
                    and labels ("labels" dictionary with loaded labels).
    """
    # Create a napari viewer
    viewer = napari.Viewer()
    
    if "raw" in data.keys():
        if isinstance(data["raw"], torch.Tensor):
            raw_data = data["raw"].cpu().detach().numpy()
        else:
            raw_data = data["raw"]

        viewer.add_image(raw_data, name="Raw")

    if "label" in data.keys():
        if isinstance(data["label"], torch.Tensor):
            label_data = data["label"].cpu().detach().numpy()
        else:
            label_data = data["label"]

        viewer.add_labels(label_data.astype(int), name="Label")  # Ensure labels are integers
    if "label2" in data.keys():
        if isinstance(data["label"], torch.Tensor):
            label_data = data["label2"].cpu().detach().numpy()
        else:
            label_data = data["label2"]

        viewer.add_labels(label_data.astype(int), name="Label2") 
    if "pred1" in data.keys():
        if isinstance(data["pred1"], torch.Tensor):
            label_data = data["pred1"].cpu().detach().numpy()
        else:
            label_data = data["pred1"]

        viewer.add_image(label_data.astype(float), blending="additive", name="Foreground Prediction")
    if "pred2" in data.keys():
        if isinstance(data["pred2"], torch.Tensor):
            label_data = data["pred2"].cpu().detach().numpy()
        else:
            label_data = data["pred2"]

        viewer.add_image(label_data.astype(float), blending="additive", name="Boundary Prediction")

    # Show the napari viewer
    napari.run()


def run_prediction(data, model, block_shape=[32, 512, 512], halo=[8, 32, 32]):
    """
    Run a prediction using a trained model on the given data.

    Args:
        data (array-like): The input data on which predictions are to be made.
        model (torch.nn.Module): The loaded model.
        block_shape (List[int], optional): The block shape to use for prediction.
            Defaults to [32, 256, 256].
        halo (List[int], optional): The halo shape to use for prediction.
            Defaults to [8, 32, 32].

    Returns:
        array-like: The predicted output from the model.
    """

    gpu_ids = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    with torch.no_grad():
        pred = predict_with_halo(
            data, model, gpu_ids=gpu_ids,
            block_shape=block_shape, halo=halo,
            preprocess=None,
        )
    return pred


def get_label_transform(label_data):
    """
    Transforms all the label ids in a label image to ones.
    Args:
        label_data (np.ndarray): A 3D array representing the label data.
            - Assumed to use integer values to represent unique labels (adjust as needed).

    Returns:
        np.ndarray: A 3D array with all label ids replaced by ones.
    """
    return np.where(label_data != 0, 1, label_data)


def get_loss_function(loss_name, affinities=False, **kwargs):
    loss_names = ["bce", "ce", "dice"]
    if isinstance(loss_name, str):
        assert loss_name in loss_names, f"{loss_name}, {loss_names}"
        if loss_name == "dice":
            loss_function = torch_em.loss.DiceLoss(**kwargs)
        elif loss_name == "ce":
            loss_function = nn.CrossEntropyLoss()
        elif loss_name == "bce":
            loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = loss_name

    # we need to add a loss wrapper for affinities
    if affinities:
        loss_function = torch_em.loss.LossWrapper(
            loss_function, transform=torch_em.loss.ApplyAndRemoveMask()
        )
    return loss_function


def get_all_metadata(data_dir, data_format="*.h5"):
    """
    Retrieves metadata for all HDF5 files in a directory and its subdirectories.

    Args:
        data_dir (str): Path to the directory containing HDF5 files.
        data_format (str, optional): File format to search for (default: "*.h5").

    Returns:
        list: A list of dictionaries containing the retrieved metadata for 
              each HDF5 file, or None if errors encountered.
    """

    # Get all file paths matching the format
    data_paths = glob(os.path.join(data_dir, "**", data_format), recursive=True)
    metadata_list = []

    # Loop through all files and get metadata
    for data_path in tqdm(data_paths):
        metadata = get_data_metadata(data_path)
        if metadata:
            metadata_list.append(metadata)

    return metadata_list


def get_data_metadata(data_path):
    """
    Retrieves metadata about the data in an HDF5 file without loading it entirely.

    Args:
        data_path (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary with filename as key and nested dictionary containing 
              the retrieved metadata, or None if error.
    """

    try:
        # Open the HDF5 file in read-only mode
        with h5py.File(data_path, "r") as f:

            # Check for existence of datasets
            if "raw" not in f:
                print(f"Error: 'raw' dataset not found in {data_path}")
                return None  # Indicate error

            # Get image size (assuming 'raw' dataset holds the 3D image)
            image_size = f["raw"].shape

            # Check for labels group
            has_cristae_label = False
            if "labels" in f:
                labels_group = f["labels"]
                has_cristae_label = "cristae" in labels_group.keys()

            # Calculate exact min and max values using NumPy functions
            min_value = np.amin(f["raw"], axis=None)
            max_value = np.amax(f["raw"], axis=None)

            # Calculate basic statistics
            average_value = np.mean(f["raw"])  # Calculate mean across all axes
            try:
                std_dev = np.std(f["raw"])  # Calculate standard deviation
            except RuntimeWarning:
                # Handle potential runtime warnings (e.g., all elements equal)
                std_dev = None

            # Extract filename from data_path
            filename = os.path.basename(data_path)  # Get filename from path

            # Create nested dictionary for metadata
            metadata = {
                "image_size": list(image_size),
                "has_cristae_label": has_cristae_label,  # Boolean
                "value_range": (float(min_value), float(max_value)),
                "average_value": float(average_value),
                "std_dev": float(std_dev) if std_dev is not None else None,
            }
            print(metadata)

            # Return dictionary with filename as key and metadata nested
            return {filename: metadata}

    except Exception as e:
        print(f"Error getting metadata for {data_path}: {e}")
        return None


def load_metadata(data_path):
    """
    Loads metadata from a single YAML file in a directory.

    Args:
        data_path (str): Path to the directory containing the metadata.yaml file.

    Returns:
        dict: The loaded metadata from the YAML file, 
            or None if no file found or parsing error.
    """

    # Construct the expected filename for metadata
    filename = os.path.join(data_path, "metadata.yaml")

    # Check if the file exists
    if not os.path.isfile(filename):
        print(f"Error: No metadata.yaml file found in {data_path}")
        return None

    # Load metadata from the file using YAML (safe_load)
    try:
        with open(filename, 'r') as f:
            metadata = yaml.full_load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {filename}: {e}")

    average_values = []
    with_cristae_labels = []
    images_sizes = []
    std_devs = []
    value_ranges = []
    for item in metadata:
        for name, data_dict in item.items():
            for key, value in data_dict.items():
                if key == "average_value":
                    average_values.append(value)
                if key == "has_cristae_label":
                    with_cristae_labels.append(value)
                if key == "image_size":
                    images_sizes.append(value)
                if key == "std_dev":
                    std_devs.append(value)
                if key == "value_range":
                    value_ranges.append(value)
        #print(len(average_values) , len(with_cristae_labels) , len(images_sizes) , len(std_devs) , (value_ranges))
    # Assuming average_values and std_devs contain numerical data
    average_average_value = sum(average_values) / len(average_values)
    average_std_dev = sum(std_devs) / len(std_devs)
    average_image_sizes = []
    for dim in range(len(images_sizes[0])):  # Assuming all images have the same number of dimensions
        # Calculate the average for each dimension
        dimension_values = [image_size[dim] for image_size in images_sizes]
        average_image_sizes.append(sum(dimension_values) / len(dimension_values))

    min_values = [range_tuple[0] for range_tuple in value_ranges]  # Extract minimum values
    max_values = [range_tuple[1] for range_tuple in value_ranges]  # Extract maximum values

    average_min_value = sum(min_values) / len(min_values)
    average_max_value = sum(max_values) / len(max_values)

    print("Average minimum value:", average_min_value)
    print("Average maximum value:", average_max_value)

    # Print the calculated averages
    print("Average Average Value:", average_average_value)
    print("Average Std Dev:", average_std_dev)
    print("Images with cristae labels:", sum(with_cristae_labels), "/", len(with_cristae_labels)) # sum(label for label in with_cristae_labels if label)
    print("Average Image Sizes:", average_image_sizes)
    # Return None if any exceptions occur
    return metadata