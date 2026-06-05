from glob import glob
import os
from typing import Optional, Tuple
import imageio
from elf.io import open_file
import numpy as np
from tqdm import tqdm
import micro_sam.util as util


def get_decoder_outputs(
    predictor,
    segmentor,
    volume: np.ndarray,
    embedding_path: Optional[str] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    batch_size: int = 1,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run prediction with a microSAM decoder, to get the boundary-, center-distance
    and foreground predictions.

    Args:
        predictor:
        segmentor:
        volume:
        embedding_path:
        tile_shape:
        halo:
        batch_size:
        verbose:

    Returns:
        The foreground predictions.
        The center distance predictions.
        The boundary distance predictions.
    """
    image_embeddings = util.precompute_image_embeddings(
        predictor=predictor,
        input_=volume,
        save_path=embedding_path,
        ndim=3,
        tile_shape=tile_shape,
        halo=halo,
        verbose=verbose,
        batch_size=batch_size,
    )

    foreground = np.zeros(volume.shape, dtype="float32")
    center_dists = np.zeros(volume.shape, dtype="float32")
    boundary_dists = np.zeros(volume.shape, dtype="float32")
    # This could also be batched.
    for i in tqdm(range(volume.shape[0]), desc="Segment slices", disable=not verbose):
        segmentor.initialize(volume[i], image_embeddings=image_embeddings, verbose=False, i=i)
        foreground[i] = segmentor._foreground
        center_dists[i] = segmentor._center_distances
        boundary_dists[i] = segmentor._boundary_distances

    return foreground, center_dists, boundary_dists


def raw_transform(x):
    x = x.astype("float32")
    x -= x.min()
    x /= x.max()
    x *= 255
    return x.astype("uint8")


def finetune_sam_v2(name, train_images, raw_key, label_key,
                    val_images, batch_size,
                    n_iterations, checkpoint_path,
                    save_root, patch_shape, check,
                    early_stopping,
                    model_type="vit_b",
                    label_transform=None, sampler=None,
                    n_samples=None,
                    min_size=None,
                    train_instance_segmentation_only=False
                    ):
    from micro_sam.training import train_sam_for_configuration, default_sam_loader

    # train_images = os.path.join(ROOT, "SERIAL-uPSTEM800_37373_G-H_JOINED_2Kb1dawbp_cropped.h5")
    # train_labels = os.path.join(ROOT, "mitos-corrected.tif")

    roi_train, roi_val = None, None

    train_loader = default_sam_loader(
        raw_paths=train_images, raw_key=raw_key,
        label_paths=train_images, label_key=label_key,
        patch_shape=patch_shape, with_segmentation_decoder=True, with_channels=False,
        batch_size=batch_size, rois=roi_train, raw_transform=raw_transform,
        label_transform=label_transform, min_size=min_size,
        sampler=sampler, n_samples=n_samples,
        train_instance_segmentation_only=train_instance_segmentation_only,
        is_multi_tensor=False
    )
    val_loader = default_sam_loader(
        raw_paths=val_images, raw_key=raw_key,
        label_paths=val_images, label_key=label_key,
        patch_shape=patch_shape, with_segmentation_decoder=True, with_channels=False,
        batch_size=batch_size, rois=roi_val, raw_transform=raw_transform,
        label_transform=label_transform, min_size=min_size,
        sampler=sampler, n_samples=n_samples,
        train_instance_segmentation_only=train_instance_segmentation_only,
        is_multi_tensor=False
    )
    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, 50)
        check_loader(val_loader, 50)

    train_sam_for_configuration(
        name, train_loader, val_loader, model_type=model_type,
        save_root=save_root, checkpoint_path=checkpoint_path,
        early_stopping=early_stopping, n_iterations=n_iterations,
        verify_n_labels_in_loader=None,
        train_instance_segmentation_only=train_instance_segmentation_only  # this disables training of without training the model parts for interactive segmentation,
        #  i.e. without training the prompt encoder and mask decoder
    )


# ---------------------------------------------------------------------------
# SAM-3D input transforms (pad/normalize raw, binarize/pad labels)
# ---------------------------------------------------------------------------
class LabelTrafoToBinary:
    """Binarize a label volume (any non-zero id -> 1)."""

    def __call__(self, labels):
        labels = (labels > 0).astype(labels.dtype)
        return labels


class RawTrafoFor3dInputs:
    """Normalize a raw volume to ``[0, 255]`` and stack it into 3 channels for SAM."""

    def _normalize_inputs(self, raw):
        from torch_em.transform.raw import normalize

        raw = normalize(raw)
        raw = raw * 255
        return raw

    def _set_channels_for_inputs(self, raw):
        raw = np.stack([raw] * 3, axis=0)
        return raw

    def __call__(self, raw):
        raw = self._normalize_inputs(raw)
        raw = self._set_channels_for_inputs(raw)
        return raw


class RawResizeTrafoFor3dInputs(RawTrafoFor3dInputs):
    """Like :class:`RawTrafoFor3dInputs` but pads the raw volume to ``desired_shape``."""

    def __init__(self, desired_shape, padding="constant"):
        super().__init__()
        self.desired_shape = desired_shape
        self.padding = padding

    def __call__(self, raw):
        from math import ceil, floor

        raw = self._normalize_inputs(raw)

        # let's pad the inputs
        tmp_ddim = (
            self.desired_shape[0] - raw.shape[0],
            self.desired_shape[1] - raw.shape[1],
            self.desired_shape[2] - raw.shape[2],
        )
        ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2, tmp_ddim[2] / 2)
        raw = np.pad(
            raw,
            pad_width=(
                (ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1])), (ceil(ddim[2]), floor(ddim[2]))
            ),
            mode=self.padding,
        )

        raw = self._set_channels_for_inputs(raw)

        return raw


class LabelResizeTrafoFor3dInputs:
    """Binarize labels (float32) and pad them to ``desired_shape``."""

    def __init__(self, desired_shape, padding="constant"):
        self.desired_shape = desired_shape
        self.padding = padding

    def __call__(self, labels):
        from math import ceil, floor

        # binarize the samples
        labels = (labels > 0).astype("float32")

        # let's pad the labels
        tmp_ddim = (
            self.desired_shape[0] - labels.shape[0],
            self.desired_shape[1] - labels.shape[1],
            self.desired_shape[2] - labels.shape[2],
        )
        ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2, tmp_ddim[2] / 2)
        labels = np.pad(
            labels,
            pad_width=(
                (ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1])), (ceil(ddim[2]), floor(ddim[2]))
            ),
            mode=self.padding,
        )

        return labels


# ---------------------------------------------------------------------------
# Volumetric segmentation from microSAM decoder outputs
# ---------------------------------------------------------------------------
def _default_seed_function(
    center_distances,
    boundary_distances,
    fg_mask,
    tile_shape,
    n_threads,
    verbose,
    **kwargs,
):
    import elf.parallel as parallel

    seed_map = boundary_distances < kwargs.get("boundary_distance_threshold", 0.5)
    if center_distances is not None:
        seed_map = np.logical_and(seed_map, center_distances < kwargs.get("center_distance_threshold", 0.5))
    if fg_mask is not None:
        seed_map[~fg_mask] = 0

    seeds = np.zeros(seed_map.shape, dtype="uint64")
    seeds = parallel.label(
        seed_map, out=seeds, block_shape=tile_shape, n_threads=n_threads, verbose=verbose,
    )
    return seeds


def _volumetric_segmentation_impl(
    center_distances,
    boundary_distances,
    foreground,
    gap_closing=0,
    min_z_extent=0,
    verbose=False,
    **kwargs,
):
    import multiprocessing as mp

    import elf.parallel as parallel
    from elf.parallel.filters import apply_filter
    from scipy.ndimage import binary_closing
    from micro_sam.multi_dimensional_segmentation import _filter_z_extent

    tile_shape = (64, 512, 512)
    halo = (8, 32, 32)
    n_threads = mp.cpu_count()

    distance_smoothing = kwargs.get("distance_smoothing", 1.6)
    if center_distances is not None:
        center_distances = apply_filter(
            center_distances, "gaussianSmoothing", sigma=distance_smoothing,
            block_shape=tile_shape, n_threads=n_threads
        )

    boundary_distances = apply_filter(
        boundary_distances, "gaussianSmoothing", sigma=distance_smoothing,
        block_shape=tile_shape, n_threads=n_threads
    )

    if foreground is None:
        fg_mask = None
    else:
        fg_mask = foreground > kwargs.get("foreground_threshold", 0.5)

    seeds = _default_seed_function(
        center_distances, boundary_distances, fg_mask, tile_shape, n_threads, verbose, **kwargs,
    )

    seg = np.zeros_like(seeds, dtype="uint64")
    seg = parallel.seeded_watershed(
        boundary_distances, seeds=seeds, out=seg, block_shape=tile_shape,
        halo=halo, n_threads=n_threads, verbose=verbose, mask=fg_mask,
    )

    segmentation = np.zeros_like(seg, dtype="uint64")
    segmentation = parallel.size_filter(
        seg, out=segmentation, min_size=kwargs.get("min_size", 0),
        block_shape=tile_shape, n_threads=n_threads, verbose=verbose
    )

    # Apply post-processing.
    if gap_closing is not None and gap_closing > 0:
        mask = segmentation > 0
        mask = np.logical_or(mask, binary_closing(mask, iterations=gap_closing))
        segmentation_ = np.zeros_like(segmentation)
        segmentation_ = parallel.seeded_watershed(
            boundary_distances, seeds=segmentation, mask=mask,
            block_shape=tile_shape, halo=halo, n_threads=mp.cpu_count(),
            verbose=False, out=segmentation_
        )
        segmentation = segmentation_

    if min_z_extent is not None and min_z_extent > 0:
        segmentation = _filter_z_extent(segmentation, min_z_extent)

    return segmentation


def volumetric_segmentation(
    foreground,
    center_dists,
    boundary_dists,
    gap_closing: int = 0,
    min_z_extent: int = 0,
    verbose: bool = False,
    use_foreground_mask: bool = True,
    use_center_distances: bool = True,
    **kwargs,
) -> np.ndarray:
    """Run volumetric segmentation based on outputs from a microSAM segmentation decoder.

    Args:
        foreground: Foreground probability map.
        center_dists: Center-distance map (set to None via ``use_center_distances=False``).
        boundary_dists: Boundary-distance map.
        gap_closing: Number of binary-closing iterations for gap closing (0 disables).
        min_z_extent: Minimum z-extent of objects to keep (0 disables).
        verbose: Whether to print progress.
        use_foreground_mask: Whether to restrict to the foreground mask.
        use_center_distances: Whether to use the center distances for seeding.

    Returns:
        The volumetric segmentation.
    """
    if not use_foreground_mask:
        foreground = None
    if not use_center_distances:
        center_dists = None
    segmentation = _volumetric_segmentation_impl(
        center_dists, boundary_dists, foreground, gap_closing=gap_closing, min_z_extent=min_z_extent, **kwargs,
    )
    return segmentation


def alt_segmentation(foreground, boundary_distances, verbose=False, **kwargs):
    """Alternative segmentation algorithm based on foreground and boundary distances only.

    Args:
        foreground: Foreground probability map.
        boundary_distances: Boundary distance map.
        verbose: Verbosity flag.
        **kwargs: Threshold overrides (``foreground_threshold``, ``seed_threshold``,
            ``boundary_distance_threshold``, ``min_size``).

    Returns:
        The segmentation label array.
    """
    import multiprocessing as mp

    import elf.parallel as parallel

    tile_shape = (64, 512, 512)
    halo = (8, 32, 32)
    n_threads = mp.cpu_count()

    if foreground is None:
        fg_mask = None
    else:
        fg_mask = foreground > kwargs.get("foreground_threshold", 0.5)

    seeds = np.zeros_like(foreground, dtype=int)
    mask = np.logical_and(foreground > kwargs.get("foreground_threshold", 0.5), foreground - boundary_distances >
                          kwargs.get("seed_threshold", 0.2))
    seeds[mask] = True
    seed_map = boundary_distances < kwargs.get("boundary_distance_threshold", 0.5)
    if fg_mask is not None:
        seed_map[~fg_mask] = 0

    seeds = parallel.label(
        seed_map, out=seeds, block_shape=tile_shape, n_threads=n_threads, verbose=verbose,
    )

    seg = np.zeros_like(seeds, dtype="uint64")
    seg = parallel.seeded_watershed(
        boundary_distances, seeds=seeds, out=seg, block_shape=tile_shape,
        halo=halo, n_threads=n_threads, verbose=verbose, mask=fg_mask,
    )

    segmentation = np.zeros_like(seg, dtype="uint64")
    segmentation = parallel.size_filter(
        seg, out=segmentation, min_size=kwargs.get("min_size", 0),
        block_shape=tile_shape, n_threads=n_threads, verbose=verbose
    )

    return segmentation


def run_decoder_prediction(data, model_type, checkpoint, use_tiling=True):
    """Predict microSAM decoder outputs (foreground + center/boundary distances).

    Returns a dict with keys ``prediction/foreground``, ``prediction/center_dists``
    and ``prediction/boundary_dists``.
    """
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter

    if use_tiling:
        tile_shape, halo = (192, 192), (32, 32)
    else:
        tile_shape, halo = None, None

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, checkpoint=checkpoint, is_tiled=use_tiling
    )

    foreground, center_dists, boundary_dists = get_decoder_outputs(
        predictor, segmenter, data, tile_shape=tile_shape, halo=halo, batch_size=4, verbose=True,
    )

    return {
        "prediction/foreground": foreground,
        "prediction/center_dists": center_dists,
        "prediction/boundary_dists": boundary_dists,
    }
