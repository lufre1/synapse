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
                    train_instance_segmentation_only=True
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
