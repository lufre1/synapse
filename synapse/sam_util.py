from glob import glob
import os
import imageio
from elf.io import open_file


# def extract_training_slices():
#     good_slices = [
#         9, 37, 103, 116, 192, 198
#     ]

#     train_im_output = os.path.join(ROOT, "train_mito", "images")
#     train_lab_output = os.path.join(ROOT, "train_mito", "labels")
#     os.makedirs(train_im_output, exist_ok=True)
#     os.makedirs(train_lab_output, exist_ok=True)

#     with open_file(CROP_MRC, "r") as f_raw, open_file(PRED, "r") as f_lab:
#         data = f_raw["data"]
#         labels = f_lab["segmentation/mitos_sam"]

#         for i, z in enumerate(good_slices):
#             im, lab = data[z], labels[z]
#             im = im.astype("float32")
#             im -= im.min()
#             im /= im.max()
#             im *= 255
#             im = im.astype("uint8")
#             imageio.imwrite(os.path.join(train_im_output, f"im-{i:02}.tif"), im, compression="zlib")
#             imageio.imwrite(os.path.join(train_lab_output, f"lab-{i:02}.tif"), lab, compression="zlib")


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
                    label_transform=None, sampler=None
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
        label_transform=label_transform,
        sampler=sampler
    )
    val_loader = default_sam_loader(
        raw_paths=val_images, raw_key=raw_key,
        label_paths=val_images, label_key=label_key,
        patch_shape=patch_shape, with_segmentation_decoder=True, with_channels=False,
        batch_size=batch_size, rois=roi_val, raw_transform=raw_transform,
        label_transform=label_transform,
        sampler=sampler
    )
    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, 5)
        check_loader(val_loader, 5)

    train_sam_for_configuration(
        name, train_loader, val_loader, model_type="vit_b",
        save_root=save_root, checkpoint_path=checkpoint_path,
        early_stopping=early_stopping, n_iterations=n_iterations
    )
