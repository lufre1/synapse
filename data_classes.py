import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.transforms import v2, RandomCrop
import numpy as np


class CustomDataset(Dataset):
    # Here we pass the parameters for creating the dataset:
    # The image data, the labels and the patch shape (= the size of the image patches used for training).
    # mask_transform is a function that is applied only to the label data, in order to convert the cell segmentation
    # we have as labels, which cannot be used for directly training the network, into a different representation
    # transform is an additonal argument that can be used for defining data augmentations (optional exercise)
    def __init__(self, images, labels, patch_shape, mask_transform=None, transform=None, label_aware_crop=None):
        self.images = images
        self.labels = labels
        self.patch_shape = patch_shape
        self.transform = transform
        self.mask_transform = mask_transform
        self.label_aware_crop = label_aware_crop

    def __len__(self):
        return len(self.images)

    # The __getitem__ method returns the image data and labels for a given sample index.
    def __getitem__(self, index):

        # get the current image and mask (= cell segmentation)
        image = self.images[index]
        mask = self.labels[index]
        assert image.ndim == mask.ndim == 3
        assert image.shape == mask.shape

        if self.label_aware_crop:
            cropped_image, cropped_mask = self.label_aware_crop(image, mask, self.patch_shape)
        else:
            # Use default cropping logic if label_aware_crop is not provided
            i, j, k, h, w, d = F.get_params(torch.tensor(image), self.patch_shape)
            cropped_image = F.crop(torch.tensor(image), i, j, k, h, w, d)
            cropped_mask = F.crop(torch.tensor(mask), i, j, k, h, w, d)
        # make sure to add the channel dimension to the image
        image, mask = np.array(cropped_image), np.array(cropped_mask)
        if image.ndim == 3:
            image = image[None]

        # Apply transform if it is present.
        if self.transform:
            image, mask = self.transform(image, mask)

        # Apply specific transform for the mask.
        mask = self.mask_transform(mask)

        return image, mask
