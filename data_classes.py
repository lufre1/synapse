import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import numpy as np

class CustomDataset(Dataset):
    # Here we pass the parameters for creating the dataset:
    # The image data, the labels and the patch shape (= the size of the image patches used for training).
    # mask_transform is a function that is applied only to the label data, in order to convert the cell segmentation
    # we have as labels, which cannot be used for directly training the network, into a different representation
    # transform is an additonal argument that can be used for defining data augmentations (optional exercise)
    def __init__(self, images, labels, patch_shape, mask_transform=None, transform=None):
        self.images = images
        self.labels = labels
        self.patch_shape = patch_shape
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    # The __getitem__ method returns the image data and labels for a given sample index.
    def __getitem__(self, index):

        # get the current image and mask (= cell segmentation)
        image = self.images[index]
        mask = self.labels[index]
        assert image.ndim == mask.ndim == 2
        assert image.shape == mask.shape

        # Extract the patches for training from the image and label data
        # Random crop same excerpt from image and mask
        i, j, h, w = v2.RandomCrop.get_params(
            torch.tensor(image), output_size=self.patch_shape
        )
        image = v2.functional.crop(torch.tensor(image), i, j, h, w)
        mask = v2.functional.crop(torch.tensor(mask), i, j, h, w)

        # make sure to add the channel dimension to the image
        image, mask = np.array(image), np.array(mask)
        if image.ndim == 2:
            image = image[None]

        # Apply transform if it is present.
        if self.transform:
            image, mask = self.transform(image, mask)         

        # Apply specific transform for the mask.
        mask = self.mask_transform(mask)

        return image, mask
