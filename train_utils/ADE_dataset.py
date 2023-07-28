import os

from PIL import Image
import numpy as np

import torch
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset
from torch.utils.data import random_split


def train_val_split(dataset, val_size=0.1):

    """
    Splits dataset into train and validation subsets

    Parameters
    ----------

    dataset: torch.utils.data.Dataset 
        Dataset to split

    val_size: float
        A number between 0 and 1 indicating the proportion of validation in the dataset

    Returns
    -------
    tuple
        Tuple of train subset and validation subset
    """

    total_samples = len(dataset)

    n_train_samples = int((1 - val_size) * total_samples)
    n_val_samples = total_samples - n_train_samples
    train_dataset, val_dataset = random_split(dataset, [n_train_samples, n_val_samples])

    return (train_dataset, val_dataset)


class ADE_Dataset(Dataset):

    """
    Dataset class for ADE segmentation dataset

    Parameters
    ----------

    root_dir: str
        Path to dataset

    train: bool
        Whether to use train or test part

    resize: int
        Number indicating what size the image will be resized to
    """

    def __init__(self, root_dir, train=True, resize=512):

        if train:
            split = 'training'
        else:
            split = 'validation'

        self.resize = resize

        mask_folder = os.path.join(root_dir, 'annotations', split)
        image_folder = os.path.join(root_dir, 'images', split)
        filenames = os.listdir(image_folder)

        self.data = []
        for filename in filenames:
            sample = (
                os.path.join(image_folder, filename),
                os.path.join(mask_folder, filename[:-3] + 'png')
            )
            self.data.append(sample)


        self.transforms = T.Compose(
            [
                T.ToTensor(),
                T.Resize((self.resize, self.resize), antialias=True),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
            ]
        )

    def load_image(self, index):
        """Returns image and mask by index in PIL.Image format"""
        image_path, mask_path = self.data[index]
        return Image.open(image_path), Image.open(mask_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, mask = self.load_image(index)

        img = img.convert("RGB")

        img = self.transforms(img)
        mask = np.array(mask)[np.newaxis, :, :]
        mask.flags.writeable = True
        mask = torch.as_tensor(mask)
        mask = F.resize(mask, (self.resize, self.resize), interpolation=T.InterpolationMode.NEAREST)

        return img, mask
