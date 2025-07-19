"""
custom_loader.py

Author: Your Name (you@example.com)
License: MIT

This module provides a PyTorch Dataset and DataLoader setup for custom segmentation
datasets, such as medical images or satellite images.

It is designed for easy integration with the Segment Anything Model (SAM) project,
supporting images and masks stored in parallel folders with matching filenames.

Advantages:
- Clean, modular PyTorch Dataset
- Pluggable transforms for augmentation
- Adaptable for medical, satellite, or other domain-specific segmentation data
- Supports resizing, normalization, custom transforms

✅ Tested with PyTorch >= 1.10

Example usage (as Python module):

    from data.custom_loader import get_custom_dataloader

    dataloader = get_custom_dataloader(
        images_dir="/path/to/images",
        masks_dir="/path/to/masks",
        batch_size=4,
        shuffle=True,
        image_size=256
    )

    for images, masks in dataloader:
        # Training loop
        ...
"""

import os
from typing import Callable, Optional
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import warnings


class CustomSegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for segmentation tasks with user-provided images and masks.
    Expects images and masks to be stored in separate directories with matching filenames.
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Args:
            images_dir (str): Path to directory with input images.
            masks_dir (str): Path to directory with corresponding masks.
            transform (callable, optional): Transform to apply to images.
            target_transform (callable, optional): Transform to apply to masks.
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted([
            f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.tif'))
        ])
        self.transform = transform
        self.target_transform = target_transform

        if len(self.image_files) == 0:
            raise RuntimeError(f"No image files found in {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.image_files[idx])

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file missing for image: {self.image_files[idx]}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale mask

        # Warning if image and mask sizes mismatch
        if image.size != mask.size:
            warnings.warn(f"Image and mask size mismatch: {image_path} vs {mask_path}")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask

    def show_sample(self, idx: int):
        """
        Debug utility to visualize an image and its mask.
        Only works in Jupyter/Colab or when matplotlib is available.
        """
        import matplotlib.pyplot as plt

        image, mask = self[idx]
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()

        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(image_np)
        axs[0].set_title("Image")
        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title("Mask")
        plt.tight_layout()
        plt.show()


def get_custom_dataloader(
    images_dir: str,
    masks_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    image_size: int = 224
) -> DataLoader:
    """
    Utility function to create a PyTorch DataLoader for custom segmentation datasets.

    Args:
        images_dir (str): Path to directory with images.
        masks_dir (str): Path to directory with masks.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of workers for DataLoader.
        image_size (int): Size to which images and masks will be resized.

    Returns:
        DataLoader: PyTorch DataLoader ready for training.
    """
    image_transforms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    mask_transforms = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    dataset = CustomSegmentationDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=image_transforms,
        target_transform=mask_transforms
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader
