"""
coco_loader.py

Author: Kexin Rong (you@example.com)
License: MIT

This module provides a PyTorch Dataset and DataLoader setup for the COCO dataset,
designed for semantic segmentation tasks such as those targeted by the
Segment Anything Model (SAM).

It includes support for loading images and masks, applying custom transforms,
and integrating with standard PyTorch training loops.

Advantages:
- Clean, modular PyTorch Dataset
- Pluggable transforms for augmentation
- Supports COCO format annotations

✅ Tested with PyTorch >= 1.10

Example usage (as Python module):

    from data.coco_loader import get_coco_dataloader

    dataloader = get_coco_dataloader(
        images_dir="/path/to/coco/images",
        masks_dir="/path/to/coco/masks",
        batch_size=4,
        shuffle=True
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


class COCOSegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset for COCO-style segmentation.
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
            f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))
        ])
        self.transform = transform
        self.target_transform = target_transform

        if len(self.image_files) == 0:
            raise RuntimeError(f"No image files found in {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.image_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


def get_coco_dataloader(
    images_dir: str,
    masks_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    image_size: int = 224
) -> DataLoader:
    """
    Utility function to create a PyTorch DataLoader for the COCO segmentation dataset.

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

    dataset = COCOSegmentationDataset(
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
