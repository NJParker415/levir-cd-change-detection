"""
Dataset class for loading and preprocessing images and masks for training and validation.
"""

# Imports
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Imagenet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_augmentation() -> A.Compose:
    """Data augmentation"""
    return A.Compose(
        [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
        ],
        additional_targets={'image_B': 'image', 'mask': 'mask'},
    )

def get_color_augmentation() -> A.Compose:
    """Color augmentation"""
    return A.Compose(
        [
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
        ],
    )

def get_eval_augmentation() -> A.Compose:
    """Data augmentation for evaluation"""
    return A.Compose(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ],
        additional_targets={'image_B': 'image', 'mask': 'mask'},
    )

def precrop_dataset(
    src_dir: str,
    dst_dir: str,
    patch_size: int = 256,
    splits: Tuple[str, ...] = ('train', 'val', 'test'),
) -> None:
    """Pre-crop dataset into patches"""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    for split in splits:
        for subfolder in ['A', 'B', 'label']:
            in_folder = src_dir / split / subfolder
            out_folder = dst_dir / split / subfolder
            out_folder.mkdir(parents=True, exist_ok=True)

            if not in_folder.exists():
                print(f"Warning: {in_folder} does not exist. Skipping.")
                continue

            image_files = sorted(in_folder.glob('*.png'))  # Assuming images are in PNG format
            print(f"Processing {len(image_files)} images from {split}/{subfolder}")

            for img_path in image_files:
                img = Image.open(img_path)
                w, h = img.size
                stem = img_path.stem

                for i in range(0, h, patch_size):
                    for j in range(0, w, patch_size):
                        patch = img.crop((j, i, j + patch_size, i + patch_size))
                        patch_name = f"{stem}_{i}_{j}.png"
                        patch.save(out_folder / patch_name)

        print(f"Patches saved to {dst_dir}")

class LEVIRCDDataset(Dataset):
    """LEVIR-CD dataset for change detection"""

    def __init__(self, root:str, augment: bool = False):
        self.root = Path(root)
        self.augment = augment

        # Assuming all splits have the same filenames
        self.filenames = sorted(os.listdir(self.root / 'A'))  

        # Sanity Check
        b_files = set(os.listdir(self.root / 'B'))
        label_files = set(os.listdir(self.root / 'label'))
        for fn in self.filenames:
            assert fn in b_files, f"Missing B: {fn}"
            assert fn in label_files, f"Missing label: {fn}"

        self.spatial_aug = (
            get_train_augmentation() if augment else get_eval_augmentation()
        )
        self.color_aug = get_color_augmentation() if augment else None

        print(f"Loaded {len(self.filenames)} samples from {root} with augment={augment}")

    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx: int) -> dict:
        """Load and preprocess a sample"""
        fn = self.filenames[idx]

        img_a = np.array(Image.open(self.root / 'A' / fn).convert('RGB'))
        img_b = np.array(Image.open(self.root / 'B' / fn).convert('RGB'))
        mask = np.array(Image.open(self.root / 'label' / fn).convert('L'))

        # Levir-CD uses 0/255
        mask = (mask > 128).astype(np.float32)

        # Independent color augmentation
        if self.color_aug is not None:
            img_a = self.color_aug(image=img_a)["image"]
            img_b = self.color_aug(image=img_b)["image"]

        # Spatial augmentation
        augmented = self.spatial_aug(image=img_a, image_B=img_b, mask=mask)
        img_a = augmented['image']
        img_b = augmented['image_B']
        mask = augmented['mask']

        # Add channel dim
        if isinstance(mask, torch.Tensor):
            mask = mask.unsqueeze(0)
        else:
            mask = torch.tensor(mask).unsqueeze(0)

        return {
            "A": img_a,
            "B": img_b,
            "mask": mask,
            "name": fn,
        }
    
def build_dataloaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
) -> dict:
    """Build dataloaders for train, val, and test splits"""
    loaders = {}
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist. Skipping {split} loader.")
            continue

        is_train = split == 'train'
        ds = LEVIRCDDataset(root=split_path, augment=is_train)
        loaders[split] = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
        )

    return loaders