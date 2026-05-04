"""
dataset.py — PyTorch Dataset classes for offroad segmentation
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from src.utils import remap_mask


def _find_pairs(rgb_dir: str, mask_dir: str):
    rgb_dir  = Path(rgb_dir)
    mask_dir = Path(mask_dir)

    rgb_files  = sorted([p for p in rgb_dir.iterdir()
                         if p.suffix.lower() == '.png'])
    mask_files = sorted([p for p in mask_dir.iterdir()
                         if p.suffix.lower() == '.png'])

    print(f"  RGB  files found : {len(rgb_files)}")
    print(f"  Mask files found : {len(mask_files)}")

    if len(rgb_files) == 0 or len(mask_files) == 0:
        print(f"  [ERROR] No files found!")
        print(f"  RGB  dir : {rgb_dir.resolve()}")
        print(f"  Mask dir : {mask_dir.resolve()}")
        return []

    # Pair by sorted position — filenames don't match so we pair by order
    min_count  = min(len(rgb_files), len(mask_files))
    if len(rgb_files) != len(mask_files):
        print(f"  [WARN] Count mismatch — using first {min_count} pairs")

    pairs = list(zip(
        [str(p) for p in rgb_files[:min_count]],
        [str(p) for p in mask_files[:min_count]]
    ))
    print(f"  Paired samples   : {len(pairs)}")
    return pairs


class SegmentationDataset(Dataset):
    def __init__(self, rgb_dir: str, mask_dir: str, transform=None):
        self.pairs     = _find_pairs(rgb_dir, mask_dir)
        self.transform = transform

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No pairs found!\n"
                f"  rgb_dir  : {rgb_dir}\n"
                f"  mask_dir : {mask_dir}"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, mask_path = self.pairs[idx]

        image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw_mask is None:
            raise FileNotFoundError(f"Cannot read: {mask_path}")
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[:, :, 0]

        mask = remap_mask(raw_mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"].long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask  = torch.from_numpy(mask).long()

        return image, mask

    def get_raw_sample(self, idx):
        rgb_path, mask_path = self.pairs[idx]
        image    = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[:, :, 0]
        mask = remap_mask(raw_mask)
        return image, mask


class TestDataset(Dataset):
    def __init__(self, test_dir: str, transform=None):
        test_dir   = Path(test_dir)
        self.paths = sorted([
            str(p) for p in test_dir.iterdir()
            if p.suffix.lower() == '.png'
        ])
        self.transform = transform
        print(f"  Found {len(self.paths)} test images in {test_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        return image, Path(path).name


def get_dataloaders(cfg: dict, train_transform, val_transform):
    data_cfg  = cfg["data"]
    train_cfg = cfg["train"]

    train_dataset = SegmentationDataset(
        rgb_dir   = data_cfg["train_rgb"],
        mask_dir  = data_cfg["train_masks"],
        transform = train_transform,
    )
    val_dataset = SegmentationDataset(
        rgb_dir   = data_cfg["val_rgb"],
        mask_dir  = data_cfg["val_masks"],
        transform = val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = train_cfg["batch_size"],
        shuffle     = True,
        num_workers = 0,          # 0 is safer on Windows
        pin_memory  = False,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = train_cfg["batch_size"],
        shuffle     = False,
        num_workers = 0,          # 0 is safer on Windows
        pin_memory  = False,
    )

    print(f"\n  Train samples : {len(train_dataset)}")
    print(f"  Val samples   : {len(val_dataset)}")
    print(f"  Batch size    : {train_cfg['batch_size']}")

    return train_loader, val_loader