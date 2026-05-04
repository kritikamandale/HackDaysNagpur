"""
transforms.py — Albumentations augmentation pipelines
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: list, cfg: dict) -> A.Compose:
    """
    Returns training augmentation pipeline.
    Augmentations are applied to both image and mask identically.
    """
    aug = cfg.get("augmentation", {})
    h, w = image_size

    transforms = [
        # 1. Resize first — everything works on a fixed size
        A.Resize(height=h, width=w, always_apply=True),
    ]

    if aug.get("use_aug", True):
        transforms += [
            # Geometric
            A.HorizontalFlip(p=aug.get("hflip_prob", 0.5)),
            A.VerticalFlip(p=aug.get("vflip_prob", 0.1)),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1,
                rotate_limit=aug.get("rotate_limit", 15),
                border_mode=0, p=0.4
            ),
            A.RandomCrop(height=h, width=w, p=0.3),
            A.Resize(height=h, width=w, always_apply=True),  # restore after crop

            # Photometric — image only (mask not affected)
            A.RandomBrightnessContrast(
                brightness_limit=aug.get("brightness_limit", 0.2),
                contrast_limit=aug.get("contrast_limit", 0.2),
                p=0.5
            ),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                 val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=aug.get("blur_prob", 0.1)),
            A.GaussNoise(var_limit=(5, 25), p=0.2),

            # Structural
            A.GridDistortion(p=aug.get("grid_distortion_prob", 0.2)),
            A.CoarseDropout(
                max_holes=4, max_height=32, max_width=32,
                fill_value=0, p=0.1
            ),
        ]

    transforms += [
        # Normalize with ImageNet stats — matches pretrained encoder
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),   # (H, W, C) → (C, H, W) float32 tensor
    ]

    return A.Compose(transforms)


def get_val_transforms(image_size: list) -> A.Compose:
    """
    Validation / test pipeline — resize + normalize only, no random ops.
    """
    h, w = image_size
    return A.Compose([
        A.Resize(height=h, width=w, always_apply=True),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_test_transforms(image_size: list) -> A.Compose:
    """
    Test pipeline — no mask, just image.
    """
    return get_val_transforms(image_size)
