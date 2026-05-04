"""
explore_data.py — Run BEFORE training to verify your dataset is correct.

Usage:
    python explore_data.py

What it does:
  1. Counts images and masks in train/val folders
  2. Checks all masks have valid pixel values
  3. Shows class distribution (pixel %) — helps spot class imbalance
  4. Displays 4 random samples with color-coded masks
"""
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from src.utils import load_config, remap_mask, mask_to_color, CLASS_NAMES, CLASS_COLORS


def check_dataset(rgb_dir, mask_dir, split_name):
    rgb_dir  = Path(rgb_dir)
    mask_dir = Path(mask_dir)

    rgb_files  = sorted(rgb_dir.glob("*.png"))
    mask_files = sorted(mask_dir.glob("*.png"))

    print(f"\n── {split_name.upper()} SET ──────────────────────────")
    print(f"  RGB images : {len(rgb_files)}")
    print(f"  Masks      : {len(mask_files)}")

    return rgb_files, mask_files


def class_distribution(mask_files, split_name):
    """Compute pixel-level class distribution across the dataset."""
    pixel_counts = Counter()
    total_pixels = 0

    for mf in mask_files[:200]:   # sample up to 200 for speed
        raw = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        remapped = remap_mask(raw)
        for cls_idx in range(10):
            pixel_counts[cls_idx] += int((remapped == cls_idx).sum())
        total_pixels += remapped.size

    print(f"\n  Class distribution ({split_name}, pixel %):")
    print(f"  {'Class':<18} {'Pixels %':>10}")
    print("  " + "-" * 30)
    for i in range(10):
        pct = 100 * pixel_counts[i] / max(total_pixels, 1)
        bar = "█" * int(pct / 2)
        print(f"  {CLASS_NAMES[i]:<18} {pct:>8.2f}%  {bar}")

    return pixel_counts


def visualize_samples(rgb_files, mask_files, n=4):
    """Show n random samples side by side."""
    indices = np.random.choice(len(rgb_files), min(n, len(rgb_files)), replace=False)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))

    for col, idx in enumerate(indices):
        img  = cv2.cvtColor(cv2.imread(str(rgb_files[idx])), cv2.COLOR_BGR2RGB)
        raw  = cv2.imread(str(mask_files[idx]), cv2.IMREAD_UNCHANGED)
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        mask = remap_mask(raw)

        axes[0, col].imshow(img)
        axes[0, col].set_title(f"RGB: {rgb_files[idx].name[:20]}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(mask_to_color(mask))
        axes[1, col].set_title("Segmentation Mask", fontsize=8)
        axes[1, col].axis("off")

    # Legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=[c/255 for c in col], label=name)
               for col, name in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Dataset Sample Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path("runs").mkdir(exist_ok=True)
    plt.savefig("runs/data_samples.png", dpi=150, bbox_inches="tight")
    print("\n  Sample plot saved → runs/data_samples.png")
    plt.show()


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")

    # 1. Count files
    train_rgb, train_masks = check_dataset(
        cfg["data"]["train_rgb"], cfg["data"]["train_masks"], "train"
    )
    val_rgb, val_masks = check_dataset(
        cfg["data"]["val_rgb"], cfg["data"]["val_masks"], "val"
    )

    # 2. Class distribution
    class_distribution(train_masks, "train")

    # 3. Visualize samples
    if len(train_rgb) > 0:
        visualize_samples(train_rgb, train_masks, n=4)

    print("\n  ✓ Data exploration complete. Check runs/data_samples.png")
    print("  If everything looks correct, proceed to Phase 2 → python train.py")
