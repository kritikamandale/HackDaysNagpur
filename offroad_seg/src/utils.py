"""
utils.py — Helper utilities
"""
import os
import random
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ── Class metadata ────────────────────────────────────────────────────
CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

# Distinct colors for visualization (RGB 0-255)
CLASS_COLORS = [
    (45, 106, 79),    # 0 Trees        — dark green
    (82, 183, 136),   # 1 Lush Bushes  — mid green
    (212, 160, 23),   # 2 Dry Grass    — golden
    (169, 121, 77),   # 3 Dry Bushes   — tan brown
    (140, 109, 92),   # 4 Ground Clutter — muted brown
    (233, 196, 106),  # 5 Flowers      — yellow
    (107, 66, 38),    # 6 Logs         — dark wood
    (156, 156, 156),  # 7 Rocks        — grey
    (196, 168, 130),  # 8 Landscape    — sandy
    (135, 206, 235),  # 9 Sky          — light blue
]

# Raw mask pixel value → model class index
RAW_ID_TO_INDEX = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4,
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}


# ── Seeding ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Config loading ────────────────────────────────────────────────────
def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── Mask remapping ────────────────────────────────────────────────────
def remap_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    Convert raw mask pixel values (100, 200, 300 …) to class indices (0-9).
    Unknown pixel values → 255 (ignored in loss).
    """
    out = np.full(raw_mask.shape, 255, dtype=np.uint8)
    for raw_id, idx in RAW_ID_TO_INDEX.items():
        out[raw_mask == raw_id] = idx
    return out


# ── Visualization ─────────────────────────────────────────────────────
def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """
    Convert class-index mask (H, W) → RGB color image (H, W, 3).
    """
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(CLASS_COLORS):
        color_img[mask == idx] = color
    return color_img


def show_sample(image: np.ndarray, mask: np.ndarray, pred: np.ndarray = None,
                title: str = "", save_path: str = None):
    """
    Visualize image + ground-truth mask + (optional) prediction side by side.
    image : (H, W, 3) uint8
    mask  : (H, W)    class indices
    pred  : (H, W)    class indices
    """
    cols = 3 if pred is not None else 2
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 5))

    axes[0].imshow(image)
    axes[0].set_title("RGB Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(mask_to_color(mask))
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    if pred is not None:
        axes[2].imshow(mask_to_color(pred))
        axes[2].set_title("Prediction", fontsize=12)
        axes[2].axis("off")

    # Legend
    patches = [mpatches.Patch(color=[c/255 for c in col], label=name)
               for col, name in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center",
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()
    plt.close()


# ── Checkpoint helpers ────────────────────────────────────────────────
def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  Checkpoint saved → {path}")


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  Loaded checkpoint from {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# ── Plot training curves ──────────────────────────────────────────────
def plot_training_curves(history: dict, save_path: str = "runs/training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train Loss", color="#E24B4A")
    axes[0].plot(history["val_loss"],   label="Val Loss",   color="#378ADD")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["val_iou"],   label="Val mIoU",   color="#1D9E75")
    if "train_iou" in history:
        axes[1].plot(history["train_iou"], label="Train mIoU", color="#BA7517")
    axes[1].set_title("Mean IoU Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Training Progress", fontsize=14, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Training curves saved → {save_path}")
    plt.close()


# ── Pretty print IoU table ────────────────────────────────────────────
def print_iou_table(per_class_iou: list, mean_iou: float):
    print("\n" + "=" * 45)
    print(f"{'Class':<18} {'IoU':>8}")
    print("-" * 45)
    for i, iou_val in enumerate(per_class_iou):
        flag = " ← weak" if iou_val < 0.4 else ""
        print(f"  {CLASS_NAMES[i]:<16} {iou_val:>8.4f}{flag}")
    print("-" * 45)
    print(f"  {'Mean IoU':<16} {mean_iou:>8.4f}")
    print("=" * 45 + "\n")
