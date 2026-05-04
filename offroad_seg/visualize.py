"""
visualize.py — Generate report-ready visualizations

Usage:
    python visualize.py

Generates:
  1. outputs/report_grid.png        — side-by-side RGB/GT/Prediction for report
  2. outputs/confusion_matrix.png   — class confusion matrix
  3. outputs/class_iou_bar.png      — per-class IoU bar chart
  4. outputs/overlay_samples.png    — semi-transparent overlay on original image
"""
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from pathlib import Path

from src.utils      import load_config, remap_mask, mask_to_color, CLASS_NAMES, CLASS_COLORS
from src.model      import load_model_for_inference
from src.transforms import get_val_transforms
from src.metrics    import IoUMeter


def overlay_prediction(image: np.ndarray, pred_mask: np.ndarray,
                        alpha: float = 0.5) -> np.ndarray:
    """Blend color prediction mask over RGB image."""
    color_mask = mask_to_color(pred_mask)
    return cv2.addWeighted(image, 1 - alpha, color_mask, alpha, 0)


def plot_report_grid(samples: list, save_path: str):
    """
    samples : list of (image_np, gt_mask, pred_mask, filename)
    Creates a 4-column grid: RGB | GT | Prediction | Overlay
    """
    n = min(len(samples), 5)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n))
    if n == 1:
        axes = [axes]

    titles = ["RGB Image", "Ground Truth", "Prediction", "Overlay"]
    for col, t in enumerate(titles):
        axes[0][col].set_title(t, fontsize=12, fontweight="bold", pad=8)

    for row, (image, gt, pred, fname) in enumerate(samples[:n]):
        overlay = overlay_prediction(image, pred)
        panels  = [image, mask_to_color(gt), mask_to_color(pred), overlay]

        for col, panel in enumerate(panels):
            axes[row][col].imshow(panel)
            axes[row][col].axis("off")
        axes[row][0].set_ylabel(fname[:25], fontsize=8, rotation=0,
                                ha="right", va="center")

    patches = [mpatches.Patch(color=[c/255 for c in col], label=name)
               for col, name in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("Qualitative Results", fontsize=15, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Report grid → {save_path}")
    plt.close()


def plot_confusion_matrix(conf_matrix: np.ndarray, save_path: str):
    """Normalized confusion matrix heatmap — great for the report."""
    # Normalize by row (gt class)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    norm_cm  = conf_matrix / row_sums

    labels   = [n[:8] for n in CLASS_NAMES]  # shorten for display
    fig, ax  = plt.subplots(figsize=(12, 10))
    sns.heatmap(norm_cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, vmin=0, vmax=1)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Confusion matrix → {save_path}")
    plt.close()


def plot_class_iou_bar(per_class_iou: list, mean_iou: float, save_path: str):
    """Horizontal bar chart of per-class IoU — clear for the report."""
    colors  = ["#E24B4A" if v < 0.4 else ("#EF9F27" if v < 0.6 else "#1D9E75")
               for v in per_class_iou]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos   = range(len(CLASS_NAMES))

    bars = ax.barh(y_pos, per_class_iou, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel("IoU Score", fontsize=12)
    ax.set_title(f"Per-Class IoU  (mean={mean_iou:.4f})", fontsize=14, fontweight="bold")
    ax.axvline(x=mean_iou, color="#378ADD", linestyle="--", linewidth=1.5,
               label=f"Mean IoU = {mean_iou:.3f}")
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=10)

    # Value labels
    for bar, val in zip(bars, per_class_iou):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)

    # Legend for colors
    from matplotlib.patches import Patch
    legend_els = [Patch(color="#E24B4A", label="Weak (<0.4)"),
                  Patch(color="#EF9F27", label="OK (0.4-0.6)"),
                  Patch(color="#1D9E75", label="Good (≥0.6)")]
    ax.legend(handles=legend_els, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Class IoU bar chart → {save_path}")
    plt.close()


def main():
    cfg    = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n── Loading model ────────────────────────────────────")
    model     = load_model_for_inference("runs/best_model.pth", cfg, str(device))
    transform = get_val_transforms(cfg["train"]["image_size"])

    from src.utils import RAW_ID_TO_INDEX
    img_exts = {".jpg", ".jpeg", ".png"}

    # Collect val samples for visualization
    val_rgb_dir  = Path(cfg["data"]["val_rgb"])
    val_mask_dir = Path(cfg["data"]["val_masks"])
    val_paths    = sorted(p for p in val_rgb_dir.iterdir() if p.suffix.lower() in img_exts)

    samples  = []
    meter    = IoUMeter(cfg["classes"]["num_classes"])

    print(f"\n── Collecting {min(20, len(val_paths))} val samples ─────────────────")
    for img_path in val_paths[:20]:
        image_bgr = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Predict
        aug    = transform(image=image_rgb)
        tensor = aug["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        pred_np = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        pred_np = cv2.resize(pred_np, (image_rgb.shape[1], image_rgb.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

        # Load GT
        gt_path = None
        for ext in img_exts:
            c = val_mask_dir / (img_path.stem + ext)
            if c.exists():
                gt_path = c
                break
        if gt_path is None:
            continue

        raw_gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if raw_gt.ndim == 3:
            raw_gt = raw_gt[:, :, 0]
        gt_mask = remap_mask(raw_gt)

        meter.update(
            torch.from_numpy(pred_np).unsqueeze(0),
            torch.from_numpy(gt_mask).unsqueeze(0)
        )
        samples.append((image_rgb, gt_mask, pred_np, img_path.name))

    mean_iou, per_class_iou = meter.compute()

    out = Path("outputs")
    out.mkdir(exist_ok=True)

    print("\n── Generating report visualizations ─────────────────")
    plot_report_grid(samples[:5], str(out / "report_grid.png"))
    plot_confusion_matrix(meter.conf_matrix, str(out / "confusion_matrix.png"))
    plot_class_iou_bar(per_class_iou, mean_iou, str(out / "class_iou_bar.png"))

    print(f"\n  Mean IoU on val set: {mean_iou:.4f}")
    print(f"\n  All visualizations saved to {out}/")
    print("  Include these in your report!")


if __name__ == "__main__":
    main()
