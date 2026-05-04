"""
test.py — Evaluate trained model on UNSEEN test images

Usage:
    python test.py
    python test.py --checkpoint runs/best_model.pth

What it does:
  1. Loads the best trained model
  2. Runs inference on all images in data/testImages/
  3. Saves color-coded prediction images to outputs/predictions/
  4. Computes IoU if ground-truth masks are available (in data/test/masks/)
  5. Saves a failure case grid → outputs/failure_cases.png
  6. Prints a final per-class IoU table
"""
import os
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import time
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from pathlib import Path

from src.utils      import load_config, mask_to_color, CLASS_NAMES, CLASS_COLORS, print_iou_table
from src.model      import load_model_for_inference
from src.transforms import get_val_transforms
from src.metrics    import IoUMeter
from src.dataset    import TestDataset


# ── Args ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default="runs/best_model.pth")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--gt_mask_dir", default=None,
                   help="Optional: path to ground-truth masks for test set (to compute IoU)")
    return p.parse_args()


# ── Inference on single image ─────────────────────────────────────────
@torch.no_grad()
def predict_image(model, image_np: np.ndarray, transform, device) -> np.ndarray:
    """
    image_np : (H, W, 3) uint8 RGB
    returns  : (H, W)    class index prediction
    """
    orig_h, orig_w = image_np.shape[:2]

    augmented = transform(image=image_np)
    tensor    = augmented["image"].unsqueeze(0).to(device)   # (1, C, H, W)

    logits = model(tensor)                 # (1, num_classes, H, W)
    pred   = logits.argmax(dim=1)[0]      # (H, W)

    # Resize prediction back to original image size
    pred_np = pred.cpu().numpy().astype(np.uint8)
    pred_np = cv2.resize(pred_np, (orig_w, orig_h),
                         interpolation=cv2.INTER_NEAREST)
    return pred_np


# ── Failure case analysis ─────────────────────────────────────────────
def save_failure_cases(cases: list, save_path: str, max_cases: int = 6):
    """
    cases : list of (image_np, gt_mask, pred_mask, filename, iou)
    Saves a grid of worst-performing images for the report.
    """
    cases = sorted(cases, key=lambda x: x[4])[:max_cases]  # sort by IoU ascending
    n = len(cases)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = [axes]

    for row, (image, gt, pred, fname, iou) in enumerate(cases):
        axes[row][0].imshow(image)
        axes[row][0].set_title(f"RGB — {fname}", fontsize=9)
        axes[row][0].axis("off")

        axes[row][1].imshow(mask_to_color(gt))
        axes[row][1].set_title("Ground Truth", fontsize=9)
        axes[row][1].axis("off")

        axes[row][2].imshow(mask_to_color(pred))
        axes[row][2].set_title(f"Prediction (mIoU={iou:.3f})", fontsize=9)
        axes[row][2].axis("off")

    patches = [mpatches.Patch(color=[c/255 for c in col], label=name)
               for col, name in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("Failure Cases — Lowest IoU Samples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Failure cases saved → {save_path}")
    plt.close()


# ── Inference speed benchmark ─────────────────────────────────────────
def benchmark_speed(model, transform, device, n_runs: int = 50):
    """Measure average inference time per image (ms)."""
    dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    # Warmup
    for _ in range(5):
        predict_image(model, dummy, transform, device)

    start = time.perf_counter()
    for _ in range(n_runs):
        predict_image(model, dummy, transform, device)
    elapsed = (time.perf_counter() - start) / n_runs * 1000  # ms per image
    return elapsed


# ── Main ──────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Load model
    print(f"\n── Loading model from {args.checkpoint} ─────────────────")
    model = load_model_for_inference(args.checkpoint, cfg, str(device))

    img_size  = cfg["train"]["image_size"]
    transform = get_val_transforms(img_size)

    output_dir = Path(args.output_dir)
    pred_dir   = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # ── Inference ────────────────────────────────────────────────────
    test_image_dir = cfg["data"]["test_images"]
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    test_paths = sorted([
        p for p in Path(test_image_dir).iterdir()
        if p.suffix.lower() in img_exts
    ])
    print(f"\n── Running inference on {len(test_paths)} test images ─────")

    meter         = IoUMeter(cfg["classes"]["num_classes"])
    has_gt        = args.gt_mask_dir is not None
    failure_cases = []

    for img_path in tqdm(test_paths, desc="  Predicting", ncols=80):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [WARN] Cannot read {img_path.name}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        pred = predict_image(model, image_rgb, transform, device)

        # Save color prediction
        color_pred = mask_to_color(pred)
        cv2.imwrite(
            str(pred_dir / img_path.name),
            cv2.cvtColor(color_pred, cv2.COLOR_RGB2BGR)
        )

        # Compute IoU if GT available
        if has_gt:
            from src.utils import remap_mask
            gt_path = Path(args.gt_mask_dir) / img_path.name
            if not gt_path.exists():
                # Try other extension
                for ext in img_exts:
                    candidate = Path(args.gt_mask_dir) / (img_path.stem + ext)
                    if candidate.exists():
                        gt_path = candidate
                        break

            if gt_path.exists():
                raw_gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
                if raw_gt.ndim == 3:
                    raw_gt = raw_gt[:, :, 0]
                gt_mask = remap_mask(raw_gt)

                pred_t = torch.from_numpy(pred).unsqueeze(0)
                gt_t   = torch.from_numpy(gt_mask).unsqueeze(0)
                meter.update(pred_t, gt_t)

                # Per-sample IoU for failure analysis
                sample_meter = IoUMeter(cfg["classes"]["num_classes"])
                sample_meter.update(pred_t, gt_t)
                sample_iou, _ = sample_meter.compute()
                failure_cases.append((image_rgb, gt_mask, pred, img_path.name, sample_iou))

    print(f"\n  Predictions saved → {pred_dir}/")

    # ── IoU Report ───────────────────────────────────────────────────
    if has_gt and len(failure_cases) > 0:
        mean_iou, per_class_iou = meter.compute()
        print(meter.per_class_table())
        print_iou_table(per_class_iou, mean_iou)

        save_failure_cases(
            failure_cases,
            save_path=str(output_dir / "failure_cases.png")
        )

    # ── Speed benchmark ──────────────────────────────────────────────
    print("\n── Benchmarking inference speed ─────────────────────")
    ms_per_image = benchmark_speed(model, transform, device)
    status = "✓ PASS" if ms_per_image < 50 else "✗ FAIL (target <50ms)"
    print(f"  Average inference time: {ms_per_image:.1f} ms/image  {status}")

    print("\n  ✓ Testing complete.")
    print(f"  Predictions → {pred_dir}")
    if has_gt:
        print(f"  Failure cases → {output_dir}/failure_cases.png")


if __name__ == "__main__":
    main()
