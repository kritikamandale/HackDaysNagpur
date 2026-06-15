"""
explain.py — GradCAM explainability for offroad segmentation

For each of the 10 terrain classes, finds up to 3 validation images where
that class is present (≥5% of GT pixels), generates GradCAM heatmaps,
and saves a [RGB | GradCAM overlay | GT mask] grid per class.

Usage:
    python explain.py
    python explain.py --checkpoint runs/best_model.pth --min_pct 0.05

Outputs:
    outputs/gradcam/class_<ClassName>.png   — one grid per class
"""
import os
import sys
import argparse

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from src.utils      import load_config, CLASS_NAMES, CLASS_COLORS, mask_to_color
from src.dataset    import SegmentationDataset
from src.transforms import get_val_transforms
from src.model      import load_model_for_inference

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_OK = True
except ImportError:
    GRADCAM_OK = False


# ── ImageNet de-normalisation constants ───────────────────────────────
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description="GradCAM explainability for segmentation")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default="runs/best_model.pth")
    p.add_argument("--output_dir", default="outputs/gradcam")
    p.add_argument("--min_pct",    type=float, default=0.05,
                   help="Minimum fraction of GT pixels for a class to be 'present'")
    p.add_argument("--n_samples",  type=int, default=3,
                   help="Max validation images to use per class")
    return p.parse_args()


def get_target_layers(model: nn.Module, encoder_name: str) -> list:
    """Return [target_layer] for GradCAM based on the encoder architecture."""
    enc  = model.encoder
    name = encoder_name.lower()

    # ResNet / ResNeXt
    if any(x in name for x in ('resnet', 'resnext', 'se_res', 'wide_res')):
        return [enc.layer4[-1]]

    # MiT (SegFormer)
    if 'mit_b' in name:
        for attr in ('block4', 'block3'):
            if hasattr(enc, attr):
                return [getattr(enc, attr)[-1]]

    # EfficientNet (timm backend in smp)
    if 'efficientnet' in name:
        for attr in ('blocks', '_blocks', 'features'):
            if hasattr(enc, attr):
                container = getattr(enc, attr)
                children  = list(container.children())
                if children:
                    last = children[-1]
                    grandchildren = list(last.children())
                    return [grandchildren[-1] if grandchildren else last]

    # Generic fallback: last 3×3 Conv2d in the encoder
    target = None
    for m in enc.modules():
        if isinstance(m, nn.Conv2d) and m.kernel_size == (3, 3):
            target = m
    if target is None:
        raise RuntimeError(
            f"Cannot find a suitable GradCAM target layer for encoder '{encoder_name}'. "
            "Specify one manually."
        )
    return [target]


def find_class_samples(dataset: SegmentationDataset,
                       num_classes: int, min_pct: float) -> dict:
    """
    Scan the dataset and return {class_idx: [sample_indices, ...]}
    containing samples where that class covers >= min_pct of the image.
    """
    class_samples = {i: [] for i in range(num_classes)}
    print(f"  Scanning {len(dataset)} val samples for class presence ≥{min_pct*100:.0f}% ...")

    for idx in range(len(dataset)):
        _, mask = dataset.get_raw_sample(idx)
        n_pixels = mask.size
        for cls in range(num_classes):
            if (mask == cls).sum() / n_pixels >= min_pct:
                class_samples[cls].append(idx)

    return class_samples


def denorm(tensor: torch.Tensor) -> np.ndarray:
    """(3,H,W) normalised tensor → (H,W,3) float32 in [0,1]."""
    return (tensor.cpu() * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy()


def run_gradcam(model, img_tensor: torch.Tensor, gt_mask: np.ndarray,
                class_idx: int, target_layers: list,
                device: torch.device) -> np.ndarray:
    """
    Run GradCAM for class_idx on img_tensor.
    Returns grayscale_cam (H, W) in [0, 1].
    """
    gt_binary = (gt_mask == class_idx).astype(np.float32)
    targets   = [SemanticSegmentationTarget(class_idx, gt_binary)]

    cam = GradCAM(model=model, target_layers=target_layers)
    input_tensor = img_tensor.unsqueeze(0).to(device)

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets,
                        eigen_smooth=True, aug_smooth=False)
    return grayscale_cam[0]  # (H, W)


def save_class_grid(panels: list, class_name: str, class_idx: int,
                    out_path: str):
    """
    panels : list of (img_float, cam_overlay, gt_colored) — one tuple per sample
    Saves a rows×3 figure to out_path.
    """
    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    col_titles = ["RGB Image", "GradCAM Overlay", "Ground Truth Mask"]
    for col, t in enumerate(col_titles):
        axes[0][col].set_title(t, fontsize=12, fontweight="bold", pad=8)

    for row, (img, overlay, gt_color) in enumerate(panels):
        axes[row][0].imshow(img)
        axes[row][1].imshow(overlay)
        axes[row][2].imshow(gt_color)
        for ax in axes[row]:
            ax.axis("off")

    # Colour legend
    patches = [
        mpatches.Patch(color=[c / 255 for c in col], label=nm)
        for col, nm in zip(CLASS_COLORS, CLASS_NAMES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(
        f"GradCAM — Class {class_idx}: {class_name}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    if not GRADCAM_OK:
        print("\n  [ERROR] pytorch-grad-cam is not installed.")
        print("          pip install grad-cam")
        sys.exit(1)

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")

    # ── Load model ─────────────────────────────────────────────────────
    print("\n── Loading model ────────────────────────────────────")
    model        = load_model_for_inference(args.checkpoint, cfg, str(device))
    encoder_name = cfg["model"]["encoder"]

    try:
        target_layers = get_target_layers(model, encoder_name)
        print(f"  GradCAM target layer: {type(target_layers[0]).__name__}")
    except RuntimeError as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)

    # ── Load val dataset ───────────────────────────────────────────────
    print("\n── Loading val dataset ───────────────────────────────")
    img_size  = cfg["train"]["image_size"]
    transform = get_val_transforms(img_size)

    dataset = SegmentationDataset(
        rgb_dir   = cfg["data"]["val_rgb"],
        mask_dir  = cfg["data"]["val_masks"],
        transform = transform,
    )
    num_classes = cfg["classes"]["num_classes"]

    # ── Find samples per class ─────────────────────────────────────────
    class_samples = find_class_samples(dataset, num_classes, args.min_pct)
    for cls, idxs in class_samples.items():
        print(f"  {CLASS_NAMES[cls]:<18}: {len(idxs)} candidate images")

    # ── GradCAM per class ──────────────────────────────────────────────
    print(f"\n── Generating GradCAM grids → {args.output_dir} ─────")

    focus_scores = {}  # class_idx → focus_score (higher = focuses on correct region)

    for class_idx in range(num_classes):
        cls_name = CLASS_NAMES[class_idx]
        sample_idxs = class_samples[class_idx][:args.n_samples]

        if not sample_idxs:
            print(f"  [{cls_name}] no samples found — skipped")
            focus_scores[class_idx] = None
            continue

        panels = []
        cam_on_class  = []
        cam_off_class = []

        for s_idx in sample_idxs:
            img_tensor, mask_tensor = dataset[s_idx]
            _, gt_mask = dataset.get_raw_sample(s_idx)

            try:
                gray_cam = run_gradcam(
                    model, img_tensor, gt_mask, class_idx,
                    target_layers, device
                )
            except Exception as exc:
                print(f"  [{cls_name}] GradCAM failed on sample {s_idx}: {exc}")
                continue

            img_float  = denorm(img_tensor)
            overlay    = show_cam_on_image(img_float, gray_cam, use_rgb=True)
            gt_colored = mask_to_color(gt_mask)

            panels.append((img_float, overlay, gt_colored))

            # Collect activation values for focus analysis
            class_mask = (gt_mask == class_idx)
            if class_mask.any():
                cam_on_class.append(float(gray_cam[class_mask].mean()))
            other_mask = (gt_mask != class_idx) & (gt_mask != 255)
            if other_mask.any():
                cam_off_class.append(float(gray_cam[other_mask].mean()))

        if not panels:
            focus_scores[class_idx] = None
            continue

        # Compute focus score: ratio of mean activation on class vs off class
        mean_on  = float(np.mean(cam_on_class))  if cam_on_class  else 0.0
        mean_off = float(np.mean(cam_off_class)) if cam_off_class else 1e-6
        focus_scores[class_idx] = mean_on / (mean_off + 1e-6)

        out_path = str(Path(args.output_dir) / f"class_{cls_name.replace(' ', '_')}.png")
        save_class_grid(panels, cls_name, class_idx, out_path)
        print(f"  [{cls_name}] saved {len(panels)} panel(s)  "
              f"focus_score={focus_scores[class_idx]:.2f}  → {out_path}")

    # ── Focus analysis summary ─────────────────────────────────────────
    print("\n── GradCAM Focus Analysis ───────────────────────────")
    print(f"  {'Class':<18} {'Focus Score':>12}  {'Assessment'}")
    print("  " + "-" * 52)

    correct   = []
    incorrect = []

    for cls_idx, score in focus_scores.items():
        name = CLASS_NAMES[cls_idx]
        if score is None:
            status = "no data"
            print(f"  {name:<18} {'—':>12}  {status}")
            continue
        if score >= 1.2:
            status = "focuses CORRECTLY"
            correct.append(name)
        elif score >= 0.9:
            status = "mixed focus"
        else:
            status = "focuses INCORRECTLY"
            incorrect.append(name)
        print(f"  {name:<18} {score:>12.2f}  {status}")

    print("\n  Classes with correct spatial focus:")
    print("   ", ", ".join(correct) if correct else "none")
    print("  Classes with incorrect spatial focus:")
    print("   ", ", ".join(incorrect) if incorrect else "none")
    print(f"\n  GradCAM grids saved → {args.output_dir}/\n")


if __name__ == "__main__":
    main()
