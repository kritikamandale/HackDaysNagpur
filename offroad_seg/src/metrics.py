"""
metrics.py — IoU / mIoU computation for semantic segmentation
"""
import torch
import numpy as np
from src.utils import CLASS_NAMES


class IoUMeter:
    """
    Running mean IoU tracker — accumulates confusion matrix over batches,
    then computes per-class and mean IoU at the end of an epoch.

    Usage:
        meter = IoUMeter(num_classes=10)
        for batch in loader:
            preds, targets = ...
            meter.update(preds, targets)
        mean_iou, per_class = meter.compute()
        meter.reset()
    """

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        # Confusion matrix: rows = gt class, cols = predicted class
        self.conf_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds   : (B, H, W) — argmax class predictions
        targets : (B, H, W) — ground-truth class indices
        """
        preds   = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Mask out ignored pixels
        valid   = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]

        # Accumulate confusion matrix
        indices = targets * self.num_classes + preds
        m = np.bincount(indices, minlength=self.num_classes ** 2)
        self.conf_matrix += m.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """
        Returns:
            mean_iou     : float
            per_class_iou: list[float]  (0.0 if class never appears)
        """
        cm   = self.conf_matrix
        tp   = np.diag(cm)
        fp   = cm.sum(axis=0) - tp   # false positives (predicted but wrong)
        fn   = cm.sum(axis=1) - tp   # false negatives (missed)
        denom = tp + fp + fn

        per_class_iou = np.where(denom > 0, tp / denom, 0.0)
        # Only average over classes that actually appear in the data
        present = denom > 0
        mean_iou = per_class_iou[present].mean() if present.any() else 0.0

        return float(mean_iou), per_class_iou.tolist()

    def per_class_table(self) -> str:
        """Pretty string table of per-class IoU."""
        mean_iou, per_class = self.compute()
        lines = ["\n  Per-Class IoU:\n",
                 f"  {'Class':<20} {'IoU':>7}  {'Status'}"]
        lines.append("  " + "-" * 40)
        for i, iou in enumerate(per_class):
            name   = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}"
            status = "✓ good" if iou >= 0.5 else ("△ ok" if iou >= 0.3 else "✗ weak")
            lines.append(f"  {name:<20} {iou:>7.4f}  {status}")
        lines.append("  " + "-" * 40)
        lines.append(f"  {'Mean IoU':<20} {mean_iou:>7.4f}")
        return "\n".join(lines)
