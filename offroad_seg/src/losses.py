"""
losses.py — Combined Dice + CrossEntropy loss for semantic segmentation

Why combined?
  CrossEntropy → punishes wrong class predictions per-pixel
  Dice Loss    → directly optimizes the IoU-like overlap metric
Together they handle class imbalance and boundary accuracy better
than either loss alone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.
    Ignores pixels with label == ignore_index (default 255).
    """

    def __init__(self, num_classes: int, smooth: float = 1.0,
                 ignore_index: int = 255):
        super().__init__()
        self.num_classes   = num_classes
        self.smooth        = smooth
        self.ignore_index  = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W)  — raw model output (no softmax applied yet)
        targets : (B, H, W)     — class indices
        """
        probs = F.softmax(logits, dim=1)   # (B, C, H, W)

        # Build valid pixel mask (ignore unknown labels)
        valid = (targets != self.ignore_index)        # (B, H, W)

        # One-hot encode targets: (B, H, W) → (B, C, H, W)
        t_clamped = targets.clone()
        t_clamped[~valid] = 0
        one_hot = F.one_hot(t_clamped, self.num_classes)  # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()     # (B, C, H, W)

        # Zero out invalid positions
        valid_4d = valid.unsqueeze(1).float()
        probs   = probs   * valid_4d
        one_hot = one_hot * valid_4d

        # Dice per class
        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        cardinality  = (probs + one_hot).sum(dim=(0, 2, 3))
        dice_per_cls = (2 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1.0 - dice_per_cls.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of Dice Loss and Cross-Entropy Loss.

    dice_weight + ce_weight should ideally sum to 1.0.
    class_weights : per-class weight tensor for CrossEntropy
                    (upweight rare classes like Logs, Flowers)
    """

    def __init__(self, num_classes: int,
                 dice_weight: float = 0.5,
                 ce_weight:   float = 0.5,
                 class_weights=None,
                 ignore_index: int = 255):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight   = ce_weight
        self.ignore_index = ignore_index

        self.dice_loss = DiceLoss(
            num_classes  = num_classes,
            ignore_index = ignore_index,
        )
        self.ce_loss = nn.CrossEntropyLoss(
            weight       = class_weights,
            ignore_index = ignore_index,
            reduction    = "mean",
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        d = self.dice_loss(logits, targets)
        c = self.ce_loss(logits, targets)
        return self.dice_weight * d + self.ce_weight * c, d.item(), c.item()


def build_loss(cfg: dict, device: str) -> CombinedLoss:
    """Build CombinedLoss from config."""
    import torch
    loss_cfg  = cfg["loss"]
    n_classes = cfg["classes"]["num_classes"]

    # Per-class weights — helps with class imbalance
    weights_list = loss_cfg.get("class_weights")
    class_weights = None
    if weights_list:
        class_weights = torch.tensor(weights_list, dtype=torch.float32).to(device)

    return CombinedLoss(
        num_classes   = n_classes,
        dice_weight   = loss_cfg.get("dice_weight", 0.5),
        ce_weight     = loss_cfg.get("ce_weight",   0.5),
        class_weights = class_weights,
    ).to(device)
