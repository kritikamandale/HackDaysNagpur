"""
traversability.py — Convert segmentation masks to traversability risk maps.

Classes (index): Trees(0), Lush Bushes(1), Dry Grass(2), Dry Bushes(3),
                 Ground Clutter(4), Flowers(5), Logs(6), Rocks(7),
                 Landscape(8), Sky(9)
"""
import numpy as np

# ── Risk assignment per class ─────────────────────────────────────────
RISK_LEVEL = {
    0: "safe",     # Trees         — navigable around
    1: "safe",     # Lush Bushes   — soft, passable
    2: "caution",  # Dry Grass     — hidden holes, fire risk
    3: "caution",  # Dry Bushes    — thorny, rough terrain
    4: "caution",  # Ground Clutter — debris / unknown objects
    5: "safe",     # Flowers       — flat, passable ground
    6: "danger",   # Logs          — hard obstacle, axle damage
    7: "danger",   # Rocks         — tyre / chassis damage
    8: "safe",     # Landscape     — open traversable ground
    9: "safe",     # Sky           — non-ground, treated safe
}

RISK_COLORS_RGB = {
    "safe":    (34,  197,  94),   # green
    "caution": (234, 179,   8),   # amber
    "danger":  (239,  68,  68),   # red
}

RISK_WEIGHTS = {"safe": 100, "caution": 50, "danger": 0}


def build_traversability_map(pred_mask: np.ndarray) -> np.ndarray:
    """Return H×W×3 RGB colour map where each pixel = risk colour."""
    trav = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for cls_idx, risk in RISK_LEVEL.items():
        trav[pred_mask == cls_idx] = RISK_COLORS_RGB[risk]
    return trav


def compute_safety_score(pred_mask: np.ndarray) -> int:
    """Pixel-weighted safety score: 0 (all danger) → 100 (all safe)."""
    total = pred_mask.size
    weighted = sum(
        int(np.sum(pred_mask == cls)) * RISK_WEIGHTS[risk]
        for cls, risk in RISK_LEVEL.items()
    )
    return round(weighted / total)


def get_risk_breakdown(pred_mask: np.ndarray) -> dict:
    """Return {'safe': %, 'caution': %, 'danger': %} (1 dp)."""
    total = pred_mask.size
    bd = {"safe": 0.0, "caution": 0.0, "danger": 0.0}
    for cls_idx, risk in RISK_LEVEL.items():
        bd[risk] += 100.0 * int(np.sum(pred_mask == cls_idx)) / total
    return {k: round(v, 1) for k, v in bd.items()}
