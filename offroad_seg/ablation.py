"""
ablation.py — Architecture ablation study for offroad semantic segmentation

Trains 4 model configurations sequentially and compares:
    val_mIoU  |  trainable params  |  training time

Configurations:
    1. DeepLabV3Plus + resnet34
    2. UNet          + resnet34
    3. FPN           + efficientnet-b2
    4. SegFormer     + mit_b2

Usage:
    python ablation.py
    python ablation.py --config configs/config.yaml --epochs 20
    python ablation.py --wandb          # log each run to W&B

Outputs:
    ablation_results.csv        — ranked results table
    ablation_comparison.png     — bar chart of val mIoU
"""
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import copy
import time
import argparse

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

from src.utils      import load_config, set_seed, CLASS_NAMES
from src.dataset    import get_dataloaders
from src.transforms import get_train_transforms, get_val_transforms
from src.model      import build_model
from src.losses     import build_loss
from src.metrics    import IoUMeter
from train          import train_one_epoch, validate

# ── Optional W&B ──────────────────────────────────────────────────────
try:
    import wandb
except ImportError:
    wandb = None

# ── The 4 ablation configurations ─────────────────────────────────────
ABLATION_CONFIGS = [
    {
        "architecture":    "DeepLabV3Plus",
        "encoder":         "resnet34",
        "encoder_weights": "imagenet",
    },
    {
        "architecture":    "Unet",
        "encoder":         "resnet34",
        "encoder_weights": "imagenet",
    },
    {
        "architecture":    "FPN",
        "encoder":         "efficientnet-b2",
        "encoder_weights": "imagenet",
    },
    {
        "architecture":    "SegFormer",
        "encoder":         "mit_b2",
        "encoder_weights": "imagenet",
    },
]


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_single_ablation(base_cfg: dict, arch_cfg: dict, device: torch.device,
                        epochs: int, use_wandb: bool = False) -> dict:
    """
    Train one architecture configuration and return a result dict.

    Returns keys: architecture, encoder, val_mIoU, params, train_time_s
    """
    arch  = arch_cfg["architecture"]
    enc   = arch_cfg["encoder"]
    label = f"{arch} + {enc}"

    print(f"\n{'='*62}")
    print(f"  Config : {label}")
    print(f"  Epochs : {epochs}")
    print(f"{'='*62}")

    # Build per-run config by deep-copying base and overriding model block
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["architecture"]    = arch
    cfg["model"]["encoder"]         = enc
    cfg["model"]["encoder_weights"] = arch_cfg["encoder_weights"]
    cfg["train"]["epochs"]          = epochs

    # ── Model ─────────────────────────────────────────────────────────
    model    = build_model(cfg).to(device)
    n_params = _count_params(model)

    # ── Data ──────────────────────────────────────────────────────────
    img_size     = cfg["train"]["image_size"]
    train_loader, val_loader = get_dataloaders(
        cfg,
        get_train_transforms(img_size, cfg),
        get_val_transforms(img_size),
    )

    # ── Loss / optimiser / scheduler ──────────────────────────────────
    loss_fn   = build_loss(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg["optimizer"]["lr"],
        weight_decay = cfg["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = epochs,
        eta_min = cfg["scheduler"]["eta_min"],
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    train_meter = IoUMeter(cfg["classes"]["num_classes"])
    val_meter   = IoUMeter(cfg["classes"]["num_classes"])

    # ── W&B run ───────────────────────────────────────────────────────
    wb_run = None
    if use_wandb and wandb is not None:
        wb_run = wandb.init(
            project = "offroad-segmentation",
            name    = f"ablation_{arch}_{enc}",
            group   = "ablation",
            config  = {
                **cfg,
                "ablation_architecture": arch,
                "ablation_encoder":      enc,
            },
            reinit  = True,
        )

    # ── Training loop ─────────────────────────────────────────────────
    best_iou   = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, loss_fn, train_meter, device, scaler
        )
        val_loss, val_iou, _ = validate(
            model, val_loader, loss_fn, val_meter, device
        )
        scheduler.step()

        lr_now   = optimizer.param_groups[0]["lr"]
        best_iou = max(best_iou, val_iou)

        print(f"  [{epoch+1:03d}/{epochs}]  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_mIoU={val_iou:.4f}  lr={lr_now:.2e}")

        if wb_run is not None:
            wandb.log({
                "epoch":         epoch + 1,
                "train_loss":    train_loss,
                "val_loss":      val_loss,
                "train_mIoU":    train_iou,
                "val_mIoU":      val_iou,
                "learning_rate": lr_now,
            })

    train_time = time.time() - start_time

    if wb_run is not None:
        wandb.log({
            "best_val_mIoU":  best_iou,
            "n_params":       n_params,
            "train_time_s":   train_time,
        })
        wandb.finish()

    print(f"\n  ✓ {label}: best val mIoU={best_iou:.4f}  "
          f"params={n_params:,}  time={train_time:.0f}s")

    return {
        "architecture": arch,
        "encoder":      enc,
        "val_mIoU":     round(best_iou, 4),
        "params":       n_params,
        "train_time_s": round(train_time, 1),
    }


def print_ranked_table(df: pd.DataFrame):
    col_w = {"rank": 5, "arch": 18, "enc": 18, "miou": 8, "params": 13, "time": 9}
    header = (f"  {'Rank':<{col_w['rank']}} {'Architecture':<{col_w['arch']}} "
              f"{'Encoder':<{col_w['enc']}} {'mIoU':>{col_w['miou']}} "
              f"{'Params':>{col_w['params']}} {'Time(s)':>{col_w['time']}}")
    sep = "  " + "-" * (sum(col_w.values()) + len(col_w) - 1)

    print("\n" + "=" * (len(header) - 2))
    print(header)
    print(sep)
    for rank, (_, row) in enumerate(df.iterrows(), start=1):
        print(f"  {rank:<{col_w['rank']}} "
              f"{row['architecture']:<{col_w['arch']}} "
              f"{row['encoder']:<{col_w['enc']}} "
              f"{row['val_mIoU']:>{col_w['miou']}.4f} "
              f"{int(row['params']):>{col_w['params']},} "
              f"{row['train_time_s']:>{col_w['time']}.0f}")
    print("=" * (len(header) - 2) + "\n")


def save_bar_chart(df: pd.DataFrame, path: str):
    labels = [f"{r['architecture']}\n{r['encoder']}" for _, r in df.iterrows()]
    values = df["val_mIoU"].tolist()
    colors = ["#1D9E75", "#378ADD", "#E24B4A", "#BA7517"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors[:len(values)],
                  edgecolor="white", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_title("Ablation Study — val mIoU by Architecture",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("val mIoU")
    y_max = max(values) if values else 1.0
    ax.set_ylim(0, y_max * 1.18)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Architecture ablation study")
    p.add_argument("--config", default="configs/config.yaml",
                   help="Path to base config YAML")
    p.add_argument("--epochs", type=int, default=None,
                   help="Epochs per run (overrides config value, default 20)")
    p.add_argument("--wandb", action="store_true",
                   help="Log each run to W&B as a separate run in the "
                        "offroad-segmentation project")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    epochs = args.epochs if args.epochs is not None else cfg["train"].get("epochs", 20)
    set_seed(cfg.get("seed", 42))

    device = torch.device(
        cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # W&B availability check
    use_wandb = False
    if args.wandb:
        if wandb is not None:
            use_wandb = True
            print("  W&B logging: enabled")
        else:
            print("  [WARN] --wandb flag set but wandb is not installed. "
                  "Install with: pip install wandb")

    print(f"\n  Running {len(ABLATION_CONFIGS)} configurations × {epochs} epochs each")

    results = []
    for arch_cfg in ABLATION_CONFIGS:
        try:
            result = run_single_ablation(
                base_cfg  = cfg,
                arch_cfg  = arch_cfg,
                device    = device,
                epochs    = epochs,
                use_wandb = use_wandb,
            )
        except Exception as exc:
            label = f"{arch_cfg['architecture']} + {arch_cfg['encoder']}"
            print(f"\n  [ERROR] {label} failed: {exc}")
            result = {
                "architecture": arch_cfg["architecture"],
                "encoder":      arch_cfg["encoder"],
                "val_mIoU":     0.0,
                "params":       0,
                "train_time_s": 0.0,
                "error":        str(exc),
            }
        results.append(result)

    # ── Build ranked DataFrame ─────────────────────────────────────────
    df = (pd.DataFrame(results)
            .sort_values("val_mIoU", ascending=False)
            .reset_index(drop=True))

    # ── Save CSV ───────────────────────────────────────────────────────
    csv_path = "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved → {csv_path}")

    # ── Print ranked table ─────────────────────────────────────────────
    print_ranked_table(df)

    # ── Save bar chart ─────────────────────────────────────────────────
    # Only chart runs that actually completed (val_mIoU > 0 or no error key)
    chart_df = df[~df.get("error", pd.Series(dtype=str)).notna()].copy() \
               if "error" in df.columns else df.copy()
    if chart_df.empty:
        chart_df = df.copy()
    save_bar_chart(chart_df, "ablation_comparison.png")

    best = df.iloc[0]
    print(f"\n  Best configuration: {best['architecture']} + {best['encoder']} "
          f"(val mIoU = {best['val_mIoU']:.4f})\n")


if __name__ == "__main__":
    main()
