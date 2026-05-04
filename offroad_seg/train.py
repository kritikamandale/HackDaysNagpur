"""
train.py — Main training script for Offroad Semantic Segmentation

Usage:
    python train.py
    python train.py --config configs/config.yaml

What happens:
  1. Loads config and builds model, dataloaders, loss, optimizer
  2. Trains for N epochs — saves best checkpoint by val mIoU
  3. Logs loss + IoU every epoch
  4. Saves training curves at the end → runs/training_curves.png
"""
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import time
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

# ── Local imports ─────────────────────────────────────────────────────
from src.utils      import load_config, set_seed, save_checkpoint, plot_training_curves, print_iou_table
from src.dataset    import get_dataloaders
from src.transforms import get_train_transforms, get_val_transforms
from src.model      import build_model
from src.losses     import build_loss
from src.metrics    import IoUMeter


# ── Argument parsing ──────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train segmentation model")
    p.add_argument("--config", default="configs/config.yaml",
                   help="Path to config YAML")
    p.add_argument("--resume",  default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


# ── One training epoch ────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, loss_fn,
                    meter, device, scaler=None):
    model.train()
    meter.reset()
    total_loss = 0.0

    pbar = tqdm(loader, desc="  Train", leave=False, ncols=90)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss, _, _ = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss, _, _ = loss_fn(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Update metrics
        preds = logits.detach().argmax(dim=1)
        meter.update(preds, masks)
        total_loss += loss.item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    mean_iou, _ = meter.compute()
    return total_loss / len(loader), mean_iou


# ── One validation epoch ──────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, loss_fn, meter, device):
    model.eval()
    meter.reset()
    total_loss = 0.0

    pbar = tqdm(loader, desc="  Val  ", leave=False, ncols=90)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        logits = model(images)
        loss, _, _ = loss_fn(logits, masks)

        preds = logits.argmax(dim=1)
        meter.update(preds, masks)
        total_loss += loss.item()

    mean_iou, per_class_iou = meter.compute()
    return total_loss / len(loader), mean_iou, per_class_iou


# ── Main ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg  = load_config(args.config)

    set_seed(cfg.get("seed", 42))

    # Device
    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU   : {torch.cuda.get_device_name(0)}")

    # Build components
    print("\n── Building model ───────────────────────────────────")
    model = build_model(cfg).to(device)

    img_size = cfg["train"]["image_size"]
    train_tf = get_train_transforms(img_size, cfg)
    val_tf   = get_val_transforms(img_size)

    print("\n── Loading data ─────────────────────────────────────")
    train_loader, val_loader = get_dataloaders(cfg, train_tf, val_tf)

    loss_fn = build_loss(cfg, device)
    print(f"\n  Loss: Dice × {cfg['loss']['dice_weight']} + CE × {cfg['loss']['ce_weight']}")

    # Optimizer
    opt_cfg   = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = opt_cfg["lr"],
        weight_decay = opt_cfg["weight_decay"],
    )

    # Scheduler
    sch_cfg   = cfg["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = sch_cfg["T_max"],
        eta_min = sch_cfg["eta_min"],
    )

    # Mixed precision scaler (only for CUDA)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Resume from checkpoint if specified
    start_epoch = 0
    best_iou    = 0.0
    if args.resume and Path(args.resume).exists():
        from src.utils import load_checkpoint
        ckpt = load_checkpoint(args.resume, model, optimizer)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_iou    = ckpt.get("best_iou", 0.0)

    # History for plotting
    history = {"train_loss": [], "val_loss": [],
               "train_iou":  [], "val_iou":  []}

    train_meter = IoUMeter(cfg["classes"]["num_classes"])
    val_meter   = IoUMeter(cfg["classes"]["num_classes"])

    n_epochs  = cfg["train"]["epochs"]
    save_dir  = cfg["train"]["save_dir"]
    log_every = cfg.get("log_interval", 5)

    print(f"\n── Training for {n_epochs} epochs ──────────────────────────\n")

    for epoch in range(start_epoch, n_epochs):
        epoch_start = time.time()

        train_loss, train_iou = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            train_meter, device, scaler
        )
        val_loss, val_iou, per_class_iou = validate(
            model, val_loader, loss_fn, val_meter, device
        )
        scheduler.step()

        # Log
        elapsed = time.time() - epoch_start
        lr_now  = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        print(f"  Epoch [{epoch+1:03d}/{n_epochs}] "
              f"| train_loss={train_loss:.4f}  val_loss={val_loss:.4f} "
              f"| train_mIoU={train_iou:.4f}  val_mIoU={val_iou:.4f} "
              f"| lr={lr_now:.6f}  ({elapsed:.1f}s)")

        # Detailed per-class table every log_every epochs
        if (epoch + 1) % log_every == 0:
            print_iou_table(per_class_iou, val_iou)

        # Save best checkpoint
        if val_iou > best_iou:
            best_iou = val_iou
            save_checkpoint(
                state={
                    "epoch":         epoch,
                    "model_state":   model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_iou":      best_iou,
                    "config":        cfg,
                },
                path=str(Path(save_dir) / "best_model.pth"),
            )
            print(f"  ★ New best val mIoU: {best_iou:.4f}")

        # Save latest checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                state={
                    "epoch":         epoch,
                    "model_state":   model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_iou":      best_iou,
                    "config":        cfg,
                },
                path=str(Path(save_dir) / f"checkpoint_epoch{epoch+1}.pth"),
            )

    # Final summary
    print(f"\n  ✓ Training complete. Best val mIoU = {best_iou:.4f}")
    print(f"  Best model saved → {save_dir}/best_model.pth")

    # Plot and save training curves
    plot_training_curves(history, save_path=str(Path(save_dir) / "training_curves.png"))

    print("\n  Next step → python test.py")


if __name__ == "__main__":
    main()
