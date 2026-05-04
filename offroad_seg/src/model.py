"""
model.py — Model factory using segmentation-models-pytorch (smp)
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def build_model(cfg: dict) -> nn.Module:
    """
    Build a segmentation model from config.

    Supported architectures:
        DeepLabV3Plus  — strong baseline, dilated convolutions
        Unet           — U-shaped encoder-decoder
        UnetPlusPlus   — dense skip connections
        FPN            — feature pyramid network
        SegFormer      — transformer-based (use mit_b2 encoder)

    Supported encoders (examples):
        resnet50, resnet101 — solid baselines
        efficientnet-b4     — good accuracy/speed trade-off
        mit_b2              — SegFormer encoder (needs SegFormer arch)
    """
    model_cfg = cfg["model"]
    arch      = model_cfg["architecture"]
    encoder   = model_cfg["encoder"]
    weights   = model_cfg["encoder_weights"]
    n_classes = cfg["classes"]["num_classes"]

    print(f"\n  Building model: {arch} + {encoder} ({n_classes} classes)")

    arch_map = {
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "Unet":          smp.Unet,
        "UnetPlusPlus":  smp.UnetPlusPlus,
        "FPN":           smp.FPN,
        "SegFormer":     smp.create_model,   # handled separately below
    }

    if arch == "SegFormer":
        # SegFormer needs mit_b0..b5 encoders
        model = smp.create_model(
            arch         = "SegformerB2",  # or B0..B5 for size trade-offs
            encoder_name = encoder,
            encoder_weights = weights,
            in_channels  = model_cfg["in_channels"],
            classes      = n_classes,
        )
    else:
        model_class = arch_map.get(arch)
        if model_class is None:
            raise ValueError(f"Unknown architecture: {arch}. "
                             f"Choose from {list(arch_map.keys())}")
        model = model_class(
            encoder_name    = encoder,
            encoder_weights = weights,
            in_channels     = model_cfg["in_channels"],
            classes         = n_classes,
            activation      = model_cfg.get("activation"),
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    return model


def load_model_for_inference(checkpoint_path: str, cfg: dict,
                              device: str = "cpu") -> nn.Module:
    """
    Load a trained model checkpoint for inference / testing.
    """
    model = build_model(cfg)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"  Model loaded for inference (epoch {ckpt.get('epoch', '?')})")
    return model
