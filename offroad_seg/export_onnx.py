"""
export_onnx.py — Export trained model to ONNX and benchmark latency

Usage:
    python export_onnx.py
    python export_onnx.py --checkpoint runs/best_model.pth --output best_model.onnx
    python export_onnx.py --n_runs 200

Outputs:
    best_model.onnx   — exported ONNX model
    Benchmark summary printed to stdout
"""
import os
import sys
import argparse
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import torch

from src.utils import load_config
from src.model import load_model_for_inference

try:
    import onnx
    import onnxruntime as ort
    ONNX_OK = True
except ImportError:
    ONNX_OK = False


def parse_args():
    p = argparse.ArgumentParser(description="Export segmentation model to ONNX")
    p.add_argument("--config",     default="configs/config.yaml")
    p.add_argument("--checkpoint", default="runs/best_model.pth")
    p.add_argument("--output",     default="best_model.onnx")
    p.add_argument("--n_runs",     type=int, default=100,
                   help="Number of forward passes for latency benchmark")
    p.add_argument("--batch_size", type=int, default=1)
    return p.parse_args()


def benchmark_pytorch(model, dummy: torch.Tensor, device: torch.device,
                      n_runs: int = 100):
    """Returns (mean_ms, std_ms) over n_runs forward passes."""
    model.eval()
    with torch.no_grad():
        for _ in range(10):  # warm-up
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies)), float(np.std(latencies))


def benchmark_onnx(onnx_path: str, dummy_np: np.ndarray, n_runs: int = 100):
    """Returns (mean_ms, std_ms) over n_runs ONNX runtime forward passes."""
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess       = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    # Warm-up
    for _ in range(10):
        sess.run(None, {input_name: dummy_np})

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: dummy_np})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies)), float(np.std(latencies))


def main():
    if not ONNX_OK:
        print("\n  [ERROR] Missing dependencies. Install them with:")
        print("          pip install onnx onnxruntime")
        sys.exit(1)

    args = parse_args()
    cfg  = load_config(args.config)

    h, w = cfg["train"]["image_size"]
    B    = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Input size : ({B}, 3, {h}, {w})")
    print(f"  Benchmark  : {args.n_runs} runs")

    # ── Load PyTorch model ─────────────────────────────────────────────
    print("\n── Loading model ────────────────────────────────────")
    model      = load_model_for_inference(args.checkpoint, cfg, str(device))
    dummy_torch = torch.randn(B, 3, h, w, device=device)
    dummy_np    = dummy_torch.cpu().numpy().astype(np.float32)

    # ── Export to ONNX ─────────────────────────────────────────────────
    print(f"\n── Exporting to ONNX ────────────────────────────────")
    model.eval()
    torch.onnx.export(
        model,
        dummy_torch,
        args.output,
        opset_version       = 17,
        input_names         = ["input"],
        output_names        = ["output"],
        dynamic_axes        = {
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        do_constant_folding = True,
    )
    onnx_size_mb = os.path.getsize(args.output) / (1024 ** 2)
    print(f"  Exported  → {args.output}  ({onnx_size_mb:.2f} MB)")

    # ── Verify ONNX model ──────────────────────────────────────────────
    print("\n── Verifying ONNX model ─────────────────────────────")
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("  ONNX graph check : PASSED")

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess     = ort.InferenceSession(args.output, providers=providers)
    ort_out  = sess.run(None, {sess.get_inputs()[0].name: dummy_np})
    expected = (B, cfg["classes"]["num_classes"], h, w)
    got      = tuple(ort_out[0].shape)
    status   = "PASSED" if got == expected else f"MISMATCH — got {got}"
    print(f"  Output shape     : {got}  ({status})")

    # Numerical parity check
    with torch.no_grad():
        pt_out = model(dummy_torch).cpu().numpy()
    max_diff = float(np.abs(pt_out - ort_out[0]).max())
    print(f"  Max abs diff (PyTorch vs ONNX): {max_diff:.6f}")

    # ── Latency benchmarks ─────────────────────────────────────────────
    print(f"\n── Benchmarking latency ({args.n_runs} runs) ─────────────────")
    pt_mean, pt_std     = benchmark_pytorch(model, dummy_torch, device, args.n_runs)
    onnx_mean, onnx_std = benchmark_onnx(args.output, dummy_np, args.n_runs)

    speedup = pt_mean / onnx_mean if onnx_mean > 0 else float("inf")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 54)
    print("  ONNX Export & Benchmark Summary")
    print("=" * 54)
    arch = cfg["model"]["architecture"]
    enc  = cfg["model"]["encoder"]
    print(f"  Model         : {arch} + {enc}")
    print(f"  ONNX size     : {onnx_size_mb:.2f} MB")
    print(f"  PyTorch  lat. : {pt_mean:.2f} ± {pt_std:.2f} ms")
    print(f"  ONNX     lat. : {onnx_mean:.2f} ± {onnx_std:.2f} ms")
    if speedup >= 1.0:
        print(f"  Speedup       : {speedup:.2f}x  (ONNX faster)")
    else:
        print(f"  Speedup       : {speedup:.2f}x  (PyTorch faster on this device)")
    print(f"  Output file   : {args.output}")
    print("=" * 54 + "\n")


if __name__ == "__main__":
    main()
