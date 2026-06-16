"""
FastAPI REST endpoint for offroad terrain segmentation.

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health    — model info
    POST /predict   — multipart/form-data, field 'file' = image file

Example curl:
    curl -X POST http://localhost:8000/predict \\
         -F "file=@my_trail.jpg" | python -m json.tool
"""

import base64

import cv2
import numpy as np
import torch

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    raise SystemExit(
        "FastAPI not installed.\n"
        "Run: pip install fastapi uvicorn python-multipart"
    )

from src.model      import load_model_for_inference
from src.transforms import get_val_transforms
from src.utils      import load_config, mask_to_color, CLASS_NAMES
from src.traversability import (
    build_traversability_map,
    compute_safety_score,
    get_risk_breakdown,
)

# ── Load model once at startup ────────────────────────────────────────
cfg       = load_config("configs/config.yaml")
device    = torch.device("cpu")
model     = load_model_for_inference("runs/best_model.pth", cfg, "cpu")
transform = get_val_transforms(cfg["train"]["image_size"])
model.eval()

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Offroad Terrain Segmentation API",
    description=(
        "Semantic segmentation + traversability analysis for offroad terrain.\n\n"
        "POST an image to `/predict` and receive terrain class percentages, "
        "a 0–100 safety score, and base64-encoded mask/traversability PNG images."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────
def _encode_png(rgb_array: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf).decode()


def _run_inference(image_rgb: np.ndarray) -> np.ndarray:
    aug    = transform(image=image_rgb)
    tensor = aug["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    h, w = image_rgb.shape[:2]
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/health", summary="Model health check")
def health():
    return {
        "status":      "ok",
        "model":       f"{cfg['model']['architecture']}+{cfg['model']['encoder']}",
        "num_classes": cfg["classes"]["num_classes"],
        "class_names": CLASS_NAMES,
    }


@app.post("/predict", summary="Segment an image and return traversability data")
async def predict(file: UploadFile = File(..., description="JPEG or PNG image")):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    bgr      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pred      = _run_inference(image_rgb)

    color_mask = mask_to_color(pred)
    trav_map   = build_traversability_map(pred)
    safety     = compute_safety_score(pred)
    breakdown  = get_risk_breakdown(pred)

    unique, counts = np.unique(pred, return_counts=True)
    total = pred.size
    terrain_stats = {
        CLASS_NAMES[i]: round(100.0 * int(c) / total, 2)
        for i, c in zip(unique, counts)
    }

    return JSONResponse({
        "safety_score":           safety,
        "risk_level":             "safe" if safety >= 70 else "caution" if safety >= 40 else "danger",
        "risk_breakdown":         breakdown,
        "terrain_stats":          terrain_stats,
        "segmentation_mask_b64":  _encode_png(color_mask),
        "traversability_map_b64": _encode_png(trav_map),
    })
