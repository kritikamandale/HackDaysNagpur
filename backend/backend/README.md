# Backend (FastAPI) for Segmentation Demo

This backend exposes a single endpoint `/predict` that accepts an image upload and returns a segmentation mask PNG.

Requirements

- Python 3.9+
- see `requirements.txt`
- Place your trained model weights as `best_model.pth` inside this `backend/` folder before starting the server.

Quickstart (virtualenv or conda)

1. Create environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Place the model file:

- Copy your `best_model.pth` to this folder: `backend/best_model.pth`

3. Run the server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API

- POST `/predict` — multipart form with field `file` (image). Returns a PNG mask file where pixel values are class indices (0..NUM_CLASSES-1).

Notes

- The app uses `segmentation_models_pytorch` to instantiate a UNet with a `resnet34` encoder. If your checkpoint uses a different state-dict structure, adapt `main.py` accordingly.
- The server will run on CPU if CUDA is not available; if GPU exists, inference will use it.
 
Mock mode
---------
If `best_model.pth` is not present in the `backend/` folder the API will automatically run in *mock mode* and return a deterministic colorized preview mask generated from the input image brightness. This allows frontend development and testing before you provide a real trained model.

When you later place your `best_model.pth` in `backend/` and restart the server, the real model will be loaded automatically.

Saved outputs
-------------
Each prediction request saves two files into `backend/outputs/`:
- `mask_raw_*.png` — raw class-id mask (values 0..NUM_CLASSES-1)
- `mask_color_*.png` — colorized preview returned to the frontend

Place your real model in `backend/best_model.pth` to enable real inference.
