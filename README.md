
---

## 🚀 The Frontend Demo App (`demo_static`)

A responsive, glassmorphic UI to visualize the segmentation model's capabilities and leverage Google's Gemini for contextual terrain insight.

### Features
- **Upload & Segment:** Upload unseen terrain images and view side-by-side original and prediction masks.
- **Overlay Mode:** Toggle a visual overlay to superimpose the predicted segmentation directly on the original RGB image.
- **Gemini Terrain Insight:** Built-in integration with Google's Gemini API to analyze the image, detect terrain risks, and highlight notable objects.

*(Note: The backend proxy handling image uploads and `GEMINI_API_KEY` calls is expected to be running for full UI functionality.)*

---

## 🧠 The ML Pipeline (`offroad_seg`)

Built using PyTorch, `segmentation-models-pytorch` (SMP), and Albumentations for robust data augmentation.

### Setup & Environment
Ensure you have Python 3.10 installed, then create your virtual environment (Conda or `venv`):
```bash
cd offroad_seg
conda create -n seg python=3.10 -y
conda activate seg
pip install -r requirements.txt
