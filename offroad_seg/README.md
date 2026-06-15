# Offroad Semantic Scene Segmentation
### Duality AI Hackathon — Nagpur

[![W&B](https://img.shields.io/badge/Weights_&_Biases-FFCC33?style=for-the-badge&logo=WeightsAndBiases&logoColor=black)](https://wandb.ai/your-username/offroad-segmentation)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)](https://pytorch.org/)

> Replace `your-username` in the W&B badge URL with your actual W&B username after running with `--wandb`.

---

## Project Structure

```
offroad_seg/
├── data/
│   ├── train/
│   │   ├── rgb/          ← training RGB images
│   │   └── masks/        ← training mask images
│   ├── val/
│   │   ├── rgb/          ← validation RGB images
│   │   └── masks/        ← validation mask images
│   └── testImages/       ← unseen test images (RGB only)
│
├── src/
│   ├── dataset.py        ← PyTorch Dataset & DataLoader
│   ├── transforms.py     ← Albumentations augmentation pipelines
│   ├── model.py          ← segmentation model factory (smp)
│   ├── losses.py         ← combined Dice + CrossEntropy loss
│   ├── metrics.py        ← IoU metric tracking
│   └── utils.py          ← helpers, visualization, checkpointing
│
├── configs/
│   └── config.yaml       ← ALL hyperparameters live here
│
├── runs/                 ← checkpoints + training curves (auto-created)
├── outputs/              ← predictions + report visuals (auto-created)
│
├── explore_data.py       ← Phase 1: verify dataset before training
├── train.py              ← Phase 2: train the model
├── test.py               ← Phase 3: run inference on test images
├── visualize.py          ← Phase 3: generate report visualizations
├── requirements.txt
└── README.md
```

---

## Setup

### Step 1 — Clone / download this project
```bash
cd your_workspace
```

### Step 2 — Create conda environment
```bash
conda create -n seg python=3.10 -y
conda activate seg
pip install -r requirements.txt
```

### Step 3 — Download the Duality AI dataset
Download from: https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert

Place files so the structure looks like:
```
data/
  train/rgb/      ← jpg/png RGB images
  train/masks/    ← matching segmentation mask images
  val/rgb/
  val/masks/
  testImages/     ← unseen RGB test images
```

---

## Running the Project

### Phase 1 — Verify data
```bash
python explore_data.py
```
Check: are image counts correct? Does the class distribution make sense?
Output saved to `runs/data_samples.png`

---

### Phase 2 — Train the model
```bash
python train.py
```
- Trains for 80 epochs by default (change in `configs/config.yaml`)
- Saves best checkpoint → `runs/best_model.pth`
- Saves training curves → `runs/training_curves.png`
- Shows per-class IoU table every 5 epochs

To resume from a checkpoint:
```bash
python train.py --resume runs/checkpoint_epoch40.pth
```

---

### Phase 3 — Test on unseen images
```bash
python test.py
```
- Runs inference on all images in `data/testImages/`
- Saves color predictions → `outputs/predictions/`
- Benchmarks inference speed (target: <50ms per image)

If you have ground-truth masks for the test set:
```bash
python test.py --gt_mask_dir data/test/masks
```

---

### Generate report visualizations
```bash
python visualize.py
```
Produces:
- `outputs/report_grid.png`       — RGB / GT / Pred / Overlay grid
- `outputs/confusion_matrix.png`  — normalized confusion matrix
- `outputs/class_iou_bar.png`     — per-class IoU bar chart

---

### Train with W&B experiment tracking
```bash
python train.py --wandb
```
Logs per-epoch loss, mIoU, learning rate, and 5 sample prediction panels every 5 epochs.
Per-class IoU table is logged as a W&B Table at the end of training.

---

### Run architecture ablation study
```bash
python ablation.py            # uses epoch count from config
python ablation.py --epochs 10 --wandb
```
Trains DeepLabV3Plus, UNet, FPN, and SegFormer sequentially and produces:
- `ablation_results.csv`      — ranked results table
- `ablation_comparison.png`   — bar chart of val mIoU by architecture

---

## Architecture Comparison Results

Results from `python ablation.py` (placeholder values — replace after running):

| Architecture | Encoder | mIoU | Params (M) | Inference (ms) |
|---|---|---|---|---|
| DeepLabV3Plus | resnet34 | 0.XXXX | XX.X | XX.X |
| UNet | resnet34 | 0.XXXX | XX.X | XX.X |
| FPN | efficientnet-b2 | 0.XXXX | XX.X | XX.X |
| SegFormer | mit_b2 | 0.XXXX | XX.X | XX.X |

Update this table by running `python ablation.py` and copying values from `ablation_results.csv`.

---

## Deployment

### Export to ONNX
```bash
python export_onnx.py
python export_onnx.py --checkpoint runs/best_model.pth --output best_model.onnx --n_runs 200
```

Outputs:
- `best_model.onnx` — optimized ONNX model with dynamic batch size (opset 17)
- Benchmark summary comparing PyTorch vs ONNX latency (mean ± std over 100 runs)
- Numerical parity check (max absolute difference between PyTorch and ONNX outputs)

The ONNX model can be served with ONNX Runtime, TensorRT, or any ONNX-compatible runtime:
```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("best_model.onnx")
dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
output = sess.run(None, {"input": dummy})[0]  # shape: (1, 10, 256, 256)
pred_mask = output.argmax(axis=1)              # shape: (1, 256, 256)
```

---

## Explainability

### GradCAM heatmaps per terrain class
```bash
python explain.py
python explain.py --min_pct 0.03 --n_samples 3
```

Generates GradCAM attention maps (using `pytorch-grad-cam`) to show where the model
"looks" for each terrain class. Target layer: last convolutional block of the encoder.

For each of the 10 classes, saves a grid of `[RGB | GradCAM Overlay | GT Mask]`
for up to 3 validation images where that class is present in the ground truth.

Outputs:
- `outputs/gradcam/class_Trees.png`
- `outputs/gradcam/class_Lush_Bushes.png`
- ... (one file per class)

Also prints a **focus analysis** — whether the model attends to the correct spatial
regions for each class (focus score = mean activation on-class / mean activation off-class).

---

## Configuration

All hyperparameters are in `configs/config.yaml`.

Key settings to tune:
| Setting | Default | What to try |
|---|---|---|
| `model.architecture` | DeepLabV3Plus | Unet, FPN, UnetPlusPlus |
| `model.encoder` | resnet50 | resnet101, efficientnet-b4 |
| `train.batch_size` | 8 | 4 (if OOM), 16 (if GPU strong) |
| `train.epochs` | 80 | 50 (quick test), 120 (full) |
| `train.image_size` | [512, 512] | [384, 384] (faster), [640, 640] |
| `loss.class_weights` | see yaml | increase weight for rare classes |

---

## Class Map

| ID | Class | Index |
|---|---|---|
| 100 | Trees | 0 |
| 200 | Lush Bushes | 1 |
| 300 | Dry Grass | 2 |
| 500 | Dry Bushes | 3 |
| 550 | Ground Clutter | 4 |
| 600 | Flowers | 5 |
| 700 | Logs | 6 |
| 800 | Rocks | 7 |
| 7100 | Landscape | 8 |
| 10000 | Sky | 9 |

---

## Evaluation Metric

**Mean IoU (Intersection over Union)** — primary metric (80% of hackathon score)

```
IoU per class = TP / (TP + FP + FN)
Mean IoU      = average IoU across all classes present in dataset
```

Benchmarks:
- < 0.30 → model not learning
- 0.30–0.50 → baseline range
- 0.50–0.70 → competitive
- > 0.70 → strong result

---

## Tips to Improve IoU

1. **Rare class underperformance** (Logs, Flowers, Ground Clutter)
   → Increase class weights in `config.yaml` → `loss.class_weights`

2. **Overfitting on train but low val IoU**
   → Increase augmentation strength, reduce model size, add dropout

3. **Training too slow**
   → Reduce `batch_size`, reduce `image_size`, use `efficientnet-b0` encoder

4. **Improve generalization to new environment**
   → Add `GridDistortion`, `RandomBrightnessContrast`, `HueSaturationValue` augmentations
   → Try a ViT-based encoder (mit_b2 with SegFormer architecture)

---

## Dependencies

See `requirements.txt`. Key packages:
- `torch >= 2.0`
- `segmentation-models-pytorch >= 0.3.3`
- `albumentations >= 1.3.1`
- `torchmetrics >= 1.0.0`
- `wandb >= 0.16.0` — experiment tracking (optional, `--wandb` flag)
- `onnx >= 1.14.0` + `onnxruntime >= 1.16.0` — ONNX export and inference
- `grad-cam >= 1.4.8` — GradCAM explainability (`pip install grad-cam`)
