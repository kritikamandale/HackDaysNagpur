
---

## 🚀 Offroad Autonomous Image segmentation Project

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

# Offroad Semantic Scene Segmentation
### Duality AI Hackathon — Nagpur

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
