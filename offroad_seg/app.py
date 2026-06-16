import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cv2
import gradio as gr
import numpy as np
import torch

from src.model          import load_model_for_inference
from src.transforms     import get_val_transforms
from src.utils          import load_config, mask_to_color, CLASS_NAMES, CLASS_COLORS
from src.traversability import (
    build_traversability_map,
    compute_safety_score,
    get_risk_breakdown,
    RISK_COLORS_RGB,
)

# ── Model loading ─────────────────────────────────────────────────────
_model_error = None
try:
    cfg       = load_config("configs/config.yaml")
    device    = torch.device("cpu")
    model     = load_model_for_inference("runs/best_model.pth", cfg, "cpu")
    transform = get_val_transforms(cfg["train"]["image_size"])
    model.eval()
except FileNotFoundError:
    _model_error = "No trained model at <code>runs/best_model.pth</code>. Run <code>python train.py</code> first."
    model = None; transform = None
    cfg = load_config("configs/config.yaml")
except Exception as e:
    _model_error = f"Model failed to load: {e}"
    model = None; transform = None
    cfg = load_config("configs/config.yaml")


# ══════════════════════════════════════════════════════════════════════
#  Core inference
# ══════════════════════════════════════════════════════════════════════
def _infer(image_rgb: np.ndarray) -> np.ndarray:
    """Run model on an RGB numpy image; return class-index mask."""
    aug    = transform(image=image_rgb)
    tensor = aug["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    h, w = image_rgb.shape[:2]
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


def _terrain_stats_html(pred: np.ndarray) -> str:
    unique, counts = np.unique(pred, return_counts=True)
    total = pred.size
    rows = sorted(zip(unique, counts), key=lambda x: -x[1])
    html = "<div class='stats-wrap'>"
    for cls_idx, count in rows:
        pct   = 100 * count / total
        name  = CLASS_NAMES[cls_idx]
        color = CLASS_COLORS[cls_idx]
        hex_c = "#{:02x}{:02x}{:02x}".format(*color)
        html += f"""
        <div class='stat-row'>
            <div class='stat-dot' style='background:{hex_c}'></div>
            <div class='stat-name'>{name}</div>
            <div class='stat-bar-wrap'>
                <div class='stat-bar-fill' style='width:{max(pct,1):.1f}%;background:{hex_c}'></div>
            </div>
            <div class='stat-pct'>{pct:.1f}%</div>
        </div>"""
    return html + "</div>"


def _traversability_html(safety_score: int, breakdown: dict) -> str:
    if safety_score >= 70:
        color, label, bg = "#22c55e", "SAFE",    "rgba(34,197,94,0.12)"
    elif safety_score >= 40:
        color, label, bg = "#eab308", "CAUTION", "rgba(234,179,8,0.12)"
    else:
        color, label, bg = "#ef4444", "DANGER",  "rgba(239,68,68,0.12)"

    risk_bars = ""
    for risk, pct in breakdown.items():
        rc = "#{:02x}{:02x}{:02x}".format(*RISK_COLORS_RGB[risk])
        risk_bars += f"""
        <div class='risk-row'>
            <span class='risk-label-text'>{risk.capitalize()}</span>
            <div class='risk-bar-bg'>
                <div class='risk-bar-fill' style='width:{max(pct,0.5):.1f}%;background:{rc}'></div>
            </div>
            <span class='risk-pct'>{pct:.1f}%</span>
        </div>"""

    return f"""
    <div class='trav-wrapper' style='background:{bg}'>
        <div class='score-ring' style='border-color:{color}'>
            <div class='score-num' style='color:{color}'>{safety_score}</div>
            <div class='score-sub'>/ 100</div>
        </div>
        <div class='score-label-text' style='color:{color}'>{label}</div>
        <div class='risk-breakdown'>{risk_bars}</div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════
#  Feature 1 — Image analysis
# ══════════════════════════════════════════════════════════════════════
def segment_image(input_image):
    if _model_error:
        err = f"<div class='error-box'>{_model_error}</div>"
        return None, None, None, err, err, None

    if input_image is None:
        msg = "<div class='placeholder-msg'>Upload an image and click <b>Analyze Terrain</b>.</div>"
        return None, None, None, msg, msg, None

    pred       = _infer(input_image)
    color_mask = mask_to_color(pred)
    overlay    = cv2.addWeighted(input_image, 0.45, color_mask, 0.55, 0)

    trav_map   = build_traversability_map(pred)
    trav_over  = cv2.addWeighted(input_image, 0.35, trav_map, 0.65, 0)

    safety     = compute_safety_score(pred)
    breakdown  = get_risk_breakdown(pred)

    stats_html = _terrain_stats_html(pred)
    trav_html  = _traversability_html(safety, breakdown)

    # Store for PDF
    unique, counts = np.unique(pred, return_counts=True)
    total = pred.size
    state = {
        "input":      input_image,
        "mask":       color_mask,
        "overlay":    overlay,
        "trav_over":  trav_over,
        "safety":     safety,
        "breakdown":  breakdown,
        "terrain":    {CLASS_NAMES[i]: round(100.0*c/total, 1)
                       for i, c in zip(unique, counts)},
    }
    
    pdf_path = generate_pdf(state)
    
    return color_mask, overlay, trav_over, stats_html, trav_html, gr.DownloadButton(value=pdf_path, interactive=True)


# ══════════════════════════════════════════════════════════════════════
#  Feature 2 — Video processing
# ══════════════════════════════════════════════════════════════════════
MAX_VIDEO_FRAMES = 250   # ~8 s at 30 fps — keeps processing under ~2 min on CPU

def process_video(video_path, progress=gr.Progress()):
    if video_path is None:
        return None
    if _model_error or model is None:
        return None

    cap = cv2.VideoCapture(video_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    to_proc  = min(n_frames, MAX_VIDEO_FRAMES)

    # Read first frame to get true dimensions
    ret, first = cap.read()
    if not ret:
        cap.release()
        return None
    fh, fw = first.shape[:2]
    out_w   = fw * 2   # side-by-side: original | traversability

    tmp      = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(out_path, fourcc, fps, (out_w, fh))

    def _process_frame(frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pred      = _infer(frame_rgb)
        trav_map  = build_traversability_map(pred)
        trav_bgr  = cv2.cvtColor(trav_map, cv2.COLOR_RGB2BGR)
        overlay   = cv2.addWeighted(frame_bgr, 0.35, trav_bgr, 0.65, 0)
        return np.hstack([frame_bgr, overlay])

    # First frame already read
    out.write(_process_frame(first))

    for i in range(1, to_proc):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(_process_frame(frame))
        progress((i + 1) / to_proc, desc=f"Frame {i+1}/{to_proc}")

    cap.release()
    out.release()
    return out_path


# ══════════════════════════════════════════════════════════════════════
#  Feature 2b — Webcam live stream
# ══════════════════════════════════════════════════════════════════════
def webcam_process(frame):
    if frame is None or model is None:
        return frame
    pred     = _infer(frame)
    trav_map = build_traversability_map(pred)
    overlay  = cv2.addWeighted(frame, 0.4, trav_map, 0.6, 0)
    score    = compute_safety_score(pred)

    # Burn safety score onto the frame
    color_bgr = (0, 200, 50) if score >= 70 else (0, 200, 220) if score >= 40 else (60, 60, 240)
    cv2.putText(overlay, f"Safety: {score}/100", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr, 2, cv2.LINE_AA)
    return overlay


# ══════════════════════════════════════════════════════════════════════
#  Feature 4 — PDF report
# ══════════════════════════════════════════════════════════════════════
def generate_pdf(state):
    if state is None:
        return None

    fig = plt.figure(figsize=(16, 22), facecolor="white")
    fig.suptitle("Offroad Terrain Analysis Report",
                 fontsize=22, fontweight="bold", color="black", y=0.98)

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.3,
                           top=0.95, bottom=0.04, left=0.06, right=0.97)

    def _show_img(ax, img, title):
        ax.imshow(img)
        ax.set_title(title, color="black", fontsize=11, pad=6)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Row 0 — images
    _show_img(fig.add_subplot(gs[0, 0]), state["input"],     "Input Image")
    _show_img(fig.add_subplot(gs[0, 1]), state["mask"],      "Segmentation Mask")
    _show_img(fig.add_subplot(gs[0, 2]), state["overlay"],   "Seg Overlay")

    # Row 1 — traversability image + safety score + risk pie
    _show_img(fig.add_subplot(gs[1, 0]), state["trav_over"], "Traversability Map")

    ax_score = fig.add_subplot(gs[1, 1])
    ax_score.set_facecolor("#f8f9fa")
    score = state["safety"]
    sc    = "#22c55e" if score >= 70 else "#eab308" if score >= 40 else "#ef4444"
    lbl   = "SAFE"    if score >= 70 else "CAUTION"  if score >= 40 else "DANGER"
    ax_score.text(0.5, 0.55, str(score),  fontsize=68, ha="center", va="center",
                  color=sc, fontweight="bold", transform=ax_score.transAxes)
    ax_score.text(0.5, 0.22, "Safety Score / 100", fontsize=12, ha="center",
                  color="black", transform=ax_score.transAxes)
    ax_score.text(0.5, 0.10, lbl, fontsize=16, ha="center",
                  color=sc, fontweight="bold", transform=ax_score.transAxes)
    ax_score.set_title("Safety Score", color="black", fontsize=11, pad=6)
    ax_score.axis("off")

    ax_pie = fig.add_subplot(gs[1, 2])
    ax_pie.set_facecolor("#f8f9fa")
    bd = state["breakdown"]
    slices = [(k, v) for k, v in bd.items() if v > 0]
    pie_colors = {"safe": "#22c55e", "caution": "#eab308", "danger": "#ef4444"}
    ax_pie.pie(
        [s[1] for s in slices],
        labels=[s[0].capitalize() for s in slices],
        colors=[pie_colors[s[0]] for s in slices],
        autopct="%1.1f%%",
        textprops={"color": "black", "fontsize": 10},
        startangle=90,
    )
    ax_pie.set_title("Risk Distribution", color="black", fontsize=11, pad=6)
    ax_pie.set_facecolor("#f8f9fa")

    # Row 2 — terrain bar chart (full width)
    ax_bar = fig.add_subplot(gs[2, :])
    ax_bar.set_facecolor("#f8f9fa")
    terrain = dict(sorted(state["terrain"].items(), key=lambda x: x[1]))
    bar_colors = ["#{:02x}{:02x}{:02x}".format(*CLASS_COLORS[CLASS_NAMES.index(n)])
                  for n in terrain]
    bars = ax_bar.barh(list(terrain.keys()), list(terrain.values()),
                       color=bar_colors, edgecolor="none", height=0.6)
    ax_bar.set_facecolor("#f8f9fa")
    ax_bar.set_xlabel("Coverage (%)", color="black", fontsize=10)
    ax_bar.set_title("Terrain Class Distribution", color="black", fontsize=11, pad=6)
    ax_bar.tick_params(colors="black")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for spine in ["bottom", "left"]:
        ax_bar.spines[spine].set_color("#ccc")
    for bar, val in zip(bars, terrain.values()):
        ax_bar.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", color="black", fontsize=9)

    # Row 3 — text summary
    ax_txt = fig.add_subplot(gs[3, :])
    ax_txt.set_facecolor("#f8f9fa")
    ax_txt.axis("off")
    dominant = max(state["terrain"], key=state["terrain"].get)
    rec = ("Terrain is suitable for vehicle navigation."
           if score >= 70 else
           "Proceed with caution — hazards are present."
           if score >= 40 else
           "High-risk terrain. Navigation not recommended.")
    summary = (
        f"  TERRAIN ANALYSIS SUMMARY\n\n"
        f"  Overall Assessment : {lbl}  (Score {score}/100)\n"
        f"  Dominant Terrain   : {dominant}  ({state['terrain'][dominant]:.1f}%)\n\n"
        f"  Risk Breakdown :\n"
        f"    Safe    {bd['safe']:.1f}%     Caution  {bd['caution']:.1f}%     Danger  {bd['danger']:.1f}%\n\n"
        f"  Recommendation : {rec}\n\n"
        f"  Generated by Offroad Terrain Segmentation AI  ·  DeepLabV3+ + ResNet-34  ·  mIoU 58.4%"
    )
    ax_txt.text(0.01, 0.95, summary, transform=ax_txt.transAxes, fontsize=10.5,
                color="black", va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.8", facecolor="white",
                          edgecolor="#ccc", linewidth=1))
    ax_txt.set_title("Summary", color="black", fontsize=11, pad=6)

    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, "Offroad_Terrain_Report.pdf")
    plt.savefig(pdf_path, format="pdf", facecolor="white",
                bbox_inches="tight", dpi=120)
    plt.close(fig)
    return pdf_path


# ══════════════════════════════════════════════════════════════════════
#  Legend HTML (built once)
# ══════════════════════════════════════════════════════════════════════
_legend_html = "<div class='legend-grid'>"
for _name, _color in zip(CLASS_NAMES, CLASS_COLORS):
    _hex = "#{:02x}{:02x}{:02x}".format(*_color)
    _legend_html += f"""
    <div class='legend-item'>
        <div class='legend-swatch' style='background:{_hex}'></div>
        <span class='legend-name'>{_name}</span>
    </div>"""
_legend_html += "</div>"


# ══════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body {
    background: #0d1117 !important;
    margin: 0; padding: 0;
}
.gradio-container {
    background: #0d1117 !important;
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 16px 20px !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh;
}
footer { display: none !important; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #161b22 0%, #1f2d3d 60%, #0f3460 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
}
.app-title { font-size: 30px; font-weight: 700; color: #ffffff; margin: 0 0 6px 0; }
.app-sub   { font-size: 13px; color: rgba(255,255,255,0.55); margin: 0; font-weight: 400; }
.badges    { display: flex; gap: 10px; flex-wrap: wrap; }
.badge {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 8px 16px;
    color: #e2e8f0;
    font-size: 11px;
    font-weight: 500;
    text-align: center;
}
.badge-val { font-size: 20px; font-weight: 700; color: #63b3ed; display: block; }

/* ── Section labels ── */
.sec-label {
    font-size: 10px; font-weight: 600; color: rgba(255,255,255,0.4);
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px;
}

/* ── Buttons ── */
.analyze-btn {
    background: linear-gradient(135deg,#667eea,#764ba2) !important;
    border: none !important; border-radius: 10px !important;
    color: white !important; font-size: 15px !important;
    font-weight: 600 !important; padding: 14px !important;
    width: 100% !important; cursor: pointer !important;
    transition: opacity 0.2s !important; margin-top: 10px !important;
}
.analyze-btn:hover { opacity: 0.85 !important; }

.pdf-btn {
    background: linear-gradient(135deg,#f093fb,#f5576c) !important;
    border: none !important; border-radius: 10px !important;
    color: white !important; font-size: 14px !important;
    font-weight: 600 !important; padding: 12px !important;
    width: 100% !important; cursor: pointer !important;
    margin-top: 8px !important;
}
.process-btn {
    background: linear-gradient(135deg,#11998e,#38ef7d) !important;
    border: none !important; border-radius: 10px !important;
    color: #0d1117 !important; font-size: 15px !important;
    font-weight: 700 !important; padding: 14px !important;
    width: 100% !important; cursor: pointer !important;
    margin-top: 10px !important;
}

/* ── Terrain stats ── */
.stats-wrap { padding: 4px 0; }
.stat-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
}
.stat-row:last-child { border-bottom: none; }
.stat-dot   { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.stat-name  { font-size: 12px; font-weight: 500; color: #e2e8f0; min-width: 110px; }
.stat-bar-wrap {
    flex: 1; height: 6px; background: rgba(255,255,255,0.07);
    border-radius: 99px; overflow: hidden;
}
.stat-bar-fill { height: 100%; border-radius: 99px; opacity: 0.85; }
.stat-pct { font-size: 11px; font-weight: 600; color: rgba(255,255,255,0.5); min-width: 40px; text-align: right; }

/* ── Traversability score card ── */
.trav-wrapper {
    border-radius: 14px; padding: 20px 16px;
    border: 1px solid rgba(255,255,255,0.07);
    display: flex; flex-direction: column; align-items: center; gap: 12px;
}
.score-ring {
    width: 110px; height: 110px; border-radius: 50%;
    border: 6px solid; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.score-num  { font-size: 42px; font-weight: 700; line-height: 1; }
.score-sub  { font-size: 11px; color: rgba(255,255,255,0.45); }
.score-label-text { font-size: 16px; font-weight: 700; letter-spacing: 1px; }
.risk-breakdown { width: 100%; }
.risk-row {
    display: flex; align-items: center; gap: 8px;
    padding: 5px 0;
}
.risk-label-text { font-size: 11px; color: #cbd5e0; min-width: 58px; }
.risk-bar-bg {
    flex: 1; height: 6px; background: rgba(255,255,255,0.07);
    border-radius: 99px; overflow: hidden;
}
.risk-bar-fill { height: 100%; border-radius: 99px; }
.risk-pct { font-size: 11px; color: rgba(255,255,255,0.45); min-width: 38px; text-align: right; }

/* ── Legend ── */
.legend-grid {
    display: grid; grid-template-columns: repeat(5,1fr); gap: 8px;
}
.legend-item {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 11px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
}
.legend-swatch { width: 13px; height: 13px; border-radius: 3px; flex-shrink: 0; }
.legend-name   { font-size: 11px; font-weight: 500; color: #cbd5e0; }

/* ── Error / placeholder ── */
.error-box {
    background: rgba(245,101,101,0.1); border: 1px solid rgba(245,101,101,0.3);
    border-radius: 10px; padding: 16px 20px;
    color: #fc8181; font-size: 13px; text-align: center;
}
.placeholder-msg {
    color: rgba(255,255,255,0.3); font-size: 13px;
    text-align: center; padding: 24px 0;
}

/* ── Video tab info box ── */
.video-info {
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 10px; padding: 14px 18px;
    color: #90cdf4; font-size: 12px; line-height: 1.6;
}
.webcam-info {
    background: rgba(72,187,120,0.08);
    border: 1px solid rgba(72,187,120,0.2);
    border-radius: 10px; padding: 14px 18px;
    color: #9ae6b4; font-size: 12px; line-height: 1.6;
}
"""


# ══════════════════════════════════════════════════════════════════════
#  Gradio layout
# ══════════════════════════════════════════════════════════════════════
with gr.Blocks(title="Offroad Terrain Segmentation") as demo:

    # ── Header ──────────────────────────────────────────────────────
    gr.HTML("""
    <div class='app-header'>
        <div>
            <h1 class='app-title'>🏔️ Offroad Terrain Segmentation</h1>
            <p class='app-sub'>Semantic segmentation · Traversability analysis · DeepLabV3+ · ResNet-34 · 10 terrain classes</p>
        </div>
        <div class='badges'>
            <div class='badge'><span class='badge-val'>58.4%</span>Val mIoU</div>
            <div class='badge'><span class='badge-val'>30</span>Epochs</div>
            <div class='badge'><span class='badge-val'>10</span>Classes</div>
            <div class='badge'><span class='badge-val'>2857</span>Train Images</div>
        </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Image Analysis ────────────────────────────────────
        with gr.Tab("Image Analysis"):

            # Row A: input | seg mask | seg overlay
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML("<div class='sec-label'>Input Image</div>")
                    input_img = gr.Image(
                        label=None, type="numpy", height=350,
                        show_label=False,
                    )
                    run_btn = gr.Button("Analyze Terrain", elem_classes="analyze-btn")

                with gr.Column(scale=1, min_width=280):
                    gr.HTML("<div class='sec-label'>Segmentation Mask</div>")
                    pred_img = gr.Image(
                        label=None, type="numpy", height=350,
                        show_label=False, interactive=False,
                    )

                with gr.Column(scale=1, min_width=280):
                    gr.HTML("<div class='sec-label'>Seg Overlay</div>")
                    overlay_img = gr.Image(
                        label=None, type="numpy", height=350,
                        show_label=False, interactive=False,
                    )

            gr.HTML("<div style='margin:10px 0'></div>")

            # Row B: traversability map | safety score | terrain stats
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=280):
                    gr.HTML("<div class='sec-label'>Traversability Map</div>")
                    trav_img = gr.Image(
                        label=None, type="numpy", height=300,
                        show_label=False, interactive=False,
                    )

                with gr.Column(scale=1, min_width=220):
                    gr.HTML("<div class='sec-label'>Safety Score</div>")
                    trav_html = gr.HTML(
                        value="<div class='placeholder-msg'>Run analysis to see safety score</div>"
                    )

                with gr.Column(scale=1, min_width=280):
                    gr.HTML("<div class='sec-label'>Terrain Distribution</div>")
                    stats_html = gr.HTML(
                        value="<div class='placeholder-msg'>Run analysis to see stats</div>"
                    )

            gr.HTML("<div style='margin:10px 0'></div>")

            # Row C: PDF download
            with gr.Row():
                with gr.Column(scale=1):
                    pdf_btn = gr.DownloadButton("Download PDF Report", elem_classes="pdf-btn", interactive=False)
                with gr.Column(scale=3):
                    pass   # spacer

            gr.HTML("<div style='margin:16px 0'></div>")

            # Row D: legend
            gr.HTML("<div class='sec-label'>Class Legend</div>")
            gr.HTML(_legend_html)

            # Events
            run_btn.click(
                fn=segment_image,
                inputs=input_img,
                outputs=[pred_img, overlay_img, trav_img,
                         stats_html, trav_html, pdf_btn],
            )

        # ── Tab 2: Video Analysis ────────────────────────────────────
        with gr.Tab("Video Analysis"):
            gr.HTML("""
            <div class='video-info' style='margin-bottom:16px'>
                Upload an MP4 / MOV video of offroad terrain.<br>
                Each frame is segmented and coloured by traversability risk
                (🟢 Safe · 🟡 Caution · 🔴 Danger).<br>
                Output is a side-by-side video: <b>Original | Traversability overlay</b>.<br>
                Processing is capped at 250 frames (~8 s at 30 fps) to keep it fast on CPU.
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    video_in = gr.Video(
                        label="Upload terrain video",
                        sources=["upload"],
                        height=360,
                    )
                    video_btn = gr.Button("Process Video", elem_classes="process-btn")

                with gr.Column(scale=1):
                    video_out = gr.Video(
                        label="Traversability output (Original | Overlay)",
                        height=360,
                        interactive=False,
                    )

            video_btn.click(
                fn=process_video,
                inputs=video_in,
                outputs=video_out,
            )

        # ── Tab 3: Live Webcam ───────────────────────────────────────
        with gr.Tab("Live Webcam"):
            gr.HTML("""
            <div class='webcam-info' style='margin-bottom:16px'>
                Point your webcam at offroad terrain for real-time traversability analysis.<br>
                Green = Safe · Amber = Caution · Red = Danger<br>
                <b>Allow camera access</b> when the browser prompts you.
            </div>
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<div class='sec-label'>Camera Feed</div>")
                    webcam_in = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        type="numpy",
                        label=None,
                        show_label=False,
                        height=400,
                    )

                with gr.Column(scale=1):
                    gr.HTML("<div class='sec-label'>Live Traversability Overlay</div>")
                    webcam_out = gr.Image(
                        type="numpy",
                        label=None,
                        show_label=False,
                        height=400,
                        interactive=False,
                    )

            webcam_in.stream(
                fn=webcam_process,
                inputs=[webcam_in],
                outputs=[webcam_out],
                stream_every=0.2,
                time_limit=120,
            )

import os
_hf = os.getenv("SPACE_ID") is not None
demo.launch(
    css=css,
    server_name="0.0.0.0" if not _hf else None,
    server_port=7860 if not _hf else None,
)
