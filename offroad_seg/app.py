import gradio as gr
import torch
import cv2
import numpy as np
from src.utils import load_config, mask_to_color, CLASS_NAMES, CLASS_COLORS
from src.model import load_model_for_inference
from src.transforms import get_val_transforms
import traceback

# Load configuration and model with error handling
try:
    print("Loading configuration...")
    cfg       = load_config("configs/config.yaml")
    print(f"Config loaded: {cfg}")
    
    print("Setting device to CPU...")
    device    = torch.device("cpu")
    
    print("Loading model from runs/best_model.pth...")
    model     = load_model_for_inference("runs/best_model.pth", cfg, "cpu")
    print("Model loaded successfully!")
    
    print("Loading transforms...")
    transform = get_val_transforms(cfg["train"]["image_size"])
    print("Transforms loaded!")
    
except Exception as e:
    print(f"ERROR during initialization: {str(e)}")
    traceback.print_exc()
    raise

def segment_image(input_image):
    if input_image is None:
        return None, None, "<p style='color:#999;text-align:center;'>Upload a desert image to begin terrain analysis.</p>"

    aug    = transform(image=input_image)
    tensor = aug["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

    h, w   = input_image.shape[:2]
    pred   = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    color_mask = mask_to_color(pred)
    overlay    = cv2.addWeighted(input_image, 0.5, color_mask, 0.5, 0)

    unique, counts = np.unique(pred, return_counts=True)
    total = pred.size
    
    # Create HTML-formatted statistics
    stats_html = "<div style='width:100%'>"
    for cls_idx, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        pct  = 100 * count / total
        name = CLASS_NAMES[cls_idx]
        
        stats_html += f"""
        <div class='stat-row'>
            <div class='stat-name'>{name}</div>
            <div class='stat-bar'><div class='stat-fill' style='width:{pct:.1f}%'></div></div>
            <div class='stat-percentage'>{pct:.1f}%</div>
        </div>
        """
    stats_html += "</div>"

    return color_mask, overlay, stats_html

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

body {
    background: linear-gradient(135deg, #F4A460 0%, #CD853F 50%, #8B6914 100%);
}

.gradio-container {
    max-width: 1400px;
    background: #FFF8F0;
}

.header-container {
    background: linear-gradient(135deg, #D2691E 0%, #CD853F 50%, #8B4513 100%);
    color: white;
    padding: 40px 20px;
    border-radius: 15px;
    margin-bottom: 30px;
    box-shadow: 0 10px 40px rgba(139, 69, 19, 0.4);
    border: 2px solid #DEB887;
}

.header-title {
    font-size: 48px;
    font-weight: 700;
    margin: 0 0 10px 0;
    text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3);
    letter-spacing: 0.5px;
}

.header-subtitle {
    font-size: 18px;
    opacity: 0.95;
    margin: 0;
    font-weight: 400;
    letter-spacing: 0.3px;
}

.upload-section {
    background: linear-gradient(135deg, rgba(244, 164, 96, 0.15), rgba(205, 133, 63, 0.15));
    border: 3px dashed #CD853F;
    border-radius: 15px;
    padding: 30px;
    text-align: center;
}

.segment-button {
    background: linear-gradient(135deg, #D2691E 0%, #CD853F 100%);
    border: 2px solid #8B4513;
    padding: 16px 40px;
    font-size: 18px;
    font-weight: 600;
    color: white;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 5px 20px rgba(210, 105, 30, 0.5);
}

.segment-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(210, 105, 30, 0.7);
    background: linear-gradient(135deg, #CD853F 0%, #DAA520 100%);
}

.results-container {
    background: linear-gradient(135deg, rgba(244, 164, 96, 0.08), rgba(205, 133, 63, 0.08));
    border-radius: 15px;
    padding: 30px;
    margin-top: 30px;
    border: 2px solid #DEB887;
}

.output-section {
    background: white;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(139, 69, 19, 0.1);
    border: 2px solid #FFDAB9;
}

.legend-container {
    background: white;
    border-radius: 12px;
    padding: 25px;
    margin-top: 20px;
    box-shadow: 0 4px 15px rgba(139, 69, 19, 0.1);
    border: 2px solid #FFDAB9;
}

.legend-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
    margin-top: 15px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px;
    background: linear-gradient(135deg, rgba(244, 164, 96, 0.12), rgba(205, 133, 63, 0.12));
    border-radius: 8px;
    border: 2px solid #FFDAB9;
}

.legend-color-box {
    width: 24px;
    height: 24px;
    border-radius: 6px;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(139, 69, 19, 0.2);
    border: 1px solid #8B4513;
}

.legend-label {
    font-size: 14px;
    font-weight: 600;
    color: #2C1810;
    letter-spacing: 0.2px;
}

.stats-container {
    background: linear-gradient(135deg, rgba(244, 164, 96, 0.12), rgba(205, 133, 63, 0.12));
    border-radius: 12px;
    padding: 20px;
    margin-top: 20px;
    border: 2px solid #FFDAB9;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid rgba(205, 133, 63, 0.2);
}

.stat-row:last-child {
    border-bottom: none;
}

.stat-name {
    font-weight: 700;
    color: #2C1810;
    min-width: 150px;
    letter-spacing: 0.2px;
}

.stat-bar {
    flex: 1;
    margin: 0 20px;
    height: 8px;
    background: rgba(205, 133, 63, 0.15);
    border-radius: 10px;
    overflow: hidden;
}

.stat-fill {
    background: linear-gradient(90deg, #D2691E, #CD853F);
    height: 100%;
    border-radius: 10px;
}

.stat-percentage {
    color: #D2691E;
    font-weight: 700;
    min-width: 60px;
    text-align: right;
    letter-spacing: 0.3px;
}

.footer {
    text-align: center;
    color: #666;
    font-size: 13px;
    margin-top: 40px;
    padding-top: 30px;
    border-top: 2px solid #DEB887;
    font-weight: 500;
    letter-spacing: 0.2px;
}

.image-section-title {
    font-size: 16px;
    font-weight: 700;
    color: #2C1810;
    margin-bottom: 10px;
    letter-spacing: 0.3px;
}
"""

with gr.Blocks(title="Offroad Segmentation", css=custom_css, theme=gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="red",
)) as demo:
    # Header
    with gr.Group(elem_classes="header-container"):
        gr.HTML("<h1 class='header-title'>🏜️ Offroad Desert Segmentation</h1>")
        gr.HTML("<p class='header-subtitle'>✨ AI-powered terrain classification for arid & desert environments</p>")

    # Upload section
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="upload-section"):
                gr.HTML("<div style='font-size:14px;color:#8B4513;font-weight:700;margin-bottom:15px;letter-spacing:0.5px;'>🏜️ UPLOAD DESERT IMAGE</div>")
                input_img = gr.Image(
                    label=None,
                    type="numpy",
                    height=450,
                    elem_classes="image-input",
                    scale=1
                )
                run_btn = gr.Button(
                    "🔍 Analyze Terrain",
                    variant="primary",
                    size="lg",
                    elem_classes="segment-button",
                    scale=1
                )

    # Results section
    with gr.Group(elem_classes="results-container"):
        gr.HTML("<div style='font-size:18px;font-weight:700;color:#2C1810;margin-bottom:20px;letter-spacing:0.5px;'>🗺️ TERRAIN SEGMENTATION MAP</div>")
        
        with gr.Row():
            with gr.Column():
                gr.HTML("<div class='image-section-title'>Segmentation Mask</div>")
                pred_img = gr.Image(label=None, type="numpy", height=380)
            
            with gr.Column():
                gr.HTML("<div class='image-section-title'>Overlay Visualization</div>")
                overlay_img = gr.Image(label=None, type="numpy", height=380)

    # Statistics section
    with gr.Group(elem_classes="stats-container"):
        gr.HTML("<div style='font-size:16px;font-weight:700;color:#2C1810;margin-bottom:15px;letter-spacing:0.5px;'>📊 TERRAIN TYPE DISTRIBUTION</div>")
        class_info = gr.HTML(value="<p style='color:#666;text-align:center;'>Upload a desert image and click 'Analyze Terrain' to see statistics</p>")

    # Color legend
    legend_html = "<div class='legend-grid'>"
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        hex_col = "#{:02x}{:02x}{:02x}".format(*color)
        legend_html += f"""
        <div class='legend-item'>
            <div class='legend-color-box' style='background-color:{hex_col}'></div>
            <span class='legend-label'>{name}</span>
        </div>"""
    legend_html += "</div>"

    with gr.Group(elem_classes="legend-container"):
        gr.HTML("<div style='font-size:16px;font-weight:700;color:#2C1810;margin-bottom:10px;letter-spacing:0.5px;'>🎨 TERRAIN TYPES & COLORS</div>")
        gr.HTML(legend_html)

    # Footer
    gr.HTML("<div class='footer'><p style='color:#666;'><strong>🏜️ Desert Terrain AI</strong> | Built with <strong>Gradio</strong> + <strong>PyTorch</strong> | Duality AI Hackathon 2026</p></div>")

    # Event handler
    run_btn.click(
        fn=segment_image,
        inputs=input_img,
        outputs=[pred_img, overlay_img, class_info]
    )

demo.launch(share=False, server_name="0.0.0.0", server_port=7860)