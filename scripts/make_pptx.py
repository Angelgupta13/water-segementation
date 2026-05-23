"""Generate submission presentation for Water Segmentation project."""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

NAVY = RGBColor(0x1B, 0x2A, 0x4A)
ACCENT = RGBColor(0x2E, 0x86, 0xAB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TEXT = RGBColor(0x33, 0x33, 0x33)
GRAY = RGBColor(0x80, 0x80, 0x80)
LIGHT_BG = RGBColor(0xF5, 0xF7, 0xFA)
ROW_ALT = RGBColor(0xEB, 0xF5, 0xFB)


def _dark_bg(slide):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = NAVY


def _light_bg(slide):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = LIGHT_BG


def _add_shape(slide, left, top, width, height, color):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape


def _center_text(slide, text, top, size=Pt(44), color=WHITE, bold=True, left=Inches(1), width=Inches(11.333)):
    txBox = slide.shapes.add_textbox(left, top, width, Inches(0.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = size
    p.font.color.rgb = color
    p.font.bold = bold
    p.alignment = PP_ALIGN.CENTER
    return tf


def _slide_title(slide, text):
    _center_text(slide, text, Inches(0.3), Pt(28), DARK_TEXT, True, Inches(0.6), Inches(12))


def _bullets(slide, items, top=Inches(1.5), left=Inches(0.8), size=Pt(16), color=DARK_TEXT):
    txBox = slide.shapes.add_textbox(left, top, Inches(11.5), Inches(5))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.font.size = size
        p.font.color.rgb = color
        p.space_after = Pt(6)
        if item.startswith("  "):
            p.level = 1
            p.font.size = Pt(14)
            p.font.color.rgb = GRAY
    return tf


def _make_table(slide, headers, rows_data, left, top, width, height):
    rows = len(rows_data) + 1
    cols = len(headers)
    tbl = slide.shapes.add_table(rows, cols, left, top, width, height).table
    for ci, h in enumerate(headers):
        cell = tbl.cell(0, ci)
        cell.text = h
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = WHITE
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
    for ri, row in enumerate(rows_data):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri + 1, ci)
            cell.text = val
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(13)
                p.font.color.rgb = DARK_TEXT
            if ri % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = ROW_ALT
    return tbl


# ===== SLIDE 1: Title =====
slide = prs.slides.add_slide(prs.slide_layouts[6])
_dark_bg(slide)
_add_shape(slide, Inches(0), Inches(3.2), Inches(13.333), Inches(0.04), ACCENT)
_center_text(slide, "Water Body Segmentation", Inches(1.5), Pt(48), WHITE, True)
_center_text(slide, "U-Net + ResNet34  |  Sentinel-2 Satellite Imagery  |  PyTorch + ONNX", Inches(3.6), Pt(18), ACCENT)
_center_text(slide, "Krishna Gupta", Inches(5.5), Pt(20), RGBColor(0xBB, 0xBB, 0xBB), False)

# ===== SLIDE 2: Problem & Dataset =====
slide = prs.slides.add_slide(prs.slide_layouts[6])
_light_bg(slide)
_slide_title(slide, "Problem: Segment Water from Satellite Imagery")
_bullets(slide, [
    "2,841 Sentinel-2 images with binary water/non-water masks (~2009x2007 px)",
    "Key challenges:",
    "  Class imbalance: 7.6% water vs 92.4% non-water pixels",
    "  JPEG artifacts: 96% of masks have compression noise (values 1-199)",
    "  RGB-only input -- no NIR band to distinguish water from shadow",
    "Data split: 70% train (1,989), 15% val (426), 15% test (426)",
    "Preprocessing: resize 256x256, ImageNet norm, aug (flip + rotate + color)",
], top=Inches(1.4), size=Pt(16))

# ===== SLIDE 3: Architecture =====
slide = prs.slides.add_slide(prs.slide_layouts[6])
_light_bg(slide)
_slide_title(slide, "Model: U-Net + ResNet34")
_bullets(slide, [
    "Encoder: ResNet34 (21M params), pretrained on ImageNet",
    "Decoder: U-Net with spatial skip connections",
    "Loss: 0.5 x BCE + 0.5 x Dice (hybrid pixel + overlap optimization)",
    "Optimizer: AdamW (lr=1e-4, wd=1e-4) + CosineAnnealingLR",
    "Early stopping: patience=10 epochs on val IOU",
], top=Inches(1.4), size=Pt(16))
_make_table(slide,
    ["Hyperparameter", "Best Value"],
    [["Learning rate", "1e-4"], ["Batch size", "8"], ["Epochs", "50 (stopped at 34)"]],
    Inches(0.8), Inches(4.2), Inches(5), Inches(1.8))

# ===== SLIDE 4: Results =====
slide = prs.slides.add_slide(prs.slide_layouts[6])
_light_bg(slide)
_slide_title(slide, "Performance")
_center_text(slide, "Grid Search (top 3 of 6)", Inches(1.2), Pt(18), DARK_TEXT, True, Inches(0.8), Inches(5.5))
_make_table(slide,
    ["LR", "Batch", "IOU", "Accuracy"],
    [["1e-4", "8", "0.8018", "92.65%"],
     ["5e-5", "8", "0.7928", "92.31%"],
     ["1e-4", "16", "0.7542", "90.86%"]],
    Inches(0.8), Inches(1.9), Inches(5.5), Inches(2))

_center_text(slide, "Final Model (Test Set)", Inches(1.2), Pt(18), DARK_TEXT, True, Inches(7), Inches(5.5))
_make_table(slide,
    ["IOU", "Accuracy", "Precision", "Recall"],
    [["0.8249", "93.80%", "91.90%", "88.70%"]],
    Inches(7), Inches(1.9), Inches(5.5), Inches(1.2))

_bullets(slide, [
    "Precision > Recall across all runs -- model under-segments",
    "  Safer bias for flood mapping: fewer false alarms",
    "Test IOU > Val IOU: no overfitting",
], top=Inches(4.4), size=Pt(15))

# ===== SLIDE 5: Inference Optimizations =====
slide = prs.slides.add_slide(prs.slide_layouts[6])
_light_bg(slide)
_slide_title(slide, "Optimization: 2.2x CPU Speedup with ONNX")
_make_table(slide,
    ["Backend", "Device", "Latency / tile", "Speedup"],
    [["PyTorch (eager)", "CPU", "~120 ms", "1.0x"],
     ["ONNX Runtime", "CPU", "~55 ms", "2.2x"],
     ["ONNX Runtime", "GPU", "~15 ms", "8.0x"]],
    Inches(0.8), Inches(1.5), Inches(7), Inches(2))
_bullets(slide, [
    "Sliding-window tiling: 256x256 tiles, 32px overlap (~72 tiles/image)",
    "Test-Time Augmentation: horizontal flip averaging (+0.5-1% IOU)",
    "Auto-fallback: ONNX -> PyTorch -> MLflow registry",
    "Dynamic batch axis for flexible input sizing",
], top=Inches(4.0), size=Pt(15))

# ===== SLIDE 6: Failure Analysis =====
slide = prs.slides.add_slide(prs.slide_layouts[6])
_light_bg(slide)
_slide_title(slide, "What the Model Misses (Transparency)")
_make_table(slide,
    ["Failure Mode", "Impact", "Root Cause"],
    [["Narrow waterways", "Streams <4px wide missed", "256x256 resize destroys thin features"],
     ["Shadow / dark terrain", "Shadows classified as water", "RGB-only input; NIR disambiguates"],
     ["JPEG boundary noise", "Ragged mask edges (+/-1px)", "Compression artifacts in source masks"]],
    Inches(0.8), Inches(1.5), Inches(11.5), Inches(2.2))
_bullets(slide, [
    "Production-ready for: lakes, large rivers, reservoirs",
    "Not suitable for: narrow canals, small ponds (<500 m2), shadow-prone terrain",
    "Biggest bottleneck: RGB-only input -- adding NIR band would give +5-10% IOU",
], top=Inches(4.2), size=Pt(15))

# ===== SLIDE 7: Deployment & MLOps =====
slide = prs.slides.add_slide(prs.slide_layouts[6])
_light_bg(slide)
_slide_title(slide, "MLOps Pipeline")
_center_text(slide, "Git Push  ->  Flake8 Lint  ->  15 Pytest  ->  Docker Build  ->  Docker Hub Push", Inches(1.3), Pt(16), ACCENT, True)
_add_shape(slide, Inches(0.8), Inches(2.5), Inches(11.5), Inches(0.03), ACCENT)
_bullets(slide, [
    "Docker: multi-stage build, ONNX baked in, HEALTHCHECK every 10s",
    "MLflow: per-epoch metrics, model registry (water-segmentation-unet v1)",
    "Security: 60 req/min rate limiter, 100 MB cap, path traversal protection",
    "Docker Hub: angelgupta/water-segmentation:latest",
    "docker-compose: API server + MLflow tracking server",
], top=Inches(2.8), size=Pt(16))

# ===== SLIDE 8: Recommendations =====
slide = prs.slides.add_slide(prs.slide_layouts[6])
_light_bg(slide)
_slide_title(slide, "Recommendations & Next Steps")

# High impact
_add_shape(slide, Inches(0.6), Inches(1.3), Inches(3.8), Inches(0.5), NAVY)
_center_text(slide, "HIGH IMPACT (+5-10% IOU)", Inches(1.3), Pt(14), WHITE, True, Inches(0.6), Inches(3.8))
_bullets(slide, [
    "Add NIR band input (Band 8)",
    "Train on full-resolution crops",
], top=Inches(2.0), left=Inches(0.6), size=Pt(15))

# Medium impact
_add_shape(slide, Inches(4.8), Inches(1.3), Inches(3.8), Inches(0.5), ACCENT)
_center_text(slide, "MEDIUM IMPACT (+1-3% IOU)", Inches(1.3), Pt(14), WHITE, True, Inches(4.8), Inches(3.8))
_bullets(slide, [
    "EfficientNet-B4 encoder",
    "Optuna Bayesian search (30+ trials)",
], top=Inches(2.0), left=Inches(4.8), size=Pt(15))

# Low impact
_add_shape(slide, Inches(9), Inches(1.3), Inches(3.8), Inches(0.5), GRAY)
_center_text(slide, "LOW IMPACT (+0.5-1% IOU)", Inches(1.3), Pt(14), WHITE, True, Inches(9), Inches(3.8))
_bullets(slide, [
    "Ensemble (3-5 seeds)",
    "CRF post-processing",
], top=Inches(2.0), left=Inches(9), size=Pt(15))

_center_text(slide, "Current model (IOU 0.82) is production-ready for coarse water mapping.", Inches(4.0), Pt(18), DARK_TEXT, False)
_center_text(slide, "Adding NIR band and full-resolution training would close the gap to high-precision applications.", Inches(4.8), Pt(18), GRAY, False)

# Save
import sys
out_path = sys.argv[1] if len(sys.argv) > 1 else "presentation.pptx"
prs.save(out_path)
print(f"Presentation saved to {out_path}")
