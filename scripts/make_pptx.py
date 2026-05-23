"""Generate submission presentation for Water Segmentation project."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

DARK = RGBColor(0x1B, 0x2A, 0x4A)
ACCENT = RGBColor(0x2E, 0x86, 0xAB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x60, 0x60, 0x60)


def _add_bg(slide, color=DARK):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _title_box(slide, text, top=Inches(2.5), left=Inches(1), size=Pt(44)):
    txBox = slide.shapes.add_textbox(left, top, Inches(11), Inches(1.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = size
    p.font.color.rgb = WHITE
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER
    return tf


def _subtitle_box(slide, text, top=Inches(4.2), left=Inches(1), size=Pt(18)):
    txBox = slide.shapes.add_textbox(left, top, Inches(11), Inches(1))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = size
    p.font.color.rgb = ACCENT
    p.alignment = PP_ALIGN.CENTER
    return tf


def _slide_title(slide, text):
    txBox = slide.shapes.add_textbox(Inches(0.6), Inches(0.3), Inches(12), Inches(0.8))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(28)
    p.font.color.rgb = DARK
    p.font.bold = True
    return tf


def _bullet_slide(slide, items, top=Inches(1.5), left=Inches(0.8), size=Pt(16)):
    txBox = slide.shapes.add_textbox(left, top, Inches(11.5), Inches(5))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = size
        p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        p.space_after = Pt(8)
        p.level = 0
        if item.startswith("  "):
            p.level = 1
            p.font.size = Pt(14)
            p.font.color.rgb = GRAY
    return tf


# ---------- Slide 1: Title ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(slide)
_title_box(slide, "Water Body Segmentation", top=Inches(2.0), size=Pt(48))
_subtitle_box(slide, "U-Net + ResNet34 | Sentinel-2 Satellite Imagery | PyTorch + ONNX")
_subtitle_box(slide, "Krishna Gupta", top=Inches(5.2), size=Pt(20))

# ---------- Slide 2: Dataset ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
_slide_title(slide, "Dataset: Satellite Images of Water Bodies")
_bullet_slide(slide, [
    "Source: Kaggle, 2,841 RGB image-mask pairs from Sentinel-2",
    "Resolution: ~2009 x 2007 pixels, sourced from Bands 8 (NIR) and 3",
    "  Masks derived via NDWI thresholding",
    "",
    "Key Challenges:",
    "  Class imbalance: water = 7.6%, non-water = 92.4%",
    "  JPEG artifacts in masks: values 1-199 from compression",
    "  Spatial diversity: coastlines, rivers, lakes, wetlands",
    "",
    "Preprocessing: resize to 256x256, mask threshold >200,",
    "  ImageNet normalization, augmentations (flip, rotate, color jitter)",
])

# ---------- Slide 3: Architecture ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
_slide_title(slide, "Model Architecture: U-Net + ResNet34")
_bullet_slide(slide, [
    "Encoder: ResNet34 pretrained on ImageNet (21M params)",
    "Decoder: U-Net with skip connections",
    "Output: Single-channel logit, sigmoid -> binary mask",
    "",
    "Loss: 0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss",
    "Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)",
    "Schedule: CosineAnnealingLR (T_max=50)",
    "Early stopping: patience=10 epochs",
    "",
    "Key Design Decisions:",
    "  BCE + Dice: pixel accuracy + overlap optimization",
    "  ResNet34: balances capacity with GPU memory (batch=8 on 6GB)",
])

# ---------- Slide 4: Results ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
_slide_title(slide, "Performance Results")

# Table
rows, cols = 4, 7
tbl = slide.shapes.add_table(rows, cols, Inches(0.8), Inches(1.5), Inches(11.5), Inches(2.2)).table
headers = ["Split", "IOU", "Accuracy", "Precision", "Recall", "Loss", "Best Epoch"]
data = [
    ["Validation", "0.7865", "0.9213", "0.9028", "0.8621", "0.1812", "34"],
    ["Test", "0.8249", "0.9380", "0.9190", "0.8870", "0.1547", "34"],
    ["Grid Search Best", "0.8018", "0.9265", "0.9093", "0.8696", "0.1710", "50"],
]
for ci, h in enumerate(headers):
    cell = tbl.cell(0, ci)
    cell.text = h
    for p in cell.text_frame.paragraphs:
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = WHITE
    cell.fill.solid()
    cell.fill.fore_color.rgb = DARK
for ri, row in enumerate(data):
    for ci, val in enumerate(row):
        cell = tbl.cell(ri + 1, ci)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        if ri == 0:
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xEB, 0xF5, 0xFB)

_bullet_slide(slide, [
    "Precision > Recall in all runs: model under-segments (safer for flood mapping)",
    "Test outperforms validation: no evidence of overfitting",
    "Grid search: LR=1e-4, batch=8 best; higher LR diverges, lower LR converges slowly",
], top=Inches(4.2), size=Pt(14))

# ---------- Slide 5: Optimization ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
_slide_title(slide, "Inference Optimizations")

rows, cols = 4, 4
tbl = slide.shapes.add_table(rows, cols, Inches(0.8), Inches(1.5), Inches(7), Inches(2)).table
headers2 = ["Backend", "Device", "Latency/tile", "Speedup"]
data2 = [
    ["PyTorch", "CPU", "~120ms", "1.0x"],
    ["ONNX Runtime", "CPU", "~55ms", "~2.2x"],
    ["ONNX Runtime", "GPU", "~15ms", "~8.0x"],
]
for ci, h in enumerate(headers2):
    cell = tbl.cell(0, ci)
    cell.text = h
    for p in cell.text_frame.paragraphs:
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = WHITE
    cell.fill.solid()
    cell.fill.fore_color.rgb = DARK
for ri, row in enumerate(data2):
    for ci, val in enumerate(row):
        cell = tbl.cell(ri + 1, ci)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

_bullet_slide(slide, [
    "Sliding-window tiling: 256x256 tiles, 32px overlap for large rasters (~72 tiles/image)",
    "Test-Time Augmentation: horizontal flip averaging (+0.5-1.0% IOU)",
    "Dynamic batch axis in ONNX export for flexible input sizing",
    "Multi-backend fallback: ONNX -> PyTorch -> MLflow registry",
], top=Inches(4.0), left=Inches(0.8), size=Pt(15))

# ---------- Slide 6: Failure Analysis ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
_slide_title(slide, "Failure Analysis & Known Limitations")

rows, cols = 4, 3
tbl = slide.shapes.add_table(rows, cols, Inches(0.8), Inches(1.5), Inches(11.5), Inches(2.2)).table
headers3 = ["Failure Mode", "Impact", "Root Cause"]
data3 = [
    ["Narrow waterways", "Streams <4px wide missed", "256x256 resize destroys thin features"],
    ["Shadow/dark terrain", "Shadows classified as water", "RGB-only input; NIR band needed"],
    ["JPEG boundary noise", "Ragged mask edges", "Compression artifacts in source masks"],
]
for ci, h in enumerate(headers3):
    cell = tbl.cell(0, ci)
    cell.text = h
    for p in cell.text_frame.paragraphs:
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = WHITE
    cell.fill.solid()
    cell.fill.fore_color.rgb = DARK
for ri, row in enumerate(data3):
    for ci, val in enumerate(row):
        cell = tbl.cell(ri + 1, ci)
        cell.text = val
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(13)
            p.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

_bullet_slide(slide, [
    "JPEG artifacts affect 95.6% of masks; threshold >=200 eliminates most",
    "RGB-only is the biggest accuracy bottleneck -- NIR band would help most",
    "Failure modes are systematic and predictable, not random noise",
], top=Inches(4.2), size=Pt(14))

# ---------- Slide 7: MLOps ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
_slide_title(slide, "MLOps & Deployment")

_bullet_slide(slide, [
    "Version Control: Git + Git LFS (model.onnx, 97 MB); .gitignore for artifacts",
    "Experiment Tracking: MLflow logs params, metrics, model artifacts per run",
    "Model Registry: 'water-segmentation-unet' v1 with serialization_format=pt2",
    "Data Versioning: SHA256 dataset hash logged with each training run",
    "",
    "CI/CD (GitHub Actions):",
    "  Push -> flake8 lint -> 15 pytest -> Docker build -> Docker Hub push",
    "",
    "Containerization:",
    "  Multi-stage Docker build, ONNX model baked in, HEALTHCHECK every 10s",
    "  docker-compose with API + MLflow server services",
    "",
    "Security: path traversal protection, 100 MB upload cap,",
    "  60 req/min/IP rate limiter, thread-safe model cache",
])

# ---------- Slide 8: Next Steps ----------
slide = prs.slides.add_slide(prs.slide_layouts[6])
_slide_title(slide, "Recommendations & Next Steps")
_bullet_slide(slide, [
    "1. Train on full-resolution crops (not resized) -- expected +5-8% IOU",
    "2. Add NIR band input -- single biggest expected gain (+5-10% IOU)",
    "3. EfficientNet-B4 encoder -- higher capacity, +2-3% IOU",
    "4. CRF post-processing -- sharpen boundaries, fix JPEG edge noise",
    "5. Optuna Bayesian search -- replace grid with 30+ trial optimization",
    "6. Ensemble 3-5 runs -- reduce variance, +1-2% IOU",
    "",
    "Key Takeaway:",
    "  The current model (IOU 0.82 test) is production-ready for coarse",
    "  water mapping. Adding NIR and full-resolution training would close",
    "  the gap to high-precision applications.",
])

# Save
prs.save("presentation.pptx")
print("Presentation saved to presentation.pptx")
