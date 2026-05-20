"""ONNX export + tiled inference with PyTorch and ONNX backends. Supports large raster inputs via sliding-window tiling."""

import os, sys, time, logging, argparse, cv2, numpy as np, torch, onnxruntime as ort
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.training.model import get_model
from src.ingestion.dataset import tile_image, merge_tiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TRANSFORM = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def export_onnx(weights_path="best_model.pth", onnx_path="model.onnx"):
    """Export trained PyTorch model to ONNX with dynamic batch axis."""
    model = get_model()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    torch.onnx.export(model, torch.randn(1, 3, 256, 256), onnx_path,
        input_names=["image"], output_names=["mask"],
        dynamic_axes={"image": {0: "batch"}, "mask": {0: "batch"}},
        opset_version=17)
    logger.info(f"ONNX exported -> {onnx_path}")
    return onnx_path


def load_pytorch(weights_path="best_model.pth"):
    model = get_model()
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


def load_onnx(onnx_path="model.onnx"):
    return ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])


def preprocess_tile(tile):
    return TRANSFORM(image=tile)["image"].unsqueeze(0).numpy()


def predict_tile_pytorch(model, tile):
    tensor = torch.from_numpy(preprocess_tile(tile)).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(tensor)).squeeze().cpu().numpy()


def predict_tile_onnx(session, tile):
    inp = preprocess_tile(tile).astype(np.float32)
    logit = session.run(["mask"], {"image": inp})[0]
    return 1 / (1 + np.exp(-logit.squeeze()))


def predict(img_path, out_path=None, backend="onnx",
            weights_path="best_model.pth", onnx_path="model.onnx",
            tile_size=256, overlap=32):
    """Run full inference pipeline: load image, tile, predict per tile, merge, save mask."""
    t0 = time.time()
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if backend == "onnx":
        if not os.path.exists(onnx_path):
            export_onnx(weights_path, onnx_path)
        runner = load_onnx(onnx_path)
        pred_fn = lambda tile: predict_tile_onnx(runner, tile)
    else:
        runner  = load_pytorch(weights_path)
        pred_fn = lambda tile: predict_tile_pytorch(runner, tile)

    tiles, coords = tile_image(img, tile_size=tile_size, overlap=overlap)
    logger.info(f"{os.path.basename(img_path)} ({h}x{w}) -> {len(tiles)} tiles")
    tile_preds = [pred_fn(tile) for tile in tiles]
    full_mask  = merge_tiles(tile_preds, coords, h, w)
    logger.info(f"Done in {time.time()-t0:.3f}s | backend={backend}")

    if out_path is None:
        base     = os.path.splitext(os.path.basename(img_path))[0]
        out_path = f"results/{base}_mask.png"
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    cv2.imwrite(out_path, full_mask * 255)
    return full_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Water body segmentation inference")
    parser.add_argument("img_path")
    parser.add_argument("--out", default=None)
    parser.add_argument("--backend", default="onnx", choices=["pytorch", "onnx"])
    parser.add_argument("--weights", default="best_model.pth")
    parser.add_argument("--onnx", default="model.onnx")
    args = parser.parse_args()
    predict(img_path=args.img_path, out_path=args.out, backend=args.backend,
            weights_path=args.weights, onnx_path=args.onnx)
