from pathlib import Path
import sys
import numpy as np
from PIL import Image
import cv2
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from .common import save_mask, save_cutout, save_studio, ensure_dir

REPO = "DavG25/modnet-pretrained-models"
FILENAME = "models/modnet_photographic_portrait_matting.onnx"

def _model_path():
    return hf_hub_download(repo_id=REPO, filename=FILENAME)


def _providers():
    """Use CUDA when present; otherwise use the portable CPU runtime."""
    available = ort.get_available_providers()
    # The MODNet graph may be only partly supported by CoreML and its provider
    # compilation can fail.  CPU is reliable on macOS, and CUDA remains opt-in
    # when an ONNX Runtime CUDA build is actually installed.
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

def _predict(image):
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    ref = 512

    if max(h, w) < ref or min(h, w) > ref:
        if w >= h:
            nh, nw = ref, int(w / h * ref)
        else:
            nw, nh = ref, int(h / w * ref)
    else:
        nh, nw = h, w

    nw -= nw % 32
    nh -= nh % 32
    nw = max(32, nw)
    nh = max(32, nh)

    x = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = (x - 127.5) / 127.5
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)

    session = ort.InferenceSession(
        _model_path(),
        providers=_providers(),
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    matte = session.run([output_name], {input_name: x})[0][0, 0]
    matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_AREA)
    return np.clip(matte, 0, 1) * 255

def run(image_path, out_dir):
    ensure_dir(out_dir)
    image = Image.open(image_path).convert("RGB")
    mask = _predict(image)
    save_mask(mask, Path(out_dir) / "alpha.png")
    save_cutout(image, mask, Path(out_dir) / "cutout.png")
    save_studio(image, mask, Path(out_dir) / "studio.jpg")
    return Path(out_dir) / "alpha.png"

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
