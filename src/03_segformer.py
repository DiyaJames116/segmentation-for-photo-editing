from pathlib import Path
import sys, torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from .common import load_image, device, save_mask, save_cutout, save_studio, ensure_dir

MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"

def run(image_path, out_dir):
    ensure_dir(out_dir)
    dev = device()
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL).to(dev).eval()
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt").to(dev)
    with torch.inference_mode():
        logits = model(**inputs).logits
    up = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )[0]
    labels = up.argmax(0)
    id2label = model.config.id2label
    person_ids = {int(k) for k,v in id2label.items() if str(v).lower() == "person"}
    if not person_ids:
        raise RuntimeError(f"Could not find 'person' in model labels: {id2label}")
    mask = torch.zeros_like(labels, dtype=torch.float32)
    for pid in person_ids:
        mask[labels == pid] = 255
    save_mask(mask, Path(out_dir) / "mask.png")
    save_cutout(image, mask, Path(out_dir) / "cutout.png")
    save_studio(image, mask, Path(out_dir) / "studio.jpg")
    return Path(out_dir) / "mask.png"
