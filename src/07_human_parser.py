from pathlib import Path
import sys, torch
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from .common import load_image, device, save_mask, save_cutout, save_studio, ensure_dir

# Current Hugging Face human-parsing model.
MODEL = "fashn-ai/fashn-human-parser"

def run(image_path, out_dir):
    ensure_dir(out_dir)
    dev = device()
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL).to(dev).eval()
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt").to(dev)
    with torch.inference_mode():
        logits = model(**inputs).logits
    logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )[0]
    labels = logits.argmax(0)

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    print("Human-parser labels:", id2label)

    # Human parser's background is normally label 0. Keep every non-background
    # human-part pixel as foreground.
    bg_ids = {k for k,v in id2label.items() if v.lower() in {"background", "bg"}}
    if not bg_ids:
        bg_ids = {0}
    mask = torch.ones_like(labels, dtype=torch.float32) * 255
    for bg in bg_ids:
        mask[labels == bg] = 0

    save_mask(mask, Path(out_dir) / "mask.png")
    save_cutout(image, mask, Path(out_dir) / "cutout.png")
    save_studio(image, mask, Path(out_dir) / "studio.jpg")

    # Save the raw part labels for studying hair/face/clothes/arms/etc.
    import numpy as np
    from PIL import Image
    arr = labels.detach().cpu().numpy()
    rng = np.random.default_rng(123)
    colors = rng.integers(0, 255, size=(max(1, int(arr.max())+1), 3), dtype=np.uint8)
    colors[list(bg_ids)] = 0
    Image.fromarray(colors[arr]).save(Path(out_dir) / "human_parts.png")

    return Path(out_dir) / "mask.png"

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
