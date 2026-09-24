from pathlib import Path
import sys, torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from .common import load_image, device, save_mask, save_cutout, save_studio, ensure_dir

MODEL = "facebook/mask2former-swin-small-coco-panoptic"

def run(image_path, out_dir):
    ensure_dir(out_dir)
    dev = device()
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(MODEL).to(dev).eval()
    image = load_image(image_path)
    inputs = processor(images=image, return_tensors="pt").to(dev)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_panoptic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    seg = result["segmentation"].cpu()
    info = result["segments_info"]
    person_ids = {
        s["id"] for s in info
        if str(model.config.id2label[s["label_id"]]).lower() == "person"
    }
    mask = torch.zeros_like(seg, dtype=torch.float32)
    for sid in person_ids:
        mask[seg == sid] = 255
    save_mask(mask, Path(out_dir) / "mask.png")
    save_cutout(image, mask, Path(out_dir) / "cutout.png")
    save_studio(image, mask, Path(out_dir) / "studio.jpg")
    # Save a simple segment-ID visualization.
    import numpy as np
    from PIL import Image
    arr = seg.numpy().astype(np.int64)
    rng = np.random.default_rng(42)
    colors = rng.integers(0, 255, size=(max(1, int(arr.max())+1), 3), dtype=np.uint8)
    vis = colors[arr]
    Image.fromarray(vis).save(Path(out_dir) / "panoptic_segments.png")
    return Path(out_dir) / "mask.png"
