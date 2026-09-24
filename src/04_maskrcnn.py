from pathlib import Path
import sys, torch
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
)
from .common import load_image, device, save_mask, save_cutout, save_studio, ensure_dir

def run(image_path, out_dir):
    ensure_dir(out_dir)
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    dev = device()
    model = maskrcnn_resnet50_fpn_v2(weights=weights).to(dev).eval()
    image = load_image(image_path)
    x = weights.transforms()(image).to(dev)
    with torch.inference_mode():
        pred = model([x])[0]
    person_id = weights.meta["categories"].index("person")
    found = []
    for i, (label, score) in enumerate(zip(pred["labels"], pred["scores"])):
        if label.item() == person_id and score.item() >= 0.5:
            mask = pred["masks"][i,0].cpu() * 255
            found.append(mask)
            save_mask(mask, Path(out_dir) / f"person_{len(found):02d}_mask.png")
            save_cutout(image, mask, Path(out_dir) / f"person_{len(found):02d}_cutout.png")
            save_studio(image, mask, Path(out_dir) / f"person_{len(found):02d}_studio.jpg")
    if found:
        combined = torch.stack(found).amax(0)
        save_mask(combined, Path(out_dir) / "mask.png")
        save_cutout(image, combined, Path(out_dir) / "cutout.png")
        save_studio(image, combined, Path(out_dir) / "studio.jpg")
    else:
        raise RuntimeError("No person detected above confidence threshold 0.5.")
    return Path(out_dir) / "mask.png"
