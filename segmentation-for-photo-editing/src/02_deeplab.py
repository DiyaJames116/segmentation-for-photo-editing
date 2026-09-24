from pathlib import Path
import sys, torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from .common import load_image, device, save_mask, save_cutout, save_studio, ensure_dir

def run(image_path, out_dir):
    ensure_dir(out_dir)
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights).to(device()).eval()
    image = load_image(image_path)
    x = weights.transforms()(image).unsqueeze(0).to(device())
    with torch.inference_mode():
        logits = model(x)["out"][0]
    classes = weights.meta["categories"]
    person_id = classes.index("person")
    mask = (logits.argmax(0) == person_id).float() * 255
    save_mask(mask, Path(out_dir) / "mask.png")
    save_cutout(image, mask, Path(out_dir) / "cutout.png")
    save_studio(image, mask, Path(out_dir) / "studio.jpg")
    return Path(out_dir) / "mask.png"
