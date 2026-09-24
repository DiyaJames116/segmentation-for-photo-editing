from pathlib import Path
import sys, torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image
from .common import device, save_mask, save_cutout, save_studio, ensure_dir

MODEL = "ZhengPeng7/BiRefNet"

def run(image_path, out_dir):
    ensure_dir(out_dir)
    dev = device()
    model = AutoModelForImageSegmentation.from_pretrained(
        MODEL, trust_remote_code=True
    )
    # The downloaded checkpoint can be stored in FP16.  On CPU (and on some
    # Apple setups) the image transform produces FP32 tensors, which otherwise
    # causes PyTorch to reject the FP32 input against an FP16 convolution bias.
    model = model.to(device=dev, dtype=torch.float32).eval()
    image = Image.open(image_path).convert("RGB")
    x = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])(image).unsqueeze(0).to(device=dev, dtype=torch.float32)
    with torch.inference_mode():
        pred = model(x)
        if isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif hasattr(pred, "logits"):
            pred = pred.logits
    if pred.ndim == 4:
        pred = pred[:,0:1]
    pred = torch.nn.functional.interpolate(
        pred, size=image.size[::-1], mode="bilinear", align_corners=False
    )[0,0]
    mask = pred.sigmoid().clamp(0,1) * 255 if pred.min() < 0 or pred.max() > 1 else pred.clamp(0,1) * 255
    save_mask(mask, Path(out_dir) / "mask.png")
    save_cutout(image, mask, Path(out_dir) / "cutout.png")
    save_studio(image, mask, Path(out_dir) / "studio.jpg")
    return Path(out_dir) / "mask.png"
