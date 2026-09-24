from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch

NAVY = (16, 27, 55)

def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_image(path):
    return Image.open(path).convert("RGB")

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def normalize_mask(mask):
    """Return a PIL L mask from ndarray/PIL/tensor."""
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().float().numpy()
    if isinstance(mask, Image.Image):
        im = mask.convert("L")
        if im.getextrema()[1] <= 1:
            im = im.point(lambda p: int(p * 255))
        return im
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr.squeeze()
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= 1.0:
        arr *= 255.0
    elif hi > lo:
        # Don't rescale already-valid 0..255 masks.
        if lo < 0 or hi > 255:
            arr = (arr - lo) / (hi - lo) * 255.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "L")

def save_cutout(image, mask, path):
    mask = normalize_mask(mask).resize(image.size, Image.Resampling.LANCZOS)
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    rgba.save(path)

def save_studio(image, mask, path, bg=NAVY):
    """Save a foreground composite, flattening alpha when the target is JPEG."""
    mask = normalize_mask(mask).resize(image.size, Image.Resampling.LANCZOS)
    background = Image.new("RGB", image.size, bg).convert("RGBA")
    fg = image.convert("RGBA")
    fg.putalpha(mask)
    out = Image.alpha_composite(background, fg)
    # JPEG has no alpha channel.  The previous version attempted to write this
    # RGBA image directly, which stopped every model after its mask/cutout had
    # already been created.
    if Path(path).suffix.lower() in {".jpg", ".jpeg"}:
        out = out.convert("RGB")
    out.save(path, quality=95)

def save_mask(mask, path):
    normalize_mask(mask).save(path)

def make_comparison(entries, output_path, thumb_size=(420, 300), columns=3, bg=NAVY):
    """Create a contact sheet from ``(title, image_path)`` entries.

    Transparent cutouts are composited over the same studio background used by
    ``save_studio``.  Simply converting an RGBA PNG to RGB discards alpha and
    makes every cutout appear to be the unsegmented source image.
    """
    if not entries:
        return
    rows = (len(entries) + columns - 1) // columns
    canvas = Image.new("RGB",
                       (columns * thumb_size[0], rows * (thumb_size[1] + 35)),
                       "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, image_path) in enumerate(entries):
        r, c = divmod(i, columns)
        box = (c * thumb_size[0], r * (thumb_size[1] + 35))
        with Image.open(image_path) as source:
            if "A" in source.getbands():
                layer = source.convert("RGBA")
                studio_bg = Image.new("RGBA", layer.size, bg + (255,))
                im = Image.alpha_composite(studio_bg, layer).convert("RGB")
            else:
                im = source.convert("RGB")
        im = ImageOps.contain(im, thumb_size)
        x = box[0] + (thumb_size[0] - im.width) // 2
        y = box[1]
        canvas.paste(im, (x, y))
        draw.text((box[0] + 8, box[1] + thumb_size[1] + 7), title, fill="black")
    ensure_dir(Path(output_path).parent)
    canvas.save(output_path, quality=95)
