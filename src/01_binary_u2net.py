from pathlib import Path
import sys
from rembg import remove, new_session
from PIL import Image
from .common import save_mask, save_cutout, save_studio, ensure_dir

def run(image_path, out_dir):
    ensure_dir(out_dir)
    image = Image.open(image_path).convert("RGB")
    session = new_session("u2net")
    rgba = remove(image, session=session)
    mask = rgba.getchannel("A")
    save_mask(mask, Path(out_dir) / "mask.png")
    rgba.save(Path(out_dir) / "cutout.png")
    save_studio(image, mask, Path(out_dir) / "studio.jpg")
    return Path(out_dir) / "mask.png"

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
