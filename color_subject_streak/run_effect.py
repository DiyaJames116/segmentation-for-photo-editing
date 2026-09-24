"""Keep the subject in colour over a monochrome, directionally blurred background."""
from __future__ import annotations

import argparse
import importlib
import logging
import math
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT.parent
EXTENSIONS = {'.heic', '.heif', '.jpg', '.jpeg', '.png', '.tif', '.tiff'}
LOG = logging.getLogger(__name__)


def load_image(path: Path) -> Image.Image:
    if path.suffix.lower() in {'.heic', '.heif'}:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as exc:
            raise RuntimeError('HEIC requires pillow-heif; install requirements.txt in your environment.') from exc
        register_heif_opener()
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source).convert('RGBA')
        # This effect produces a finished opaque photograph.
        canvas = Image.new('RGBA', oriented.size, (255, 255, 255, 255))
        return Image.alpha_composite(canvas, oriented).convert('RGB')


def newest_input() -> Path:
    folder = ROOT / 'input'
    candidates = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS] if folder.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f'Put an image in {folder}, or pass --image PATH.')
    return max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name))


def validate_options(blur: float, angle: float) -> None:
    if not math.isfinite(blur) or not 0 <= blur <= 0.5:
        raise ValueError('Blur must be between 0 and 0.5 (a fraction of the shorter image dimension).')
    if not math.isfinite(angle):
        raise ValueError('Angle must be a finite number of degrees.')


def motion_kernel(length: int, angle: float) -> np.ndarray:
    """0 degrees is horizontal; 90 is vertical; positive angles slope down-right."""
    length = max(1, length | 1)
    kernel = np.zeros((length, length), np.uint8)
    center = length // 2
    theta = math.radians(angle % 180)
    dx, dy = center * math.cos(theta), center * math.sin(theta)
    cv2.line(kernel, (round(center - dx), round(center - dy)),
             (round(center + dx), round(center + dy)), 255, 1, cv2.LINE_AA)
    weights = kernel.astype(np.float32)
    return weights / weights.sum()


def apply_effect(image: Image.Image, mask: Image.Image, blur: float = 0.08, angle: float = -25) -> Image.Image:
    """Composite untouched colour pixels through the subject's soft alpha matte."""
    validate_options(blur, angle)
    if image.size != mask.size:
        raise ValueError('Mask dimensions must match the oriented input image.')
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    alpha = np.asarray(mask.convert('L'), dtype=np.float32) / 255
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    length = min(501, max(1, round(min(image.size) * blur)))
    kernel = motion_kernel(length, angle)
    # Exclude the subject from the blur samples, preventing a streaked silhouette
    # from spilling into the background. Normalize for missing background samples.
    background_weight = np.where(alpha < 0.05, 1 - alpha, 0).astype(np.float32)
    numerator = cv2.filter2D(gray * background_weight, -1, kernel, borderType=cv2.BORDER_REFLECT)
    denominator = cv2.filter2D(background_weight, -1, kernel, borderType=cv2.BORDER_REFLECT)
    background = gray.copy()
    np.divide(numerator, denominator, out=background, where=denominator > 1e-5)
    if blur == 0:
        background = gray
    result = rgb * alpha[..., None] + background[..., None] * (1 - alpha[..., None])
    return Image.fromarray(np.clip(result, 0, 255).round().astype(np.uint8))


def extract_mask(image_path: Path, destination: Path) -> Path:
    if str(MODEL_ROOT) not in sys.path:
        sys.path.insert(0, str(MODEL_ROOT))
    model = importlib.import_module('src.06_birefnet')
    model.run(str(image_path), str(destination))
    return destination / 'mask.png'


def run_effect(image: str | Path | None = None, output: str | Path | None = None,
               blur: float = 0.08, angle: float = -25, mask: str | Path | None = None) -> Path:
    """Run segmentation and compositing; optionally reuse an existing L-mode mask."""
    validate_options(blur, angle)
    source = Path(image).expanduser().resolve() if image is not None else newest_input()
    destination = Path(output).expanduser().resolve() if output is not None else ROOT / 'output' / f'{source.stem}_color_streak.jpg'
    mask_path = Path(mask).expanduser().resolve() if mask is not None else None
    if destination.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
        raise ValueError('Output must end in .jpg, .jpeg, or .png.')
    if destination.resolve() == source.resolve() or (mask_path is not None and destination.resolve() == mask_path):
        raise ValueError('Output must not overwrite an input image or mask.')
    normalized = load_image(source)
    if mask_path is not None:
        with Image.open(mask_path) as supplied:
            matte = supplied.convert('L')
        if matte.size != normalized.size:
            raise ValueError('Mask dimensions must match the oriented input image.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.color-streak-', dir=destination.parent) as temp:
        staging = Path(temp)
        if mask_path is None:
            LOG.info('[1/3] Finding the subject with BiRefNet...')
            normalized_path = staging / 'input.png'
            normalized.save(normalized_path)
            generated_mask = extract_mask(normalized_path, staging / 'segmentation')
            with Image.open(generated_mask) as generated:
                matte = generated.convert('L')
        else:
            LOG.info('[1/3] Using supplied subject mask...')
        LOG.info('[2/3] Making the background monochrome with directional blur...')
        result = apply_effect(normalized, matte, blur, angle)
        LOG.info('[3/3] Saving finished image...')
        staged_output = staging / destination.name
        if destination.suffix.lower() == '.png':
            result.save(staged_output)
        else:
            result.save(staged_output, quality=95, subsampling=0)
        staged_output.replace(destination)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', help='HEIC/JPG/JPEG/PNG/TIFF input; default: newest image in this feature’s input folder.')
    parser.add_argument('--output', help='Finished .jpg/.jpeg/.png; default: output/<image>_color_streak.jpg.')
    parser.add_argument('--blur', type=float, default=0.08, help='Streak length as a fraction of the shorter side, 0–0.5; default 0.08, capped at 501 pixels.')
    parser.add_argument('--angle', type=float, default=-25, help='Blur angle in degrees: 0 horizontal, 90 vertical; default -25.')
    parser.add_argument('--mask', help='Optional grayscale subject mask: white subject, black background; skips BiRefNet.')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    try:
        result = run_effect(args.image, args.output, args.blur, args.angle, args.mask)
    except Exception as exc:
        LOG.error('Effect failed: %s', exc, exc_info=args.debug)
        return 1
    print(f'Finished image: {result}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
