from pathlib import Path
import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from .common import ensure_dir, save_mask, save_cutout, save_studio


SAM_REPO = "facebook/sam2.1-hiera-small"
CHECKPOINT_NAME = "sam2.1_hiera_small.pt"

# This is the config that comes with the Meta SAM2 installation.
CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_s.yaml"


def _checkpoint():
    """Download the SAM2.1 checkpoint from Hugging Face if needed."""
    return hf_hub_download(
        repo_id=SAM_REPO,
        filename=CHECKPOINT_NAME,
    )


def _device():
    """Use Apple MPS on M1/M2/M3 Macs, otherwise CPU/CUDA."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def run(image_path, out_dir, point=(0.5, 0.45)):
    ensure_dir(out_dir)

    try:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2
    except Exception as e:
        raise RuntimeError(
            "Install Meta SAM 2 first. See README.md."
        ) from e

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = _device()

    print(f"[SAM2] Using device: {device}")

    # ---------------------------------------------------------
    # Download checkpoint
    # ---------------------------------------------------------
    checkpoint = _checkpoint()

    print(f"[SAM2] Checkpoint: {checkpoint}")
    print(f"[SAM2] Config: {CONFIG_NAME}")

    # ---------------------------------------------------------
    # Build SAM2 model
    #
    # IMPORTANT:
    # CONFIG_NAME is relative to SAM2's config directory.
    # Do NOT pass the Hugging Face absolute YAML path here.
    # ---------------------------------------------------------
    model = build_sam2(
        CONFIG_NAME,
        checkpoint,
        device=device,
    )

    predictor = SAM2ImagePredictor(model)

    # ---------------------------------------------------------
    # Load image
    # ---------------------------------------------------------
    image = np.array(
        Image.open(image_path).convert("RGB")
    )

    predictor.set_image(image)

    # ---------------------------------------------------------
    # Convert normalized point (0-1) to pixel coordinates
    # ---------------------------------------------------------
    h, w = image.shape[:2]

    coords = np.array(
        [[point[0] * w, point[1] * h]],
        dtype=np.float32,
    )

    labels = np.array(
        [1],
        dtype=np.int32,
    )

    print(
        f"[SAM2] Point prompt: "
        f"normalized=({point[0]:.3f}, {point[1]:.3f}), "
        f"pixel=({coords[0][0]:.1f}, {coords[0][1]:.1f})"
    )

    # ---------------------------------------------------------
    # Generate masks
    # ---------------------------------------------------------
    masks, scores, _ = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=True,
    )

    # ---------------------------------------------------------
    # Select highest scoring mask
    # ---------------------------------------------------------
    best_index = int(np.argmax(scores))
    best = masks[best_index]

    print(
        f"[SAM2] Generated {len(masks)} masks"
    )

    print(
        f"[SAM2] Selected mask {best_index} "
        f"with score {scores[best_index]:.4f}"
    )

    # ---------------------------------------------------------
    # Convert boolean mask -> 0/255 grayscale image
    # ---------------------------------------------------------
    mask = Image.fromarray(
        (best.astype(np.uint8) * 255)
    )

    save_mask(
        mask,
        Path(out_dir) / "mask.png",
    )

    # ---------------------------------------------------------
    # Generate cutout
    # ---------------------------------------------------------
    im = Image.fromarray(image)

    save_cutout(
        im,
        mask,
        Path(out_dir) / "cutout.png",
    )

    # ---------------------------------------------------------
    # Generate navy studio background
    # ---------------------------------------------------------
    save_studio(
        im,
        mask,
        Path(out_dir) / "studio.jpg",
    )

    # ---------------------------------------------------------
    # Save all SAM2 scores
    # ---------------------------------------------------------
    with open(
        Path(out_dir) / "scores.txt",
        "w",
    ) as f:
        f.write(
            "\n".join(
                map(str, scores.tolist())
            )
        )

    print(
        f"[SAM2] Outputs saved to {out_dir}"
    )

    return Path(out_dir) / "mask.png"