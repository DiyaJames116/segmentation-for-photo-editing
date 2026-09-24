from pathlib import Path

import numpy as np
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from .common import ensure_dir, save_mask


SAM_REPO = "facebook/sam2.1-hiera-small"

# Hydra configuration bundled with the installed SAM2 package
CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Actual model weights downloaded from Hugging Face
CHECKPOINT_NAME = "sam2.1_hiera_small.pt"


def _checkpoint():
    """Download SAM2.1 checkpoint if it is not already cached."""
    return hf_hub_download(
        repo_id=SAM_REPO,
        filename=CHECKPOINT_NAME,
    )


def _device():
    """Select the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def run(image_path, out_dir):
    ensure_dir(out_dir)

    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

    except Exception as e:
        raise RuntimeError(
            "Install Meta SAM 2 first. See README.md."
        ) from e

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = _device()

    print(f"[SAM2 Auto] Using device: {device}")

    # ---------------------------------------------------------
    # Get checkpoint
    #
    # IMPORTANT:
    # We download ONLY the .pt checkpoint from Hugging Face.
    # The YAML config comes from the installed SAM2 package.
    # ---------------------------------------------------------
    checkpoint = _checkpoint()

    print(f"[SAM2 Auto] Checkpoint: {checkpoint}")
    print(f"[SAM2 Auto] Config: {CONFIG_NAME}")

    # ---------------------------------------------------------
    # Build SAM2
    # ---------------------------------------------------------
    model = build_sam2(
        CONFIG_NAME,
        checkpoint,
        device=device,
    )

    # ---------------------------------------------------------
    # Create automatic mask generator
    # ---------------------------------------------------------
    generator = SAM2AutomaticMaskGenerator(model)

    # ---------------------------------------------------------
    # Load image
    # ---------------------------------------------------------
    image = np.array(
        Image.open(image_path).convert("RGB")
    )

    print(
        f"[SAM2 Auto] Image size: "
        f"{image.shape[1]}x{image.shape[0]}"
    )

    print("[SAM2 Auto] Generating masks...")

    # ---------------------------------------------------------
    # Generate masks automatically
    # ---------------------------------------------------------
    masks = generator.generate(image)

    # Largest masks first
    masks = sorted(
        masks,
        key=lambda x: x["area"],
        reverse=True,
    )

    print(
        f"[SAM2 Auto] Generated {len(masks)} masks"
    )

    # ---------------------------------------------------------
    # Save top 30 masks
    # ---------------------------------------------------------
    for i, item in enumerate(masks[:30], start=1):

        mask_array = (
            item["segmentation"].astype(np.uint8) * 255
        )

        mask_image = Image.fromarray(mask_array)

        save_mask(
            mask_image,
            Path(out_dir) / f"mask_{i:02d}.png",
        )

    # ---------------------------------------------------------
    # Build combined mask of larger objects
    # ---------------------------------------------------------
    if masks:

        combined = np.zeros(
            image.shape[:2],
            dtype=np.uint8,
        )

        image_area = (
            image.shape[0] * image.shape[1]
        )

        for item in masks:

            # Keep segments covering >1% of the image
            if item["area"] > image_area * 0.01:

                combined |= item[
                    "segmentation"
                ].astype(np.uint8)

        combined_image = Image.fromarray(
            combined * 255
        )

        output_path = (
            Path(out_dir)
            / "combined_large_objects.png"
        )

        save_mask(
            combined_image,
            output_path,
        )

        print(
            f"[SAM2 Auto] Saved masks to {out_dir}"
        )

        return output_path

    raise RuntimeError(
        "SAM 2 automatic mask generator returned no masks."
    )