"""Augmentation utilities using torchvision

This script provides a function `augment_image(path)` that accepts a file path
to a JPG/JPEG image and writes one augmented image per operation with a suffix
of the form `_(Augmentation name)` before the original extension. Example:

    photo.jpg -> photo_(Rotation).jpg

Operations created:
- Rotation (fixed angle)
- Blur (Gaussian blur)
- Contrast (adjust contrast)
- Scaling (affine scale)
- Illumination (brightness change)
- Projective (perspective / projective transform)

Dependencies: Pillow, torchvision
Install: pip install pillow torchvision

This module is intentionally simple and deterministic (uses fixed params).
Adjust parameters in the TRANSFORM_PARAMS dict below when needed.
"""

from pathlib import Path
from typing import Tuple
import os
from PIL import Image, ImageFilter
import random
import torchvision.transforms.functional as functional
import torchvision.transforms as transforms

# Parameters for transforms (tweak as needed)
TRANSFORM_PARAMS = {
    # These values are used as nominal ranges. Each transform will pick a
    # randomized value on each call derived from these settings.
    "rotation_degrees": 30,            # max abs degrees for Rotation -> angle ~ U(-30,30)
    "blur_radius": 2.0,                # max radius for Gaussian blur -> radius ~ U(0,2)
    "contrast_factor": 1.5,            # nominal multiplier. Contrast ~ U(1/1.5,1.5)
    "scale_factor": 1.2,               # nominal scale. Scale ~ U(1/1.2,1.2)
    "illumination_factor": 1.2,        # nominal brightness multiplier -> U(1/1.2,1.2)
    "projective_distortion_scale": 0.3,# max distortion for projective transform -> U(0,0.3)
}


def _save_image(img: Image.Image, orig_path: Path, suffix: str) -> Path:
    """Save PIL image next to orig_path with suffix inserted before extension.

    Example: /path/photo.jpg with suffix 'Rotation' -> /path/photo_(Rotation).jpg
    """
    # Save outputs to the directory where this script lives, not the source image
    script_dir = Path(__file__).resolve().parent
    stem = orig_path.stem
    ext = orig_path.suffix or ".jpg"
    out_name = f"{stem}_({suffix}){ext}"
    out_path = script_dir / out_name
    img.save(out_path)
    return out_path


def _rotation_transform(img: Image.Image) -> Image.Image:
    """Rotate the image by TRANSFORM_PARAMS['rotation_degrees']."""
    max_deg = float(TRANSFORM_PARAMS["rotation_degrees"])
    angle = random.uniform(-max_deg, max_deg)
    return functional.rotate(img, angle)


def _blur_transform(img: Image.Image) -> Image.Image:
    """Apply Gaussian blur using Pillow."""
    max_radius = float(TRANSFORM_PARAMS["blur_radius"])
    radius = random.uniform(0.0, max_radius)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _contrast_transform(img: Image.Image) -> Image.Image:
    """Adjust contrast using torchvision functional helper."""
    base = float(TRANSFORM_PARAMS["contrast_factor"])
    if base >= 1.0:
        low, high = 1.0 / base, base
    else:
        low, high = base, 1.0 / base
    factor = random.uniform(low, high)
    return functional.adjust_contrast(img, factor)


def _scaling_transform(img: Image.Image) -> Image.Image:
    """Scale image content via affine transform while keeping canvas size."""
    base = float(TRANSFORM_PARAMS["scale_factor"])
    if base >= 1.0:
        low, high = 1.0 / base, base
    else:
        low, high = base, 1.0 / base
    scale = random.uniform(low, high)
    return functional.affine(img, angle=0.0, translate=(0, 0), scale=scale, shear=0.0)


def _illumination_transform(img: Image.Image) -> Image.Image:
    """Adjust brightness (illumination)."""
    base = float(TRANSFORM_PARAMS["illumination_factor"])
    if base >= 1.0:
        low, high = 1.0 / base, base
    else:
        low, high = base, 1.0 / base
    factor = random.uniform(low, high)
    return functional.adjust_brightness(img, factor)


def _projective_transform(img: Image.Image) -> Image.Image:
    """Apply a deterministic projective/perspective transform."""
    max_dist = float(TRANSFORM_PARAMS["projective_distortion_scale"])
    dist = random.uniform(0.0, max_dist)
    proj_t = transforms.RandomPerspective(distortion_scale=dist, p=1.0)
    return proj_t(img)


def augment_image(file_path: str) -> Tuple[Path, ...]:
    """Apply a set of deterministic augmentations to a single JPG/JPEG image.

    Args:
        file_path: Path to an image file (jpg/jpeg). Case-insensitive.

    Returns:
        Tuple of Paths to saved augmented images.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if p.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("Only JPG/JPEG images are supported.")

    img = Image.open(p).convert("RGB")
    w, h = img.size

    saved = []

    # 1) Rotation (fixed angle)
    rotate_img = _rotation_transform(img)
    saved.append(_save_image(rotate_img, p, "Rotation"))

    # 2) Blur (Gaussian)
    blur_img = _blur_transform(img)
    saved.append(_save_image(blur_img, p, "Blur"))

    # 3) Contrast adjustment
    contrast_img = _contrast_transform(img)
    saved.append(_save_image(contrast_img, p, "Contrast"))

    # 4) Scaling (affine with scale parameter)
    scaling_img = _scaling_transform(img)
    saved.append(_save_image(scaling_img, p, "Scaling"))

    # 5) Illumination (brightness change)
    illum_img = _illumination_transform(img)
    saved.append(_save_image(illum_img, p, "Illumination"))

    # 6) Projective (perspective) transform
    proj_img = _projective_transform(img)
    saved.append(_save_image(proj_img, p, "Projective"))

    return tuple(saved)


if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Apply torchvision augmentations to a JPG image and save outputs.")
    # Allow both short -p and long --path flags
    parser.add_argument("-p", "--path", dest="path", required=True, help="Path to a JPG/JPEG image file")
    args = parser.parse_args()

    # Simple console logger for the CLI entrypoint. Libraries shouldn't
    # configure root logging; we do it only when run as a script.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    try:
        outs = augment_image(args.path)
    except Exception:
        logger.exception("Error while augmenting %s", args.path)
        raise
    else:
        logger.info("Saved augmented images:")
        for o in outs:
            logger.info("%s", o)
