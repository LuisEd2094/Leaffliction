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
import shutil
import time
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


def _save_image(img: Image.Image, orig_path: Path, suffix: str, out_dir: Path = None) -> Path:
    """Save PIL image next to orig_path with suffix inserted before extension.

    Example: /path/photo.jpg with suffix 'Rotation' -> /path/photo_(Rotation).jpg
    """
    # Determine output directory: prefer explicit out_dir, otherwise script dir
    if out_dir is None:
        out_base = Path(__file__).resolve().parent
    else:
        out_base = Path(out_dir)
    stem = orig_path.stem
    ext = orig_path.suffix or ".jpg"
    out_name = f"{stem}_({suffix}){ext}"
    out_path = out_base / out_name
    # Avoid overwriting an existing file: append a numeric suffix if needed
    if out_path.exists():
        i = 1
        while True:
            alt_name = f"{stem}_({suffix})_{i}{ext}"
            alt_path = out_base / alt_name
            if not alt_path.exists():
                out_path = alt_path
                break
            i += 1
    img.save(out_path)
    return out_path


def _list_images_in_dir(dir_path: Path):
    """Return list of jpg/jpeg image Paths in the given directory (non-recursive)."""
    imgs = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
    return imgs


def _create_augmented_from_image(src_path: Path, out_dir: Path = None) -> Path:
    """Create a single augmented image from src_path by applying a random subset of transforms sequentially.

    Returns the saved Path object.
    """
    img = Image.open(src_path).convert("RGB")
    transforms_choices = [_rotation_transform, _blur_transform, _contrast_transform, _scaling_transform, _illumination_transform, _projective_transform]
    # Choose exactly one transform at random
    fn = random.choice(transforms_choices)
    out_img = fn(img)
    # derive name from function: _rotation_transform -> Rotation
    name = fn.__name__.lstrip("_").replace("_transform", "").capitalize()
    suffix = name
    return _save_image(out_img, src_path, suffix, out_dir=out_dir)


def process_root_folder(root: str, target_per_folder: int, logger=None):
    """Process each immediate subfolder of `root` and augment images until each subfolder
    has `target_per_folder` images. New images are created in the script directory.
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    # Create an `augmented_directory` sibling and copy the source inside it
    parent = root_path.parent
    augmented_root = parent / "augmented_directory"
    augmented_root.mkdir(exist_ok=True)
    augmented_child = augmented_root / root_path.name
    if augmented_child.exists():
        # avoid clobbering: append timestamp to the child folder name
        ts = time.strftime("%Y%m%d-%H%M%S")
        augmented_child = augmented_root / f"{root_path.name}_{ts}"
    logger.info("Copying %s -> %s", root_path, augmented_child)
    shutil.copytree(root_path, augmented_child)

    for child in sorted(augmented_child.iterdir()):
        if not child.is_dir():
            continue
        imgs = _list_images_in_dir(child)
        if not imgs:
            logger.warning("Folder %s contains no images; skipping", child)
            continue
        count = len(imgs)
        logger.info("Folder %s: %d images found", child, count)
        if count >= target_per_folder:
            logger.info("Folder %s already meets target (%d), skipping", child, target_per_folder)
            continue
        needed = target_per_folder - count
        logger.info("Folder %s needs %d more images, creating...", child, needed)
        created = 0
        # create images until we reach needed
        while created < needed:
            src = random.choice(imgs)
            _create_augmented_from_image(src, out_dir=child)
            created += 1

        logger.info("Folder %s: finished creating %d images", child, created)

    logger.info("Augmented dataset created at: %s", augmented_root)
    return augmented_child

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

    parser = argparse.ArgumentParser(description="Apply torchvision augmentations to a JPG image or a root folder and save outputs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--path", dest="path", help="Path to a JPG/JPEG image file (single-file mode)")
    group.add_argument("-r", "--root", dest="root", help="Root folder containing subfolders to augment (batch mode)")
    parser.add_argument("-n", "--num", dest="num", type=int, help="Target number of images per subfolder when using --root")
    parser.add_argument("--seed", dest="seed", type=int, default=None, help="Optional random seed for reproducible augmentations")
    args = parser.parse_args()

    # Simple console logger for the CLI entrypoint. Libraries shouldn't
    # configure root logging; we do it only when run as a script.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    # Optionally seed randomness for reproducible runs
    if args.seed is not None:
        random.seed(args.seed)

    if args.path:
        # Single-file mode
        try:
            outs = augment_image(args.path)
        except Exception:
            logger.exception("Error while augmenting %s", args.path)
            raise
        else:
            logger.info("Saved augmented images:")
            for o in outs:
                logger.info("%s", o)
    elif args.root:
        # Batch mode: --root provided
        if args.num is None:
            parser.error("--num is required when using --root")
        process_root_folder(args.root, args.num, logger=logger)

    else:
        parser.error("Either --path or --root must be provided.")