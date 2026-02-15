#!/usr/bin/env python3
"""Image data augmentation for a local dataset.

Create augmented copies in the same folder as each original image.
Before generating new files, remove previous augmentations.
"""

# --- Import ---------------------------------------------------------------
# Required imports for CLI, file handling, and image transforms.

import argparse
import random
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps


# --- Constants ------------------------------------------------------------
# Image extensions supported by the augmentation process.

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# --- CLI arguments --------------------------------------------------------
# Argument parser definition for script execution.

def build_parser():
    parser = argparse.ArgumentParser(
        description="Augment images in the train folder."
    )
    parser.add_argument(
        "--root",
        default="data",
        help="Dataset root containing train/val/test",
    )
    parser.add_argument(
        "--per-image",
        type=int,
        default=2,
        help="Number of augmented images to create for each original",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser


# --- Data augmentation ----------------------------------------------------
# Functions that apply random transforms to images.

def augment_image(img, rng):
    # Build a list of random transforms to apply in sequence.
    ops = []

    # Random rotation.
    if rng.random() < 0.7:
        angle = rng.uniform(-20, 20)
        ops.append(lambda im: im.rotate(angle, resample=Image.BILINEAR, expand=True))

    # Horizontal flip.
    if rng.random() < 0.5:
        ops.append(ImageOps.mirror)

    # Vertical flip.
    if rng.random() < 0.2:
        ops.append(ImageOps.flip)

    # Gaussian blur.
    if rng.random() < 0.4:
        radius = rng.uniform(0.5, 1.5)
        ops.append(lambda im: im.filter(ImageFilter.GaussianBlur(radius)))

    # Brightness adjustment.
    if rng.random() < 0.5:
        factor = rng.uniform(0.8, 1.2)
        ops.append(lambda im: ImageEnhance.Brightness(im).enhance(factor))

    # Contrast adjustment.
    if rng.random() < 0.5:
        factor = rng.uniform(0.8, 1.2)
        ops.append(lambda im: ImageEnhance.Contrast(im).enhance(factor))

    # Color saturation adjustment.
    if rng.random() < 0.4:
        factor = rng.uniform(0.8, 1.2)
        ops.append(lambda im: ImageEnhance.Color(im).enhance(factor))

    # Grayscale (disabled).
    # if rng.random() < 0.2:
    #     ops.append(lambda im: ImageOps.grayscale(im).convert("RGB"))

    rng.shuffle(ops)
    out = img
    for op in ops:
        out = op(out)

    # Force RGB output to avoid unexpected formats.
    if out.mode != "RGB":
        out = out.convert("RGB")
    return out


# --- Main -----------------------------------------------------------------
# CLI entrypoint for the data augmentation script.

def main():
    # Read CLI args and initialize a reproducible RNG.
    parser = build_parser()
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root)

    # Run augmentation on train only.
    split_dir = root / "train"
    if not split_dir.exists():
        return

    # --- Cleanup block -----------------------------------------------------
    # Remove previously augmented images from earlier runs.
    for old_aug_path in split_dir.rglob("*"):
        if (
            old_aug_path.is_file()
            and old_aug_path.suffix.lower() in IMG_EXTS
            and "_aug" in old_aug_path.stem
        ):
            old_aug_path.unlink(missing_ok=True)

    # --- Generation block --------------------------------------------------
    # Generate new augmented images starting only from originals.
    for img_path in split_dir.rglob("*"):
        if (
            not img_path.is_file()
            or img_path.suffix.lower() not in IMG_EXTS
            or "_aug" in img_path.stem
        ):
            continue
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                # --- Output block ------------------------------------------------------
                # Ensure the output directory exists.
                img_path.parent.mkdir(parents=True, exist_ok=True)
                for i in range(1, args.per_image + 1):
                    aug = augment_image(img, rng)
                    out_name = f"{img_path.stem}_aug{i}{img_path.suffix.lower()}"
                    aug.save(img_path.parent / out_name)
        except Exception:
            # Skip unreadable images without stopping the run.
            continue


# --- Script start ---------------------------------------------------------
# Script entrypoint executed from shell.

if __name__ == "__main__":
    main()
