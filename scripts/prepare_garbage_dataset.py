"""Utility script to organise a garbage classification dataset for fastai.

Example usage:
    python scripts/prepare_garbage_dataset.py \
        --source ~/raw_garbage_images \
        --dest ~/datasets/garbage \
        --valid-pct 0.2

The script expects the source directory to contain one subfolder per label,
with the images for that label inside. It copies the images into a new
train/valid folder structure that fastai's `ImageDataLoaders.from_folder`
understands.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable


def get_image_files(path: Path) -> Iterable[Path]:
    """Return image files inside ``path`` with common extensions."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    for file in path.iterdir():
        if file.is_file() and file.suffix.lower() in exts:
            yield file


def copy_subset(files: Iterable[Path], dest_dir: Path) -> None:
    """Copy ``files`` into ``dest_dir`` preserving file names."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dest_dir / src.name)


def prepare_dataset(source: Path, dest: Path, valid_pct: float, seed: int) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    if (dest / "train").exists() or (dest / "valid").exists():
        raise FileExistsError(
            "Destination already contains train/valid folders. "
            "Please choose an empty destination or remove the existing folders."
        )

    random.seed(seed)

    for label_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        label = label_dir.name
        images = list(get_image_files(label_dir))
        if not images:
            print(f"Skipping '{label}' because no images were found.")
            continue

        random.shuffle(images)
        cutoff = int(len(images) * (1 - valid_pct))
        train_imgs = images[:cutoff] or images
        valid_imgs = images[cutoff:] or images[:1]

        copy_subset(train_imgs, dest / "train" / label)
        copy_subset(valid_imgs, dest / "valid" / label)

        print(
            f"Copied {len(train_imgs)} train and {len(valid_imgs)} valid images for label '{label}'."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare garbage dataset for fastai")
    parser.add_argument("--source", type=Path, required=True, help="Directory with labelled subfolders")
    parser.add_argument("--dest", type=Path, required=True, help="Output directory for train/valid folders")
    parser.add_argument("--valid-pct", type=float, default=0.2, help="Fraction of images to reserve for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(args.source, args.dest, args.valid_pct, args.seed)


if __name__ == "__main__":
    main()
