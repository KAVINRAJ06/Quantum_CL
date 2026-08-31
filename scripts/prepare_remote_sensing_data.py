"""Prepare arbitrary paired remote-sensing datasets for the configured loader.

Supports archives with explicit train/val folders and OpenEarthMap's regional
``<region>/images/...`` + ``<region>/labels/...`` structure.  If an archive has
no split directories, it creates a reproducible train/validation split.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import random
import shutil

IMAGE_DIRS = {"image", "images", "images_png", "img"}
MASK_DIRS = {"label", "labels", "mask", "masks", "masks_png", "annotations"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val"}


def split_of(path: Path) -> str | None:
    for part in path.parts:
        if part.lower() in SPLIT_NAMES:
            return SPLIT_NAMES[part.lower()]
    return None


def link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    if destination.exists(): return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not copy:
            os.link(source, destination)
            return
    except OSError:
        pass
    shutil.copy2(source, destination)


def find_pairs(source: Path):
    """Find masks by replacing an ancestor images directory with labels/masks."""
    pairs = []
    for image in source.rglob("*"):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
            continue
        ancestors = [parent for parent in image.parents if parent.name.lower() in IMAGE_DIRS]
        if not ancestors:
            continue
        image_dir = ancestors[0]
        relative = image.relative_to(image_dir)
        for name in MASK_DIRS:
            candidate = image_dir.parent / name / relative
            # Some archives use TIFF imagery with PNG labels (or vice versa).
            if not candidate.is_file():
                matches = [p for p in candidate.parent.glob(f"{candidate.stem}.*") if p.suffix.lower() in IMAGE_EXTS]
                candidate = matches[0] if len(matches) == 1 else candidate
            if candidate.is_file():
                pairs.append((image, candidate, split_of(image)))
                break
    if not pairs:
        raise RuntimeError(
            "No paired files found. Expected an images/ and labels|masks/ directory pair; "
            f"first archive entries: {[str(p.relative_to(source)) for p in list(source.rglob('*'))[:20]]}"
        )
    return pairs


def assign_splits(pairs, val_ratio: float, seed: int):
    if any(split for _, _, split in pairs):
        return [(image, mask, split or "train") for image, mask, split in pairs if split != "test"]
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    val_count = max(1, round(len(indices) * val_ratio))
    validation = set(indices[:val_count])
    return [(image, mask, "val" if index in validation else "train") for index, (image, mask, _) in enumerate(pairs)]


def unique_name(source: Path, image: Path) -> str:
    # Include the region path to prevent same-name tiles from overwriting each other.
    parts = [part for part in image.relative_to(source).with_suffix("").parts if part.lower() not in IMAGE_DIRS]
    return "__".join(parts)


def main(args):
    source, destination = Path(args.source), Path(args.destination)
    pairs = assign_splits(find_pairs(source), args.val_ratio, args.seed)
    counts = {"train": 0, "val": 0}
    for image, mask, split in pairs:
        name = unique_name(source, image)
        out_image = destination / "images" / split / f"{name}{image.suffix.lower()}"
        out_mask = destination / "masks" / split / f"{name}{mask.suffix.lower()}"
        link_or_copy(image, out_image, args.copy)
        link_or_copy(mask, out_mask, args.copy)
        counts[split] = counts.get(split, 0) + 1
    if not counts["train"] or not counts["val"]:
        raise RuntimeError(f"Both train and val need pairs; found {counts}.")
    print(f"Prepared {sum(counts.values())} pairs in {destination} (train={counts['train']}, val={counts['val']}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of attempting cheap hard links.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())