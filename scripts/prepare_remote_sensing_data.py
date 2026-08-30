"""Create the paired layout consumed by train.py from extracted Kaggle data.

The Kaggle mirrors can change their top-level folder name, so this utility finds
the known image/mask leaf directories rather than relying on archive paths.
It links source files when possible; pass --copy if links are unsuitable.
"""
import argparse
import os
from pathlib import Path
import shutil

IMAGE_DIRS = {"image", "images", "images_png", "img"}
MASK_DIRS = {"label", "labels", "mask", "masks", "masks_png", "annotations"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def split_of(path: Path) -> str | None:
    names = {part.lower() for part in path.parts}
    if names & {"val", "valid", "validation"}: return "val"
    if names & {"train", "training"}: return "train"
    return None

def link_or_copy(source: Path, destination: Path, copy: bool):
    if destination.exists(): return
    try:
        if not copy: os.link(source, destination); return
    except OSError:
        pass
    shutil.copy2(source, destination)

def main(args):
    source, destination = Path(args.source), Path(args.destination)
    image_dirs = [p for p in source.rglob('*') if p.is_dir() and p.name.lower() in IMAGE_DIRS]
    mask_dirs = [p for p in source.rglob('*') if p.is_dir() and p.name.lower() in MASK_DIRS]
    made = 0
    for image_dir in image_dirs:
        split = split_of(image_dir)
        if not split: continue
        # Prefer a mask directory inside the same split/region tree.
        candidates = [p for p in mask_dirs if split_of(p) == split]
        same_region = [p for p in candidates if p.parent.name.lower() == image_dir.parent.name.lower()]
        candidates = same_region or candidates
        if not candidates: continue
        images = {p.stem: p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}
        for mask_dir in candidates:
            masks = {p.stem: p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() == '.png'}
            shared = images.keys() & masks.keys()
            if not shared: continue
            region = image_dir.parent.name.lower().replace(' ', '_')
            for stem in shared:
                name = f'{region}_{stem}'
                out_image = destination / 'images' / split / f'{name}{images[stem].suffix.lower()}'
                out_mask = destination / 'masks' / split / f'{name}.png'
                out_image.parent.mkdir(parents=True, exist_ok=True); out_mask.parent.mkdir(parents=True, exist_ok=True)
                link_or_copy(images[stem], out_image, args.copy); link_or_copy(masks[stem], out_mask, args.copy); made += 1
            break
    if not made:
        raise RuntimeError('No paired files found. Inspect the archive and provide the expected images/masks layout manually.')
    print(f'Prepared {made} image/mask pairs in {destination}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True); p.add_argument('--destination', required=True); p.add_argument('--copy', action='store_true')
    main(p.parse_args())
