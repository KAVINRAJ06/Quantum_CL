"""Dataset discovery and preprocessing for paired semantic segmentation data."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _path(root: Path, pattern: str, split: str) -> Path:
    return root / pattern.format(split=split)


def _id(path: Path, suffix: str) -> str:
    stem = path.stem
    return stem[:-len(suffix)] if suffix and stem.endswith(suffix) else stem


def discover_pairs(root: str | Path, split: str, images: str = "images/{split}", masks: str = "masks/{split}", image_suffix: str = "", mask_suffix: str = "", recursive: bool = True):
    """Find image/mask pairs by configurable normalized filename stems."""
    root = Path(root)
    image_dir, mask_dir = _path(root, images, split), _path(root, masks, split)
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing split '{split}': images={image_dir}, masks={mask_dir}")
    walker = "rglob" if recursive else "glob"
    image_files = sorted(p for p in getattr(image_dir, walker)("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    mask_files = sorted(p for p in getattr(mask_dir, walker)("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    mask_by_id = {_id(p, mask_suffix): p for p in mask_files}
    pairs = [(image, mask_by_id[_id(image, image_suffix)]) for image in image_files if _id(image, image_suffix) in mask_by_id]
    if not pairs:
        raise RuntimeError(f"No image/mask pairs for split '{split}'. Check paths and pairing suffixes: {image_dir}, {mask_dir}")
    return pairs


def infer_palette(pairs, max_colours: int = 256) -> dict[tuple[int, int, int], int] | None:
    """Create a deterministic RGB-colour-to-class mapping when masks are RGB."""
    colours: set[tuple[int, int, int]] = set()
    rgb_masks = 0
    for _, path in pairs:
        with Image.open(path) as mask:
            if mask.mode not in {"P", "L", "I", "I;16"}:
                rgb_masks += 1
                values = np.asarray(mask.convert("RGB")).reshape(-1, 3)
                colours.update(map(tuple, np.unique(values, axis=0).tolist()))
                if len(colours) > max_colours:
                    raise ValueError(f"{path}: more than {max_colours} mask colours; provide data.palette explicitly.")
    if rgb_masks and rgb_masks != len(pairs):
        raise ValueError("A split mixes indexed and RGB masks. Convert masks or configure one consistent encoding.")
    return {colour: index for index, colour in enumerate(sorted(colours))} if rgb_masks else None


class PairedAerialDataset(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int, *, images: str = "images/{split}", masks: str = "masks/{split}", image_suffix: str = "", mask_suffix: str = "", recursive: bool = True, palette: Mapping | str | None = "auto"):
        self.image_size = image_size
        self.samples = discover_pairs(root, split, images, masks, image_suffix, mask_suffix, recursive)
        self.palette = infer_palette(self.samples) if palette == "auto" else palette
        if self.palette:
            self.palette = {tuple(map(int, key.strip("[]").split(","))) if isinstance(key, str) else tuple(key): int(value) for key, value in self.palette.items()}

    def __len__(self):
        return len(self.samples)

    def _mask_tensor(self, mask_path: Path) -> torch.Tensor:
        with Image.open(mask_path) as mask:
            if self.palette is None:
                if mask.mode not in {"P", "L", "I", "I;16"}:
                    raise ValueError(f"{mask_path} is {mask.mode}; set data.palette: auto or an explicit palette.")
                array = np.asarray(mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST), dtype=np.int64)
            else:
                rgb = np.asarray(mask.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.NEAREST))
                array = np.full(rgb.shape[:2], 255, dtype=np.int64)
                for colour, label in self.palette.items():
                    array[(rgb == colour).all(axis=-1)] = label
                if (array == 255).any():
                    raise ValueError(f"{mask_path} contains a colour not present in the configured palette.")
        return torch.from_numpy(array.copy())

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as raw:
            image = raw.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            image = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255)
        mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
        return (image - mean) / std, self._mask_tensor(mask_path)