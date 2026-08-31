"""Dataset discovery and preprocessing for paired semantic segmentation data."""
from __future__ import annotations
from pathlib import Path
from typing import Mapping
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
IGNORE_INDEX = 255

def _path(root: Path, pattern: str, split: str) -> Path:
    return root / pattern.format(split=split)

def _id(path: Path, suffix: str) -> str:
    return path.stem[:-len(suffix)] if suffix and path.stem.endswith(suffix) else path.stem

def discover_pairs(root: str | Path, split: str, images: str = "images/{split}", masks: str = "masks/{split}", image_suffix: str = "", mask_suffix: str = "", recursive: bool = True):
    root = Path(root); image_dir, mask_dir = _path(root, images, split), _path(root, masks, split)
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing split '{split}': images={image_dir}, masks={mask_dir}")
    walker = "rglob" if recursive else "glob"
    image_files = sorted(p for p in getattr(image_dir, walker)("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    mask_files = sorted(p for p in getattr(mask_dir, walker)("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    masks_by_id = {_id(path, mask_suffix): path for path in mask_files}
    pairs = [(image, masks_by_id[_id(image, image_suffix)]) for image in image_files if _id(image, image_suffix) in masks_by_id]
    if not pairs: raise RuntimeError(f"No image/mask pairs for split '{split}'. Check paths and pairing suffixes: {image_dir}, {mask_dir}")
    return pairs

def infer_palette(pairs, max_classes: int = 256) -> dict:
    """Infer one stable mapping from source mask values/colours to 0..C-1.

    Indexed masks are remapped too: datasets often use 1..C IDs, which would
    otherwise cause a device-side assert in PyTorch CrossEntropyLoss.
    """
    rgb_colours, indexed_values, rgb_count = set(), set(), 0
    for _, path in pairs:
        with Image.open(path) as mask:
            if mask.mode in {"P", "L", "I", "I;16"}:
                indexed_values.update(int(value) for value in np.unique(np.asarray(mask)) if int(value) != IGNORE_INDEX)
            else:
                rgb_count += 1
                rgb = np.asarray(mask.convert("RGB")).reshape(-1, 3)
                rgb_colours.update(map(tuple, np.unique(rgb, axis=0).tolist()))
    if rgb_count and indexed_values: raise ValueError("A dataset cannot mix RGB and indexed masks when palette: auto is used.")
    values = sorted(rgb_colours) if rgb_count else sorted(indexed_values)
    if not values: raise ValueError("No non-ignore mask labels were found.")
    if len(values) > max_classes: raise ValueError(f"Detected {len(values)} labels; provide an explicit mapping instead.")
    return {value: index for index, value in enumerate(values)}

class PairedAerialDataset(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int, *, images: str = "images/{split}", masks: str = "masks/{split}", image_suffix: str = "", mask_suffix: str = "", recursive: bool = True, palette: Mapping | str = "auto"):
        self.image_size = image_size
        self.samples = discover_pairs(root, split, images, masks, image_suffix, mask_suffix, recursive)
        self.palette = infer_palette(self.samples) if palette == "auto" else dict(palette)
        self.rgb_palette = bool(self.palette and isinstance(next(iter(self.palette)), tuple))
        if self.rgb_palette:
            self.palette = {tuple(map(int, key.strip("[]").split(","))) if isinstance(key, str) else tuple(key): int(value) for key, value in self.palette.items()}
        else:
            self.palette = {int(key): int(value) for key, value in self.palette.items()}
    def __len__(self): return len(self.samples)
    def _mask_tensor(self, mask_path: Path) -> torch.Tensor:
        with Image.open(mask_path) as mask:
            if self.rgb_palette:
                source = np.asarray(mask.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.NEAREST))
                target = np.full(source.shape[:2], IGNORE_INDEX, dtype=np.int64)
                for colour, label in self.palette.items(): target[(source == colour).all(axis=-1)] = label
            else:
                source = np.asarray(mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST), dtype=np.int64)
                target = np.full(source.shape, IGNORE_INDEX, dtype=np.int64)
                for source_label, label in self.palette.items(): target[source == source_label] = label
                target[source == IGNORE_INDEX] = IGNORE_INDEX
            unknown_mask = target == IGNORE_INDEX if self.rgb_palette else ((target == IGNORE_INDEX) & (source != IGNORE_INDEX))
            if unknown_mask.any():
                unknown = np.unique(source[unknown_mask])[:10]
                raise ValueError(f"{mask_path} contains labels absent from the dataset mapping: {unknown.tolist()}")
        return torch.from_numpy(target.copy())
    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        with Image.open(image_path) as raw:
            image = raw.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            image = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(255)
        mean, std = image.new_tensor([.485, .456, .406])[:, None, None], image.new_tensor([.229, .224, .225])[:, None, None]
        return (image - mean) / std, self._mask_tensor(mask_path)