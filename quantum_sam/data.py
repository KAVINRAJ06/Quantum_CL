from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

class PairedAerialDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int):
        root = Path(root); self.image_size = image_size
        image_dir, mask_dir = root / 'images' / split, root / 'masks' / split
        self.samples = [(p, mask_dir / (p.stem + '.png')) for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'} and (mask_dir / (p.stem + '.png')).exists()]
        if not self.samples: raise RuntimeError(f'No paired images found under {root}')
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = Image.open(mask_path)
        # Indexed PNGs (mode P/L) preserve their class IDs. Do not silently use
        # RGB palette values as labels: OpenEarthMap exports can be colourized,
        # in which case they must first be converted with the dataset palette.
        if mask.mode not in {'P', 'L', 'I', 'I;16'}:
            raise ValueError(
                f'{mask_path} is {mask.mode}, not an indexed class mask. Convert colour masks to class IDs first.'
            )
        mask = mask.resize((self.image_size, self.image_size), Image.Resampling.NEAREST)
        image = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float() / 255
        # SAM's expected normalization.
        image = (image - torch.tensor([0.485,0.456,0.406])[:,None,None]) / torch.tensor([0.229,0.224,0.225])[:,None,None]
        return image, torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
