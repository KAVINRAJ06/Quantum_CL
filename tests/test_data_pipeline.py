from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
from quantum_sam.data import PairedAerialDataset

with TemporaryDirectory() as directory:
    root = Path(directory)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "masks" / split).mkdir(parents=True)
        Image.fromarray(np.full((8, 9, 3), 127, dtype=np.uint8)).save(root / "images" / split / "tile_1.jpg")
        Image.fromarray(np.array([[0, 1], [2, 0]], dtype=np.uint8)).save(root / "masks" / split / "tile_1.png")
    data = PairedAerialDataset(root, "train", 14)
    image, mask = data[0]
    assert image.shape == (3, 14, 14) and mask.shape == (14, 14)
print("PASS: paired discovery, resizing, normalization and indexed mask preprocessing")