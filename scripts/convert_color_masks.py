"""Convert colourized masks to indexed PNGs using an explicit dataset palette.

Example palette JSON: {"[0,0,0]": 0, "[255,255,255]": 1}
Never guess a palette: both datasets have mirrors with different encodings.
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('--source',required=True); parser.add_argument('--destination',required=True); parser.add_argument('--palette-json',required=True)
args=parser.parse_args()
palette={tuple(map(int,key.strip('[]').split(','))):value for key,value in json.loads(Path(args.palette_json).read_text()).items()}
source,destination=Path(args.source),Path(args.destination); destination.mkdir(parents=True,exist_ok=True)
for path in source.glob('*.png'):
    rgb=np.asarray(Image.open(path).convert('RGB')); indexed=np.full(rgb.shape[:2],255,np.uint8)
    for colour,label in palette.items(): indexed[(rgb==colour).all(-1)]=label
    unknown=(indexed==255).sum()
    if unknown: raise ValueError(f'{path}: {unknown} pixels absent from palette')
    Image.fromarray(indexed).save(destination/path.name)
