# Quantum SAM Continual Segmentation

Parameter-efficient semantic segmentation for aerial imagery. The model uses a frozen (or selectively fine-tuned) Segment Anything Model (SAM) image encoder and SAM mask decoder, with a PennyLane variational quantum circuit only at the compact image-embedding bottleneck. Sequential-domain training is supported with experience replay and Elastic Weight Consolidation (EWC).

## Installation

```powershell
python -m pip install -r requirements.txt
```

For the full Google Colab workflow, open [the Colab notebook](colab/Quantum_SAM_Continual_Training.ipynb). It retains Colab's PyTorch build, requests your Kaggle API token interactively, downloads both supplied datasets, and launches the sequential training command.

Install a SAM checkpoint from Hugging Face (for example `facebook/sam-vit-base`) before training. It is downloaded automatically by Transformers on first use.

## Dataset layout

The loader intentionally uses a simple, inspectable paired-file layout. Prepare each dataset as:

```
data/
  openearthmap/
    images/{train,val}/*.jpg
    masks/{train,val}/*.png
  loveda/
    images/{train,val}/*.png
    masks/{train,val}/*.png
```

Image and mask names must share the stem. Masks must be indexed PNGs containing class IDs from `0` to `num_classes - 1`; use `255` for ignored pixels. OpenEarthMap has 8 semantic classes. LoveDA's official masks contain 7 foreground classes; the default configuration includes a background class, giving 8 output classes. Inspect the Kaggle mirror's mask format before setting `--num-classes`. If it has RGB colour masks, first convert them with your verified palette using `scripts/convert_color_masks.py`; the training loader deliberately refuses to guess colour-to-class mappings.

## Train sequentially

```powershell
python train.py --data-root data --tasks openearthmap loveda --num-classes 8 --sam-model facebook/sam-vit-base --image-size 512 --batch-size 2 --epochs-per-task 15 --qubits 8 --freeze-sam
```

For a quick CPU/API check without data or a SAM download:

```powershell
python smoke_test.py
```

The circuit is simulated with `default.qubit`; it is not evidence of an advantage over a comparable classical bottleneck. Full SAM training is memory-intensive; start with ViT-B, frozen SAM, 512px crops, and eight qubits.
