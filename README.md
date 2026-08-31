# Quantum SAM Continual Segmentation

Configurable semantic segmentation for aerial imagery using a SAM encoder/decoder with a compact PennyLane bottleneck. It supports standard indexed class masks and RGB palette masks, pairs files automatically by filename, and trains on CUDA without CPU/CUDA tensor mixing.

## Run OpenEarthMap

Use the clean Colab notebook: [colab/Quantum_SAM_OpenEarthMap_Colab.ipynb](colab/Quantum_SAM_OpenEarthMap_Colab.ipynb). It downloads and prepares OpenEarthMap only, validates the dataset, then trains from the YAML configuration.

Locally:

```powershell
python -m pip install -r requirements.txt
python train.py --config configs/openearthmap.yaml
```

## Configure another dataset

Copy `configs/openearthmap.yaml`, then set these values:

- `data.root`: extracted dataset directory.
- `data.images` and `data.masks`: folder patterns, with `{split}` replaced by `train` or `val`.
- `image_suffix` / `mask_suffix`: text removed before filename pairing, such as `_mask`.
- `palette`: `auto` accepts indexed PNG masks and deterministically maps a consistent RGB palette; use an explicit mapping when class IDs must follow a prescribed order.
- `num_classes` and `training`: model and runtime settings.

The loader searches subfolders recursively, pairs images and masks by normalized stems, preserves indexed class IDs, resizes images bilinearly and masks with nearest-neighbour interpolation, and validates class counts before training.

## Important fixes

The PennyLane circuit is host-evaluated by `default.qubit`; the pipeline now explicitly returns its result to the embedding’s CUDA device. SAM inputs are also internally resized to SAM’s required encoder size and logits are resized back to the configured training crop size. This fixes the common `cuda:0 and cpu` mismatch and fixed-position-embedding shape failures.

Run the no-data checks with:

```powershell
python smoke_test.py
python tests/test_data_pipeline.py
```