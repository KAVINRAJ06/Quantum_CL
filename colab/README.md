# Google Colab

Open `Quantum_SAM_Continual_Training.ipynb` in Google Colab and run its cells in order. It clones this repository, installs the Colab-specific dependencies without replacing Colab's PyTorch build, downloads both Kaggle datasets, prepares their paired image/mask layout, and starts sequential training.

You need a Kaggle account and API token. Download `kaggle.json` from Kaggle Settings > API, then upload it only when the notebook asks. Do not commit that file.

After the preparation cell, inspect the reported image/mask counts. If OpenEarthMap was distributed as RGB palette masks, convert them with `scripts/convert_color_masks.py` using the palette documented by that specific Kaggle mirror before training.
