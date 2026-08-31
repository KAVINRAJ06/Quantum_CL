# Google Colab

Use [Quantum_SAM_OpenEarthMap_Colab.ipynb](Quantum_SAM_OpenEarthMap_Colab.ipynb) for the clean OpenEarthMap-only workflow. It installs Colab-safe dependencies, validates GPU availability, downloads and prepares the dataset, checks the discovered pairs, runs the two smoke tests, then starts training from `configs/openearthmap.yaml`.

For a new dataset, copy `configs/openearthmap.yaml` and edit `data.root`, `images`, `masks`, optional filename suffixes, `num_classes`, and `training`. The loader discovers matching filenames recursively and handles indexed masks or a consistent RGB palette automatically.