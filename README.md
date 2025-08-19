# Biometric template generation

A multimodal biometric recognition system that uses both fingerprint and iris images to identify individuals. Separate deep learning models extract embeddings for each modality, which are then fused to form a compact biometric template — a numerical representation that can be stored securely and used for efficient, privacy-preserving authentication or identification.

### Venvs Creation

- for cpu

```bash
    conda env create -f envs\environment-cpu.yml
```

- for gpu

```bash
    conda env create -f envs\environment-gpu.yml
```

### Venv activation

```bash
    conda activate biometric-env
```

### Dataset prep
Dataset - https://www.kaggle.com/datasets/ninadmehendale/multimodal-iris-fingerprint-biometric-data

run this command to generate coresponding csv files for fingerprint and iris images.

```bash
python src/gen-labels.py --root ./data/IRIS_and_FINGERPRINT_DATASET --outdir ./labels
```