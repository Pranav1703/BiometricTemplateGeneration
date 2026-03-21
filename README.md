# Biometric Template Generation

A fingerprint recognition system using deep learning (ResNet50 + ArcFace) for feature extraction. Supports multiple datasets (CASIA, FVC2000) with cancelable biometric techniques for privacy-preserving template protection.

## Quick Start

### 1. Environment Setup

```bash
# Create environment
conda env create -f envs/environment-cpu.yml   # CPU
conda env create -f envs/environment-gpu.yml     # GPU

# Activate
conda activate biometric-env

# Install additional dependencies
pip install pytest gdown
```

### 2. Download Datasets

#### FVC2000 Dataset
- **Website**: http://bias.csr.unibo.it/fvc2000/download.asp
- **Direct Download**: https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-030-83624-5_4/MediaObjects/74034_3_En_4_MOESM1_ESM.zip
- Extract to: `datasets/FVC2000/DB1_a/`

#### CASIA Dataset
- **Drive**: https://drive.google.com/drive/folders/1yFb8jmAO72nIamSHVKkXtmCY5I52iRCX?usp=sharing
- Extract to: `datasets/CASIA-dataset/`

### 3. Auto-Setup (Creates Required Folders)

```bash
# This creates all necessary directories automatically
python -m src.config
```

### 4. Generate Labels

```bash
# Generate FVC2000 labels
python -m src.utils.gen_labels --dataset fvc2000

# Generate CASIA labels
python -m src.utils.gen_labels --dataset casia
```

### 5. Train Model

```bash
# Train on FVC2000
python -m src.fingerprint.train --dataset fvc2000

# Train on CASIA
python -m src.fingerprint.train --dataset casia
```

### 6. Test/Evaluate

```bash
# Generate embeddings
python -m src.fingerprint.test --dataset fvc2000 --mode generate

# Evaluate (EER, FAR, FRR)
python -m src.fingerprint.test --dataset fvc2000 --mode evaluate

# Visualize similarity distributions
python -m src.fingerprint.test --dataset fvc2000 --mode visualize
```

## Project Structure

```
BiometricTemplateGeneration/
├── datasets/                    # Dataset directory
│   ├── labels/                 # Generated CSV labels
│   │   ├── FVC2000_labels/
│   │   │   ├── fvc2000_train.csv
│   │   │   └── fvc2000_val.csv
│   │   └── CASIA_labels/
│   │       ├── casia_train.csv
│   │       └── casia_val.csv
│   ├── FVC2000/               # FVC2000 raw dataset
│   │   └── DB1_a/            # DB1_a fingerprint images
│   └── CASIA-dataset/         # CASIA raw dataset
├── artifacts/                  # Generated outputs
│   ├── models/                # Trained model files (*_arcface_model.pth)
│   ├── embeddings/            # Generated embeddings
│   ├── metrics/               # Evaluation metrics
│   ├── plots/                 # Visualizations
│   │   └── tensorboard/       # TensorBoard logs
│   └── logs/                  # Training logs
├── src/                       # Source code
│   ├── config.py              # Configuration & auto-folder creation
│   ├── fingerprint/
│   │   ├── train.py           # Unified training script
│   │   ├── test.py            # Unified testing script
│   │   ├── inference/         # Embedding generation
│   │   ├── core/              # Biometric crypto systems
│   │   └── benchmarks/        # Cancelable biometric benchmarks
│   └── utils/                 # Utility functions
│       ├── Dataset_Loader.py
│       ├── preprocess_fingerprint.py
│       ├── gen_labels.py
│       ├── model_downloader.py
│       ├── hash_utils.py
│       ├── xor_utils.py
│       ├── quantization.py
│       ├── ecc_utils.py
│       └── logger.py
├── tests/                     # Test suite
├── docs/                      # Documentation
├── envs/                      # Conda environments
│   ├── environment-cpu.yml
│   └── environment-gpu.yml
├── AGENTS.md                  # Agent guidelines
└── README.md
```

## Usage

### Training
```bash
# Basic training
python -m src.fingerprint.train --dataset fvc2000

# Resume from checkpoint
python -m src.fingerprint.train --dataset casia --resume artifacts/models/casia_arcface_model.pth
```

### Testing
```bash
# Generate embeddings
python -m src.fingerprint.test --dataset fvc2000 --mode generate

# Evaluate performance
python -m src.fingerprint.test --dataset fvc2000 --mode evaluate

# Run all (generate + evaluate + visualize)
python -m src.fingerprint.test --dataset fvc2000 --mode all
```

### Download Pre-trained Models
```bash
python -m src.utils.model_downloader
```

### Running Tests
```bash
# All tests
python -m pytest tests/

# Single test
python -m pytest tests/test_crypto_utils.py::test_sha256_determinism
```

## Features

- **Deep Learning**: ResNet50 backbone with ArcFace loss for robust embeddings
- **Multi-dataset**: Unified training/testing for CASIA and FVC2000
- **Cancelable Biometrics**: Template protection techniques
- **Automatic Setup**: Config auto-creates required directories
- **Model Download**: Automatic download from Google Drive

## Requirements

- Python 3.10+
- PyTorch + TorchVision
- NumPy, OpenCV, scikit-learn
- TensorBoard for visualization
- Conda for environment management