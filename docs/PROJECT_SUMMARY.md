# Biometric Template Protection - Project Summary

## Overview

This project implements a **multi-layered biometric cryptosystem** for fingerprint recognition using deep learning (ResNet50 + ArcFace). It provides privacy-preserving template protection through **Cancelable Biometrics + PBKDF2 key derivation**.

## How It Works

### System Architecture (Dual-Path Verification)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ENROLLMENT PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────┘

Raw Fingerprint Image
        ↓
┌─────────────────────────────────┐
│  Deep Learning Model (ResNet50)  │
│       + ArcFace Loss             │
└─────────────────────────────────┘
        ↓
   Raw Embedding (512-dim)
        ↓
┌─────────────────────────────────┐
│  Cancelable Transform            │
│  R = α × R_bio + (1-α) × R_key │
└─────────────────────────────────┘
        ↓
   Transformed Embedding (512-dim)
        ↓
          PBKDF2 Key
         Derivation
                   ↓
   Cryptographic Key
          (256-bit)
                 ↓             
                   Cosine   
                   Similarity
                   ↓   
                   Authentication  
                   (ACCEPT/  
                   REJECT)  



```

### Verification Pipeline

```
Query Fingerprint → Embedding → Cancelable Transform → Transformed
                            ↓
                   ┌────────┴────────┐
                   ↓                 ↓
              PATH A:             PATH B:
              Cosine             PBKDF2
              Similarity          Derivation
                   ↓                 ↓
              Compare to          Key Hash
              Threshold            Match
                   ↓                 ↓
              ACCEPT/            ACCEPT/
              REJECT             REJECT
```

### Key Components

1. **Deep Learning Model**: ResNet50 backbone with ArcFace loss for 512-dimensional embeddings
2. **Cancelable Transform**: User-specific random projection for revocability
3. **PBKDF2 Key Derivation**: Cryptographic key from transformed embedding
4. **Dual-Path Verification**: Both similarity-based (Path A) and key-based (Path B) authentication

### Security Properties

- **Revocability**: Different user keys → different templates
- **Unlinkability**: Different applications → different templates
- **Non-invertibility**: Original biometric cannot be recovered
- **Cryptographic Binding**: Key derived via PBKDF2 with SHA-256
- **Dual Security**: Both biometric similarity AND cryptographic key verification

### Configuration (src/config.py)

```python
# FVC2000 4-Finger Configuration
FVC2000_ENROLLMENT_FINGERS = [1, 2]  # Fingers for enrollment
FVC2000_VERIFICATION_FINGERS = [3, 4]  # Fingers for verification
FVC2000_NUM_PERSONS = 100

# Cancelable Transform Parameters
DEFAULT_CANCELABLE_ALPHA = 0.6  # Blend factor
DEFAULT_SIMILARITY_THRESHOLD = 0.5  # Cosine similarity threshold

# PBKDF2 Parameters
PBKDF2_ITERATIONS = 10000
PBKDF2_KEY_LENGTH = 32  # 256-bit key
PBKDF2_SALT_LENGTH = 16  # 128-bit salt
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create environment
conda env create -f envs/environment-cpu.yml   # CPU
conda env create -f envs/environment-gpu.yml   # GPU

# Activate
conda activate biometric-env

# Install dependencies
pip install pytest gdown
```

### 2. Auto-Setup (Creates Required Folders)

```bash
python -m src.config
```

### 3. Download Datasets

**FVC2000 Dataset:**
- Website: http://bias.csr.unibo.it/fvc2000/download.asp
- Direct: https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-030-83624-5_4/MediaObjects/74034_3_En_4_MOESM1_ESM.zip
- Extract to: `datasets/FVC2000/DB1_a/`

**CASIA Dataset:**
- Drive: https://drive.google.com/drive/folders/1yFb8jmAO72nIamSHVKkXtmCY5I52iRCX
- Extract to: `datasets/CASIA-dataset/`

### 4. Generate Labels

```bash
# FVC2000
python -m src.utils.gen_labels --dataset fvc2000

# CASIA
python -m src.utils.gen_labels --dataset casia
```

### 5. Train Model

```bash
# FVC2000
python -m src.fingerprint.train --dataset fvc2000

# CASIA
python -m src.fingerprint.train --dataset casia
```

### 6. Generate Embeddings

```bash
python -m src.fingerprint.test --dataset fvc2000 --mode generate
```

### 7. Evaluate Model

```bash
python -m src.fingerprint.test --dataset fvc2000 --mode evaluate
```

### 8. Download Pre-trained Models

```bash
python -m src.utils.model_downloader
```

---

## Usage Examples

### Training

```bash
# Train on FVC2000
python -m src.fingerprint.train --dataset fvc2000

# Train on CASIA
python -m src.fingerprint.train --dataset casia

# Resume from checkpoint
python -m src.fingerprint.train --dataset fvc2000 --resume artifacts/models/fvc2000_arcface_model.pth
```

### Testing

```bash
# Generate embeddings
python -m src.fingerprint.test --dataset fvc2000 --mode generate

# Evaluate (EER, FAR, FRR)
python -m src.fingerprint.test --dataset fvc2000 --mode evaluate

# Visualize distributions
python -m src.fingerprint.test --dataset fvc2000 --mode visualize

# Run all
python -m src.fingerprint.test --dataset fvc2000 --mode all
```

### Using Biometric Cryptosystem

```python
import numpy as np
from src.fingerprint.core.biometric_crypto_system import BiometricCryptoSystem

# Initialize system
system = BiometricCryptoSystem(
    embedding_dim=512,
    key_size=32,           # 256-bit key
    ecc_capacity=0.3,      # 30% error correction
    cancelable_alpha=0.6
)

# Get embedding from your model
embedding = backbone(fingerprint_image)  # 512-dim array

# Enroll
template = system.enroll(embedding, "user_001")
print(f"Template keys: {template.keys()}")

# Verify later
success, key = system.verify(query_embedding, template, "user_001")
if success:
    print(f"Authenticated! Key: {key.hex()}")
```

---

## Project Structure

```
BiometricTemplateGeneration/
├── datasets/
│   ├── labels/
│   │   ├── FVC2000_labels/
│   │   └── CASIA_labels/
│   ├── FVC2000/DB1_a/
│   └── CASIA-dataset/
├── artifacts/
│   ├── models/          # Trained models
│   ├── embeddings/      # Generated embeddings
│   ├── metrics/        # Evaluation results
│   └── plots/          # Visualizations
├── src/
│   ├── config.py           # Configuration
│   ├── fingerprint/
│   │   ├── train.py       # Unified training
│   │   ├── test.py       # Unified testing
│   │   ├── core/
│   │   │   ├── biometric_crypto_system.py
│   │   │   ├── cancelable_transform.py
│   │   │   ├── fuzzy_commitment.py
│   │   │   └── biometric_key_generator.py
│   │   ├── inference/
│   │   │   └── gen_embeddings.py
│   │   └── benchmarks/
│   └── utils/
│       ├── Dataset_Loader.py
│       ├── preprocess_fingerprint.py
│       ├── gen_labels.py
│       ├── model_downloader.py
│       ├── hash_utils.py
│       ├── xor_utils.py
│       ├── quantization.py
│       └── ecc_utils.py
├── tests/
├── docs/
└── envs/
```

---

## Configuration

All configuration is centralized in `src/config.py`:

```python
# Dataset paths
FVC2000_DIR = "datasets/FVC2000"
CASIA_DIR = "datasets/CASIA-dataset"

# Model settings
EMBEDDING_DIM = 512
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# Cryptosystem settings
KEY_SIZE = 32          # 256-bit
ECC_CAPACITY = 0.3     # 30%
CANCELABLE_ALPHA = 0.6
```

---

## Benchmark Results

### Cancelable Biometrics (Working)
```
AUC: 1.0000
EER: 0.0000
d-prime: 6.31
```

### Cancelable + PBKDF2 (Current)
- Uses PBKDF2 for key derivation instead of fuzzy commitment
- Similarity-based verification (threshold = 0.5)
- Works well with high-dimensional deep embeddings

**Benchmark Results:**
```
AUC: 0.9915
EER: 0.0074
d-prime: 6.79
```

### 4-Finger Authentication Test (Dual-Path)
```python
# Run test
python -m src.fingerprint.test_4finger

# Or with custom parameters
python -c "from src.fingerprint.test_4finger import run_4finger_test; run_4finger_test(threshold=0.2)"
```

**Results:**
```
Configuration:
  Alpha: 0.6
  Threshold: 0.2
  Intra-class std: 0.02

Genuine Verification (Path A + Path B):
  Key derivation success: 98/100
  Mean similarity: 0.2817

Impostor Verification:
  Mean similarity: 0.0008

Metrics:
  EER: 0.00%
  FAR: 0.00%
  FVR: 0.00%
  GAR: 100.00%
  AUC: 1.0000
  d-prime: 6.69
```

---

## Security Features

1. **Cancelable Transform**: R = alpha x R_bio + (1-alpha) x R_key
2. **PBKDF2 Key Derivation**: PBKDF2-SHA256(transformed, salt, iter=10000)
3. **Dual-Path Verification**: Both similarity AND key-based authentication
4. **Key Generation**: 256-bit cryptographic keys from biometrics
5. **Template Protection**: Non-invertible transformation

---

## Troubleshooting

### No models found
```bash
# Download from Google Drive
python -m src.utils.model_downloader
```

### Dataset not found
```bash
# Check dataset paths
ls datasets/
ls datasets/FVC2000/
ls datasets/CASIA-dataset/
```

### Label files missing
```bash
# Generate labels
python -m src.utils.gen_labels --dataset fvc2000
python -m src.utils.gen_labels --dataset casia
```

---

## References

- Ratha et al. (2001) - Cancelable Biometrics
- PBKDF2 (RFC 2898) - Key Derivation Function
- ResNet50 + ArcFace - Deep Learning Embeddings
- *Note: Fuzzy Commitment not suitable for high-dimensional deep embeddings*

---

## Contact

For questions, refer to:
- `docs/CHANGES.md` - Detailed change log
- `docs/plan.txt` - Implementation plan
- `README.md` - Quick reference
- `AGENTS.md` - Developer guidelines
