# AGENTS.md - Biometric Template Generation

This file provides essential information for agentic coding agents working in this biometric template generation repository.

## Project Structure

```
BiometricTemplateGeneration/
├── datasets/                    # Dataset directory
│   ├── labels/                 # Generated CSV labels
│   │   ├── FVC2000_labels/
│   │   └── CASIA_labels/
│   ├── FVC2000/               # FVC2000 raw dataset
│   └── CASIA-dataset/         # CASIA raw dataset
├── artifacts/                  # Generated outputs
│   ├── models/                # Trained model files
│   ├── embeddings/            # Generated embeddings
│   ├── metrics/               # Evaluation metrics
│   ├── plots/                 # Visualizations
│   └── logs/                  # Training logs
├── src/
│   ├── config.py              # Centralized configuration
│   ├── fingerprint/
│   │   ├── train.py           # Unified training (--dataset flag)
│   │   ├── test.py            # Unified testing (--dataset, --mode flags)
│   │   ├── inference/         # Embedding generation
│   │   ├── core/              # Biometric crypto systems
│   │   └── benchmarks/        # Cancelable biometric benchmarks
│   └── utils/                 # Consolidated utilities
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
└── envs/                      # Conda environments
```

## Build and Test Commands

### Environment Setup
```bash
conda env create -f envs/environment-cpu.yml   # CPU environment
conda env create -f envs/environment-gpu.yml   # GPU environment
conda activate biometric-env
pip install pytest gdown    # Install test and download dependencies
```

### Auto-Setup (Creates Required Folders)
```bash
python -m src.config
```

### Dataset Download Links
```bash
# FVC2000 Dataset
- Website: http://bias.csr.unibo.it/fvc2000/download.asp
- Direct: https://static-content.springer.com/esm/chp%3A10.1007%2F978-3-030-83624-5_4/MediaObjects/74034_3_En_4_MOESM1_ESM.zip

# CASIA Dataset
- Drive: https://drive.google.com/drive/folders/1yFb8jmAO72nIamSHVKkXtmCY5I52iRCX?usp=sharing
```

### Dataset Preparation
```bash
# Generate FVC2000 training/validation labels
python -m src.utils.gen_labels --dataset fvc2000

# Generate CASIA labels
python -m src.utils.gen_labels --dataset casia
```

### Training (Unified - supports both datasets)
```bash
python -m src.fingerprint.train --dataset fvc2000
python -m src.fingerprint.train --dataset casia
python -m src.fingerprint.train --dataset fvc2000 --resume artifacts/models/fvc2000_arcface_model.pth
```

### Testing (Unified - multiple modes)
```bash
# Generate embeddings
python -m src.fingerprint.test --dataset fvc2000 --mode generate

# Evaluate (EER, FAR, FRR)
python -m src.fingerprint.test --dataset fvc2000 --mode evaluate

# Visualize similarity distributions
python -m src.fingerprint.test --dataset fvc2000 --mode visualize

# Run all steps
python -m src.fingerprint.test --dataset fvc2000 --mode all
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_crypto_utils.py

# Run single test function
python -m pytest tests/test_crypto_utils.py::test_sha256_determinism

# Run with verbose output
python -m pytest tests/ -v
```

### Model Management
```bash
# Download pre-trained models from Google Drive
python -m src.utils.model_downloader
```

## Code Style Guidelines

### Import Style
- Group imports: stdlib, third-party, local
- Use absolute imports: `from src.config import FVC2000_TRAIN_CSV`
- Avoid wildcard imports: `from module import *`
- Keep imports at file top, alphabetical within groups

### Naming Conventions
- **Variables/functions**: `snake_case` (e.g., `preprocess_fingerprint`)
- **Classes**: `PascalCase` (e.g., `FingerprintEmbeddingNet`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `EMBEDDING_DIM`)
- **Private members**: `_single_underscore` prefix
- **Config variables**: `UPPER_SNAKE_CASE` in config files

### Type Hints
- Use type hints for function signatures
- Example: `def get_logger(name: str = "train") -> logging.Logger`
- Import types from `typing` when needed

### File Structure
- Source code in `src/`
- Utilities in `src/utils/`
- Tests in `tests/` at root
- Artifacts in `artifacts/`
- Use `__init__.py` for proper package structure

### Logging
- Use centralized logger: `from src.utils.logger import get_logger`
- Descriptive names: `logger = get_logger("fingerprint_train")`
- Levels: DEBUG (details), INFO (progress), WARNING/ERROR (issues)

### Configuration
- Centralized in `src/config.py`
- Use `os.path.join()` for paths
- Create directories: `os.makedirs(dir, exist_ok=True)`

### Error Handling
- Try-catch for file I/O
- Validate input types/ranges
- Raise descriptive exceptions
- Use assertions in tests

### Deep Learning Best Practices
- Device-agnostic: `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- L2 normalize embeddings: `nn.functional.normalize(x, p=2, dim=1)`
- Proper DataLoader with batch size
- Save models with state dictionaries
- Use TensorBoard for visualization

### Biometric Patterns
- L2 normalization for unit hypersphere embeddings
- Cosine similarity for comparison
- Dataset splitting to avoid data leakage
- Quantization for template protection

### Testing Guidelines
- Descriptive names: `test_sha256_determinism`
- Use pytest fixtures
- Test edge cases and errors
- Test crypto functions with known inputs

### Security
- Never hardcode secrets/API keys
- Use proper randomness for crypto
- Validate inputs in preprocessing
- Follow secure coding for biometric data

## Common Patterns

### Dataset Loading
```python
from src.utils.Dataset_Loader import FingerprintDataset

dataset = FingerprintDataset('path/to/labels.csv', train=True)
# Auto-detects dataset type from CSV path
```

### Training
```python
from src.fingerprint.train import train

train('fvc2000')  # or 'casia'
# Supports: --dataset, --resume flags
```

### Testing
```python
from src.fingerprint.test import generate_embeddings, evaluate_model

generate_embeddings('fvc2000')
evaluate_model('fvc2000')
# Modes: generate, evaluate, visualize, all
```

This project focuses on biometric template generation with privacy-preserving techniques and deep learning-based feature extraction.