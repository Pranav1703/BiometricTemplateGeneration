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
│   ├── embedding/             # Generated embeddings
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
│   │   │   ├── biometric_crypto_system.py
│   │   │   ├── cancelable_transform.py
│   │   │   └── fuzzy_commitment.py
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

### Dataset Preparation
```bash
python -m src.utils.gen_labels --dataset fvc2000
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
python -m src.fingerprint.test --dataset fvc2000 --mode generate
python -m src.fingerprint.test --dataset fvc2000 --mode evaluate
python -m src.fingerprint.test --dataset fvc2000 --mode visualize
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

# Run with detailed output on failures
python -m pytest tests/ -v --tb=long

# Run tests matching a pattern
python -m pytest tests/ -k "test_sha256"

# Stop on first failure
python -m pytest tests/ -x
```

### Running Benchmarks
```bash
# Run cancelable biometric benchmark (FVC2000 embeddings)
python -m src.fingerprint.benchmarks.cancelable_benchmark
```

### Model Management
```bash
python -m src.utils.model_downloader
```

## Code Style Guidelines

### Import Style
- Group imports in order: stdlib, third-party, local
- Use absolute imports: `from src.config import FVC2000_TRAIN_CSV`
- Avoid wildcard imports: `from module import *`
- Keep imports at file top, alphabetical within groups
- Use try-except for optional dependencies (e.g., torch, cryptography)

### Naming Conventions
- **Variables/functions**: `snake_case` (e.g., `preprocess_fingerprint`)
- **Classes**: `PascalCase` (e.g., `FingerprintEmbeddingNet`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `EMBEDDING_DIM`)
- **Private members**: `_single_underscore` prefix
- **Config variables**: `UPPER_SNAKE_CASE` in config files

### Type Hints
- Use type hints for all function signatures
- Example: `def get_logger(name: str = "train") -> logging.Logger`
- Import types from `typing` when needed (Tuple, Dict, List, Optional)
- Use `np.ndarray` for numpy arrays, `torch.Tensor` for PyTorch

### File Structure
- Source code in `src/`
- Utilities in `src/utils/`
- Tests in `tests/` at root
- Artifacts in `artifacts/`
- Use `__init__.py` for proper package structure

### Formatting
- Maximum line length: 100 characters
- Use 4 spaces for indentation (not tabs)
- Add blank lines between functions and classes
- Use Black-style formatting for clarity

### Error Handling
- Try-except for file I/O operations
- Validate input types and ranges at function entry
- Raise descriptive exceptions with context
- Use assertions in tests for expected conditions
- Log warnings instead of silent failures when appropriate

### Logging
- Use centralized logger: `from src.utils.logger import get_logger`
- Descriptive logger names: `logger = get_logger("fingerprint_train")`
- Log levels: DEBUG (details), INFO (progress), WARNING/ERROR (issues)
- Avoid excessive logging that slows down execution

### Configuration
- Centralized in `src/config.py`
- Use `os.path.join()` for cross-platform paths
- Create directories with `os.makedirs(dir, exist_ok=True)`
- Use pathlib for modern path handling when appropriate

### Deep Learning Best Practices
- Device-agnostic: `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- L2 normalize embeddings: `nn.functional.normalize(x, p=2, dim=1)`
- Proper DataLoader with appropriate batch size
- Save models with state dictionaries: `torch.save(model.state_dict(), path)`
- Use `model.eval()` and `with torch.no_grad():` for inference

### Biometric Patterns
- L2 normalization for unit hypersphere embeddings
- Cosine similarity for comparison: `np.dot(a, b) / (||a|| * ||b||)`
- Dataset splitting to avoid data leakage
- Cancelable transform: `R = α × R_bio + (1-α) × R_key`

### Testing Guidelines
- Descriptive test names: `test_sha256_determinism`
- Use pytest fixtures for reusable test data
- Test edge cases and error conditions
- Test crypto functions with known inputs
- Group related tests in classes

### Security
- Never hardcode secrets/API keys in source code
- Use `os.urandom()` for cryptographic randomness
- Validate inputs in preprocessing functions
- Follow secure coding practices for biometric data
- Use PBKDF2 for key derivation (not fuzzy commitment for deep embeddings)

## Common Patterns

### Cancelable Biometrics (Recommended)
```python
from src.fingerprint.core.cancelable_transform import CancelableTransform

transform = CancelableTransform(embedding_dim=512, alpha=0.6)

# Enrollment
template, params = transform.enroll_with_key(embedding, user_key)

# Verification
success, key = transform.verify_with_key(query_embedding, template, params)
```

### Dataset Loading
```python
from src.utils.Dataset_Loader import FingerprintDataset

dataset = FingerprintDataset('path/to/labels.csv', train=True)
```

### Running Full Pipeline
```bash
# 1. Train model
python -m src.fingerprint.train --dataset fvc2000

# 2. Generate embeddings
python -m src.fingerprint.test --dataset fvc2000 --mode generate

# 3. Run benchmarks
python -m src.fingerprint.benchmarks.cancelable_benchmark
```

This project focuses on biometric template generation with privacy-preserving techniques using **Cancelable Biometrics + PBKDF2** (not fuzzy commitment, which fails with high-dimensional deep embeddings).