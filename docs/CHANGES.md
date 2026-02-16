# Project Changes Documentation

This document details all the changes made to the Biometric Template Generation project, organized by phase and date.

---

## Overview

This project implements a biometric cryptosystem combining cancelable biometrics and fuzzy commitment for privacy-preserving fingerprint authentication. The system generates cryptographic keys from biometric embeddings while providing template protection.

---

## Phase 1: Core Infrastructure ✅ COMPLETED

### Tasks Completed:
1. **Crypto Utilities Package** - Created comprehensive cryptographic utility functions

### Files Created/Modified:

#### `src/utils/hash_utils.py`
- Implements SHA-256 hashing for key verification
- Function: `get_sha256_hash(data: bytes) -> bytes`

#### `src/utils/xor_utils.py`
- Byte-wise XOR operations for helper data computation
- Function: `xor_bytes(a: bytes, b: bytes) -> bytes`

#### `src/utils/quantization.py`
- Embedding quantization (float32 → bytes)
- Functions: ` `dequantizequantize_embedding()`,_embedding()`
- MSE < 0.01 as required

#### `src/utils/ecc_utils.py`
- Reed-Solomon Error Correction Code (ECC) wrapper
- Class: `ECCWrapper`
- Supports configurable key size and error capacity (default 20%)
- Can correct up to 6-7 byte errors

#### `tests/test_crypto_utils.py`
- Unit tests for all crypto utilities
- Tests for:
  - SHA-256 determinism
  - XOR correctness
  - Quantization accuracy (MSE < 0.01)
  - ECC error correction within capacity
  - ECC failure beyond capacity

---

## Phase 2: Cancelable Transform ✅ COMPLETED

### Tasks Completed:
1. **CancelableTransform Class** - Key-driven random projection for revocability

### Files Created/Modified:

#### `src/fingerprint/core/cancelable_transform.py`
- Class: `CancelableTransform`
- Transformation: `R = α * R_bio + (1-α) * R_key`
- Features:
  - Revocability: Different keys produce different transforms
  - Unlinkability: Different apps → different keys
  - Non-invertibility: Without key, hard to reverse
- Methods:
  - `enroll(embedding, user_key)` → (transformed, params)
  - `verify(embedding, params)` → transformed
  - `cancel_and_reissue()` → new template

#### `tests/test_phase2.py`
- Unit tests for CancelableTransform
- Tests for:
  - Enrollment produces consistent output
  - Same embedding + same key → same transform
  - Same embedding + different key → different transform
  - Cancellation and reissuance

---

## Phase 3: Fuzzy Commitment ✅ COMPLETED

### Tasks Completed:
1. **FuzzyCommitment Class** - Implemented from scratch

### Files Created:

#### `src/fingerprint/core/fuzzy_commitment.py` (NEW)
- Class: `FuzzyCommitment`
- Based on Juels & Wattenberg (1999) scheme
- Features:
  - Non-invertibility: Helper data reveals nothing
  - Cryptographic binding: Key bound to biometric
  - Error tolerance: Reed-Solomon corrects quantization errors
  
- Methods:
  - `enroll(embedding)` → (hash_key, helper_data)
  - `verify(embedding, hash_key, helper_data)` → (success, key)
  
- Template Format:
  ```python
  {
      'hash_key': str,      # SHA-256(key) hex string
      'helper_data': bytes, # δ = quantized_x ⊕ codeword
  }
  ```

### Files Modified:

#### `src/fingerprint/core/biometric_crypto_system.py`
- Fixed imports to use correct module paths
- Updated to work with new FuzzyCommitment class
- Cleaned up deprecated `self.ecc` references

---

## Phase 4: System Integration ✅ COMPLETED

### Tasks Completed:
1. **BiometricCryptoSystem Class** - Full integration of cancelable + fuzzy commitment
2. **Integration Tests** - Comprehensive end-to-end testing
3. **Embedding Generation Integration** - Complete pipeline

### Files Created/Modified:

#### `src/fingerprint/core/biometric_crypto_system.py`
- Complete enrollment workflow: Raw Embedding → Cancelable → Fuzzy Commitment
- Complete verification workflow: Query → Cancelable → Fuzzy Commitment Verify
- Template cancellation: `cancel_and_reissue()` method
- Methods:
  - `enroll(raw_embedding, user_key)` → Protected template
  - `verify(raw_embedding, template, user_key)` → (success, key)
  - `cancel_and_reissue(embedding, old_key, new_key)` → New template
  - `get_system_info()` → Configuration dict

#### `tests/integration_phase2.py`
- Multi-user enrollment test
- Authentication test (exact embedding, noisy embedding, wrong key)
- Template cancellation test
- Performance metrics

#### `src/fingerprint/inference/gen_embeddings.py` (UPDATED)
- Now supports both FVC2000 and CASIA datasets
- Command-line interface for dataset selection
- Usage:
  ```bash
  python -m src.fingerprint.inference.gen_embeddings --dataset fvc2000
  python -m src.fingerprint.inference.gen_embeddings --dataset casia
  ```

#### `src/fingerprint/inference/crypto_inference.py` (NEW)
- Complete biometric cryptosystem integration
- User enrollment with templates
- User verification
- Usage:
  ```bash
  python -m src.fingerprint.inference.crypto_inference enroll --dataset fvc2000 --user-key "app_001"
  python -m src.fingerprint.inference.crypto_inference verify --dataset fvc2000 --user-key "app_001" --sample 0
  ```

### Acceptance Criteria Met:
- ✅ Full enrollment works end-to-end
- ✅ Verification with same sample succeeds
- ✅ Verification with noisy sample succeeds (within tolerance)
- ✅ Template cancellation works
- ✅ Integration with existing codebase complete

---

## Project Reorganization

### Directory Structure Updates:

#### Before:
```
src/
├── fingerprint/
│   ├── train.py           # FVC2000 training
│   ├── casia_train.py     # CASIA training
│   ├── test.py
│   ├── preprocess_fingerprint.py
│   ├── casia_preprocess.py
│   ├── gen_labels.py
│   ├── casia_genlabels.py
│   └── crypto_utils/
│       ├── hash_utils.py
│       ├── xor_utils.py
│       ├── quantization.py
│       └── ecc_utils.py
```

#### After:
```
src/
├── config.py              # Centralized config + auto-folder creation
├── fingerprint/
│   ├── train.py          # Unified training (--dataset flag)
│   ├── test.py          # Unified testing (--dataset, --mode flags)
│   ├── core/            # Biometric crypto system
│   │   ├── cancelable_transform.py
│   │   ├── fuzzy_commitment.py (NEW)
│   │   ├── biometric_crypto_system.py
│   │   └── biometric_key_generator.py
│   ├── inference/
│   └── benchmarks/
└── utils/               # Consolidated utilities
    ├── Dataset_Loader.py
    ├── preprocess_fingerprint.py
    ├── gen_labels.py
    ├── model_downloader.py
    ├── hash_utils.py
    ├── xor_utils.py
    ├── quantization.py
    ├── ecc_utils.py
    └── logger.py
```

---

## Code Unification

### Training Scripts:

#### Old (Separate Files):
- `src/fingerprint/train.py` - FVC2000 only
- `src/fingerprint/casia_train.py` - CASIA only

#### New (Unified):
```bash
python -m src.fingerprint.train --dataset fvc2000
python -m src.fingerprint.train --dataset casia
python -m src.fingerprint.train --dataset fvc2000 --resume path/to/model.pth
```

### Testing Scripts:

#### Old:
- Multiple separate test files

#### New (Unified):
```bash
python -m src.fingerprint.test --dataset fvc2000 --mode generate
python -m src.fingerprint.test --dataset fvc2000 --mode evaluate
python -m src.fingerprint.test --dataset fvc2000 --mode visualize
python -m src.fingerprint.test --dataset fvc2000 --mode all
```

### Label Generation:

#### Old:
- `src/utils/gen_labels.py` - FVC2000 only
- `src/casia_genlabels.py` - CASIA only

#### New (Unified):
```bash
python -m src.utils.gen_labels --dataset fvc2000
python -m src.utils.gen_labels --dataset casia
```

---

## Configuration Updates

### `src/config.py`:

1. **Removed Kaggle dataset references** (no longer used)
2. **Added CASIA dataset paths**
3. **Added auto-folder creation** when importing config
4. **Added model download settings**

### Auto-Created Directories:
```bash
python -m src.config
# Creates:
# - datasets/labels/FVC2000_labels/
# - datasets/labels/CASIA_labels/
# - datasets/FVC2000/
# - datasets/CASIA-dataset/
# - artifacts/models/
# - artifacts/logs/
# - artifacts/plots/
```

---

## Documentation Updates

### `README.md`:
- Updated project structure
- Added dataset download links:
  - FVC2000: http://bias.csr.unibo.it/fvc2000/download.asp
  - CASIA: https://drive.google.com/drive/folders/1yFb8jmAO72nIamSHVKkXtmCY5I52iRCX
- Updated all command examples to use module format (`python -m`)

### `AGENTS.md`:
- Comprehensive agent guidelines
- Build/test commands
- Code style guidelines
- Common patterns

---

## Dataset Structure

### Expected Layout:
```
datasets/
├── labels/
│   ├── FVC2000_labels/
│   │   ├── fvc2000_train.csv
│   │   └── fvc2000_val.csv
│   └── CASIA_labels/
│       ├── casia_train.csv
│       └── casia_val.csv
├── FVC2000/
│   └── DB1_a/           # FVC2000 fingerprint images
└── CASIA-dataset/       # CASIA fingerprint images
```

---

## Model Management

### Google Drive Integration:

- URL: https://drive.google.com/drive/folders/1hh4CHY4jFk8gJhsPziOKqKNlbuW4nD08
- Auto-download when models not found
- Pattern: `*_arcface_model.pth`

### Usage:
```bash
python -m src.utils.model_downloader
```

---

## Running the Project

### Quick Start:
```bash
# 1. Setup environment
conda env create -f envs/environment-cpu.yml
conda activate biometric-env
pip install pytest gdown

# 2. Auto-create folders
python -m src.config

# 3. Download datasets
# (See README for download links)

# 4. Generate labels
python -m src.utils.gen_labels --dataset fvc2000
python -m src.utils.gen_labels --dataset casia

# 5. Train
python -m src.fingerprint.train --dataset fvc2000

# 6. Test
python -m src.fingerprint.test --dataset fvc2000 --mode all
```

---

## Known Issues Fixed

1. **Missing FuzzyCommitment class** - Implemented
2. **Wrong import paths** - Fixed in:
   - `biometric_crypto_system.py`
   - `test_phase2.py`
   - `integration_phase2.py`
3. **Duplicate config paths** - Cleaned up
4. **Old Kaggle references** - Removed

---

## Next Steps (Phase 4-6)

1. **Phase 4**: Complete system integration testing
2. **Phase 5**: Benchmark all three methods (raw, cancelable, fuzzy commitment)
3. **Phase 6**: Analysis and paper preparation

---

## Contact

For questions about these changes, refer to:
- `docs/plan.txt` - Detailed implementation plan
- `README.md` - Usage instructions
- `AGENTS.md` - Developer guidelines
