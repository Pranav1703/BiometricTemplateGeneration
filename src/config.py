"""
Docstring for src.config
to create the folder structure run this cmd:
python -m src.config
"""

import os

# Base directory of the project (project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths (dataset directories expected to be at project root)
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")
LABELS_DIR = os.path.join(DATA_DIR, "labels")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "preprocess")

# Dataset paths
CASIA_DIR = os.path.join(DATA_DIR, "CASIA-dataset")
FVC2000_DIR = os.path.join(DATA_DIR, "FVC2000")
FVC2004_DIR = os.path.join(DATA_DIR, "FVC2004")
CMBD_root_DIR = os.path.join(DATA_DIR, "CMBD")
CMBD_DIR = os.path.join(CMBD_root_DIR, "fp")

FVC2000_LABELS_DIR = os.path.join(LABELS_DIR, "FVC2000_labels")
FVC2000_DB1A_DIR = os.path.join(FVC2000_DIR, "Db1_a")

# FVC2004 dataset path (using Db1_a only)
FVC2004_LABELS_DIR = os.path.join(LABELS_DIR, "FVC2004_labels")
FVC2004_DB1A_DIR = os.path.join(FVC2004_DIR, "DB1_A")

# CSV paths for FVC2000 dataset
FVC2000_TRAIN_CSV = os.path.join(FVC2000_LABELS_DIR, "fvc2000_train.csv")
FVC2000_VAL_CSV = os.path.join(FVC2000_LABELS_DIR, "fvc2000_val.csv")

# CSV paths for FVC2004 dataset
FVC2004_TRAIN_CSV = os.path.join(FVC2004_LABELS_DIR, "fvc2004_train.csv")
FVC2004_VAL_CSV = os.path.join(FVC2004_LABELS_DIR, "fvc2004_val.csv")

# CSV paths for CASIA dataset (if available)
CASIA_LABELS_DIR = os.path.join(LABELS_DIR, "CASIA_labels")
CASIA_TRAIN_CSV = os.path.join(CASIA_LABELS_DIR, "casia_train.csv")
CASIA_VAL_CSV = os.path.join(CASIA_LABELS_DIR, "casia_val.csv")

# CSV paths for CMBD dataset (if available)
CMBD_LABELS_DIR = os.path.join(LABELS_DIR, "CMBD_labels")
CMBD_TRAIN_CSV = os.path.join(CMBD_LABELS_DIR, "CMBD_train.csv")
CMBD_VAL_CSV = os.path.join(CMBD_LABELS_DIR, "CMBD_val.csv")

# Artifacts paths (outputs and saved files)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
LOG_DIR = os.path.join(ARTIFACTS_DIR, "logs")
TRAINING_LOG_DIR = os.path.join(LOG_DIR, "training")
ANALYSIS_LOG_DIR = os.path.join(LOG_DIR, "analysis")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
TENSORBOARD_DIR = os.path.join(PLOTS_DIR, "tensorboard")
EMBEDDING_DIR = os.path.join(ARTIFACTS_DIR, "embedding")
METRICS_DIR = os.path.join(ARTIFACTS_DIR, "metrics")
SCORES_DIR = os.path.join(ARTIFACTS_DIR, "scores")

# Model paths
SAVED_MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")

# Documentation paths
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")

# Utility paths (now consolidated in src/utils)
UTILS_DIR = os.path.join(SRC_DIR, "utils")

# Model component paths
FINGERPRINT_DIR = os.path.join(SRC_DIR, "fingerprint")
CRYPTO_CORE_DIR = os.path.join(FINGERPRINT_DIR, "core")

# Example Images paths
FINGERPRINT_EX_CASIA = os.path.join(CASIA_DIR, "000", "R000_R0_0.bmp")
FINGERPRINT_EX_FVC2000 = os.path.join(FVC2000_DB1A_DIR, "1_1.tif")

# Google Drive model download settings
GOOGLE_DRIVE_MODELS_URL = "https://drive.google.com/drive/folders/1hh4CHY4jFk8gJhsPziOKqKNlbuW4nD08?usp=sharing"

# FVC2000 4-Finger Configuration (2 fingers per person for testing)
# Format: {person_id}_{finger_id}.tif where finger_id 1-8
# Using fingers 1,2 for enrollment (gallery) and 3,4 for verification (probe)
FVC2000_ENROLLMENT_FINGERS = [1, 2]  # Fingers used for enrollment
FVC2000_VERIFICATION_FINGERS = [3, 4]  # Fingers used for verification
FVC2000_NUM_PERSONS = 100  # Total persons in FVC2000 Db1_a

# FVC2004 Configuration
# Format: {person_id}_{finger_id}.tif where finger_id 1-8
FVC2004_ENROLLMENT_FINGERS = [1, 2]  # Fingers used for enrollment
FVC2004_VERIFICATION_FINGERS = [3, 4]  # Fingers used for verification
FVC2004_NUM_PERSONS = 100  # Total persons in FVC2004
FVC2004_DEFAULT_DB = "Db1_a"  # Default subset

# Cancelable Transform Parameters
DEFAULT_CANCELABLE_ALPHA = 0.6  # Blend factor: R = alpha * R_bio + (1-alpha) * R_key
DEFAULT_SIMILARITY_THRESHOLD = 0.5  # Threshold for cosine similarity authentication

# PBKDF2 Parameters
PBKDF2_ITERATIONS = 10000
PBKDF2_KEY_LENGTH = 32  # 256-bit key
PBKDF2_SALT_LENGTH = 16  # 128-bit salt

# Configuration constants
EMBEDDING_DIM = 512
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# Make sure essential folders exist
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(FVC2000_LABELS_DIR, exist_ok=True)
os.makedirs(FVC2004_LABELS_DIR, exist_ok=True)
os.makedirs(CASIA_LABELS_DIR, exist_ok=True)
os.makedirs(CMBD_LABELS_DIR, exist_ok=True)
os.makedirs(TRAINING_LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(SCORES_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
