import os

# Base directory of the project it is src folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
LABELS_DIR = os.path.join(DATA_DIR, "labels")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "preprocess")

# CSV paths
FINGERPRINT_TRAIN_CSV = os.path.join(LABELS_DIR, "fingerprint_train.csv")
FINGERPRINT_VAL_CSV = os.path.join(LABELS_DIR, "fingerprint_val.csv")
IRIS_TRAIN_CSV = os.path.join(LABELS_DIR, "iris_train.csv")
IRIS_VAL_CSV = os.path.join(LABELS_DIR, "iris_val.csv")

# Artifacts paths
ARTIFACTS = os.path.abspath(os.path.join(BASE_DIR, "..", "artifacts"))
LOG_DIR = os.path.abspath(os.path.join(ARTIFACTS, "logs"))
TRAINING_LOG_DIR = os.path.join(LOG_DIR, "training")
PLOTS_DIR = os.path.abspath(os.path.join(ARTIFACTS, "plots"))
TENSORBOARD_DIR = os.path.abspath(os.path.join(PLOTS_DIR, "tensorboard"))

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, "models")
SAVED_MODELS_DIR = os.path.join(ARTIFACTS, "models")
FINGERPRINT_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "fingerprint_embedding_model.pth")
IRIS_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "iris_embedding_model.pth")

# utils paths
UTILS_DIR = os.path.join(BASE_DIR, "utils")

# Example Images paths
FINGERPRINT_EX_1_1 = "D:/code/Projects/biometric-template-gen/data/CASIA-dataset/000/R000_R0_0.bmp"

# Make sure folders exist
os.makedirs(TRAINING_LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Hyperparameters
# BATCH_SIZE = 32
# LEARNING_RATE = 0.001
# EPOCHS = 50
VAL_RATIO = 0.2
