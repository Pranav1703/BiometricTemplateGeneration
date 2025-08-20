import os

# Base directory of the project it is src folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
DATASET_DIR = os.path.join(DATA_DIR, "IRIS and FINGERPRINT DATASET")
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
FINGERPRINT_EX_1_0 = os.path.join(DATASET_DIR, "1", "Fingerprint", "1__M_Left_middle_finger.BMP")
FINGERPRINT_EX_1_1 = os.path.join(DATASET_DIR, "1", "Fingerprint", "1__M_Right_middle_finger.BMP")
FINGERPRINT_EX_2_0 = os.path.join(DATASET_DIR, "2", "Fingerprint", "1__M_Left_middle_finger.BMP")
FINGERPRINT_EX_2_1 = os.path.join(DATASET_DIR, "2", "Fingerprint", "1__M_Right_middle_finger.BMP")
IRIS_EX_1_0 = os.path.join(DATASET_DIR, "1", "left", "aeval1.bmp")
IRIS_EX_1_1 = os.path.join(DATASET_DIR, "1", "right", "aevar1.bmp")
IRIS_EX_2_0 = os.path.join(DATASET_DIR, "30", "left", "roslil4.bmp")
IRIS_EX_2_1 = os.path.join(DATASET_DIR, "2", "right", "aevar1.bmp")

# Make sure folders exist
os.makedirs(TRAINING_LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(FINGERPRINT_MODEL_PATH, exist_ok=True)
os.makedirs(IRIS_MODEL_PATH, exist_ok=True)

# Hyperparameters
# BATCH_SIZE = 32
# LEARNING_RATE = 0.001
# EPOCHS = 50
VAL_RATIO = 0.2
