import os
import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# Set the path to your fingerprint images
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if not (
    os.path.exists(
        os.path.join(
            SCRIPT_DIR, "../../data/SOCOFing/Real"
        )
    )
):
    # Set custom cache directory for kagglehub
    os.environ["KAGGLEHUB_CACHE"] = os.path.join(SCRIPT_DIR, "../../")
    # Download the Dataset
    path = kagglehub.dataset_download("ruizgara/socofing/versions/2")
    print("Path to dataset files:", path)

CurrPath = os.path.join(
    SCRIPT_DIR, "../../data/SOCOFing/Real"
)
DATA_DIR = os.path.normpath(CurrPath)
IMG_SIZE = 224

# 1. Get file paths
image_paths = [
    os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".BMP")
]


# 2. Preprocessing Function
def preprocess_fingerprint(img_path):
 # Read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Resize to 224x224 for ResNet-18
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0

    # Replicate grayscale to 3 channels
    img = np.stack([img, img, img], axis=2)

    # Normalize using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    # Change HWC to CHW format (channels first)
    img = np.transpose(img, (2, 0, 1))

    return img

# 3. Visualize Original vs Preprocessed
def show_sample(idx=3):
    orig = cv2.imread(image_paths[idx], cv2.IMREAD_GRAYSCALE)
    processed = preprocess_fingerprint(image_paths[idx])

    # Convert CHW to HWC for visualization
    processed_vis = np.transpose(processed, (1, 2, 0))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(orig, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Preprocessed")
    plt.imshow(processed_vis)
    plt.show()

# Show a sample
show_sample()