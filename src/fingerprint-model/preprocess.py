import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Set the path to your fingerprint images
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CurrPath = os.path.join(SCRIPT_DIR, "../../data/SOCOFing/Real")
DATA_DIR = os.path.normpath(CurrPath)
IMG_SIZE = 128

# 1. Get file paths
image_paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".BMP")]

# 2. Preprocessing Function
def preprocess_fingerprint(img_path):
    # Read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Normalize to [0,1]
    img = img.astype(np.float32) / 255.0

    return img

# 3. Visualize Original vs Preprocessed
def show_sample(idx=0):
    orig = cv2.imread(image_paths[idx], cv2.IMREAD_GRAYSCALE)
    processed = preprocess_fingerprint(image_paths[idx])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(orig, cmap='gray')
    
    plt.subplot(1, 2, 2)
    plt.title("Preprocessed")
    plt.imshow(processed, cmap='gray')
    
    plt.show()

# Show a sample
show_sample(0)
