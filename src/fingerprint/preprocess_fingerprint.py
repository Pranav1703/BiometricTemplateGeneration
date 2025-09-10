import os
import cv2
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from src.config import BASE_DIR, FINGERPRINT_EX_1_1
# # Set the path to your fingerprint images
SCRIPT_DIR = BASE_DIR


# CurrPath = os.path.join(
#     SCRIPT_DIR, "../../data/SOCOFing/Real"
# )
# DATA_DIR = os.path.normpath(CurrPath)
IMG_SIZE = 224

# # 1. Get file paths
# image_paths = [
#     os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".BMP")
# ]

# Training transforms (augmentation + normalization)
train_transforms = transforms.Compose([
    transforms.ToPILImage(),                     # Convert numpy array to PIL
    transforms.RandomRotation(10),               # Random rotation ±10 degrees
    transforms.RandomHorizontalFlip(),           # Random horizontal flip
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
    transforms.ToTensor(),                       # Convert to tensor (C,H,W)
    transforms.Normalize([0.485, 0.456, 0.406], # ImageNet mean
                         [0.229, 0.224, 0.225]) # ImageNet std
])

# Validation transforms (no augmentation)
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# 2. Preprocessing Function
def preprocess_fingerprint(img_path, train=True):
    # Read image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Resize to 224x224
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Convert to float32 and normalize to [0,1]
    img = img.astype(np.float32) / 255.0

    # Replicate grayscale to 3 channels
    img = np.stack([img, img, img], axis=2)

    # Convert HWC to CHW
    img = np.transpose(img, (2, 0, 1))

    # Apply augmentation / normalization using transforms
    img = img.transpose(1, 2, 0)  # CHW -> HWC for PIL
    if train:
        img_tensor = train_transforms(img)
    else:
        img_tensor = val_transforms(img)

    return img_tensor

# # 3. Visualize Original vs Preprocessed
def show_sample():
    orig = cv2.imread(FINGERPRINT_EX_1_1, cv2.IMREAD_GRAYSCALE)
    processed = preprocess_fingerprint(FINGERPRINT_EX_1_1)

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

# # Show a sample
if __name__ == "__main__":
    show_sample()