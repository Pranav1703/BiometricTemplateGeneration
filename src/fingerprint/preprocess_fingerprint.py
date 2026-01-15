import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from src.config import BASE_DIR, FINGERPRINT_EX_1_2

# Config
IMG_SIZE = 224

# 1. Define Transforms
# We define these outside the function to avoid re-initializing them every call

# Training: Use RandomAffine instead of Flip/Crop to preserve ridge structure
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    # Rotation +/- 15 deg, and slight translation (shift) to make it robust
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    # ImageNet Mean/Std (Standard for ResNet transfer learning)
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# Validation: No augmentation, just convert and normalize
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

def pad_to_square(img, target_size=224):
    """
    Resizes an image to target_size while maintaining aspect ratio 
    by adding black padding (letterboxing).
    """
    h, w = img.shape[:2]
    
    # Calculate scaling factor to fit the largest side
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize the image using high-quality interpolation
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a black canvas
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # Calculate centering offsets
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    
    # Paste resized image into center of canvas
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized_img
    
    return canvas

# 2. Main Preprocessing Function
def preprocess_fingerprint(img_path, train=True):
    """
    Loads, enhances (CLAHE), pads, and normalizes a fingerprint image.
    """
    # Read image in grayscale
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")
        
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This enhances ridge definition, crucial for FVC datasets
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    # Pad to Square (Preserve Aspect Ratio)
    img = pad_to_square(img, target_size=IMG_SIZE)

    # Stack to 3 channels (H, W) -> (H, W, 3) for ResNet
    img = np.stack([img, img, img], axis=2)

    # Apply PyTorch Transforms
    if train:
        img_tensor = train_transforms(img)
    else:
        img_tensor = val_transforms(img)

    return img_tensor

# 3. Visualization for Debugging
def show_sample(img_path_str):
    if not os.path.exists(img_path_str):
        print(f"Sample image not found: {img_path_str}")
        return

    # Load raw for comparison
    orig = cv2.imread(img_path_str, cv2.IMREAD_GRAYSCALE)
    
    # Process
    processed_tensor = preprocess_fingerprint(img_path_str, train=False)
    
    # Denormalize for visualization: (Tensor * std) + mean
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Convert [C, H, W] -> [H, W, C]
    processed_vis = processed_tensor.permute(1, 2, 0).numpy()
    
    # Reverse normalization to see actual pixels
    processed_vis = (processed_vis * std) + mean
    processed_vis = np.clip(processed_vis, 0, 1)

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original ({orig.shape})")
    plt.imshow(orig, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Processed (Padded) {processed_vis.shape}")
    plt.imshow(processed_vis)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test with a dummy path or import from config
    # Replace this string with a real path to an FVC image on your disk to test
    SAMPLE_PATH = FINGERPRINT_EX_1_2
    show_sample(SAMPLE_PATH)