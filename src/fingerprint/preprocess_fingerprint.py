"""
Unified Preprocessing Module for Fingerprint Images
Supports: CASIA and FVC2000 datasets
"""

import os
import cv2
import numpy as np
from typing import Optional
from torchvision import transforms

# Config
IMG_SIZE = 224

# Dataset-specific preprocessing configurations
DATASET_CONFIGS = {
    "casia": {
        "use_clahe": True,
        "clahe_clip_limit": 2.0,
        "clahe_grid_size": (8, 8),
        "normalization": "imagenet",  # Use ImageNet stats
    },
    "fvc2000": {
        "use_clahe": True,
        "clahe_clip_limit": 2.0,
        "clahe_grid_size": (8, 8),
        "normalization": "imagenet",
    },
    "default": {
        "use_clahe": True,
        "clahe_clip_limit": 2.0,
        "clahe_grid_size": (8, 8),
        "normalization": "imagenet",
    },
}

# ImageNet normalization stats (standard for ResNet transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True) -> transforms.Compose:
    """Get transforms for training or validation.

    Args:
        train: If True, returns training transforms with augmentation.
              If False, returns validation transforms without augmentation.

    Returns:
        Composed transforms
    """
    if train:
        # Training: Use RandomAffine for augmentation to preserve ridge structure
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                # Rotation +/- 15 deg, and slight translation (shift)
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                # ImageNet Mean/Std
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:
        # Validation: No augmentation
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )


def pad_to_square(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    """
    Resizes an image to target_size while maintaining aspect ratio
    by adding black padding (letterboxing).

    Args:
        img: Input image (grayscale)
        target_size: Target size for both dimensions

    Returns:
        Padded square image
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
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized_img

    return canvas


def apply_clahe(
    img: np.ndarray, clip_limit: float = 2.0, grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        img: Input grayscale image
        clip_limit: Threshold for contrast limiting
        grid_size: Size of grid for histogram equalization

    Returns:
        Enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img)


def preprocess_fingerprint(
    img_path: str,
    train: bool = True,
    dataset_type: Optional[str] = None,
    target_size: int = IMG_SIZE,
) -> transforms.Compose:
    """
    Preprocess a fingerprint image for model input.

    Supports both CASIA and FVC2000 datasets with dataset-specific optimizations.

    Args:
        img_path: Path to the fingerprint image
        train: Whether this is for training (applies augmentations if True)
        dataset_type: Type of dataset ('casia', 'fvc2000', or None for default)
        target_size: Target image size (default: 224 for ResNet)

    Returns:
        Preprocessed image tensor (C, H, W)

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image cannot be loaded
    """
    # Validate input
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at {img_path}")

    # Get dataset configuration
    config = DATASET_CONFIGS.get(dataset_type, DATASET_CONFIGS["default"])

    # Read image in grayscale
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Apply CLAHE if configured (enhances ridge definition)
    if config.get("use_clahe", True):
        img = apply_clahe(
            img,
            clip_limit=config.get("clahe_clip_limit", 2.0),
            grid_size=config.get("clahe_grid_size", (8, 8)),
        )

    # Pad to Square (Preserve Aspect Ratio)
    img = pad_to_square(img, target_size=target_size)

    # Stack to 3 channels (H, W) -> (H, W, 3) for ResNet
    img = np.stack([img, img, img], axis=2)

    # Apply PyTorch Transforms
    transforms_pipeline = get_transforms(train=train)
    img_tensor = transforms_pipeline(img)

    return img_tensor


def preprocess_batch(
    img_paths: list, train: bool = False, dataset_type: Optional[str] = None
) -> transforms.Compose:
    """
    Preprocess a batch of fingerprint images.

    Args:
        img_paths: List of image paths
        train: Whether this is for training
        dataset_type: Type of dataset ('casia', 'fvc2000', or None)

    Returns:
        Batch of preprocessed image tensors (B, C, H, W)
    """
    import torch

    tensors = []
    for img_path in img_paths:
        try:
            tensor = preprocess_fingerprint(
                img_path, train=train, dataset_type=dataset_type
            )
            tensors.append(tensor)
        except Exception as e:
            # Return zero tensor for failed loads
            print(f"Warning: Failed to load {img_path}: {e}")
            tensor = torch.zeros((3, IMG_SIZE, IMG_SIZE))
            tensors.append(tensor)

    return torch.stack(tensors)


# --------------------------
# Visualization Functions
# --------------------------
def denormalize_for_visualization(tensor):
    """
    Denormalize a tensor for visualization.

    Args:
        tensor: Normalized tensor (C, H, W)

    Returns:
        Denormalized numpy array (H, W, C) in range [0, 1]
    """
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    # Convert [C, H, W] -> [H, W, C]
    img = tensor.permute(1, 2, 0).numpy()

    # Reverse normalization
    img = (img * std) + mean
    img = np.clip(img, 0, 1)

    return img


def show_preprocessing_comparison(img_path: str, dataset_type: Optional[str] = None):
    """
    Visualize original vs preprocessed image.

    Args:
        img_path: Path to image
        dataset_type: Type of dataset
    """
    import matplotlib.pyplot as plt

    if not os.path.exists(img_path):
        print(f"Sample image not found: {img_path}")
        return

    # Load raw for comparison
    orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if orig is None:
        print(f"Failed to load image: {img_path}")
        return

    # Process
    processed_tensor = preprocess_fingerprint(
        img_path, train=False, dataset_type=dataset_type
    )

    # Denormalize for visualization
    processed_vis = denormalize_for_visualization(processed_tensor)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title(f"Original ({orig.shape})")
    plt.imshow(orig, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Processed (224x224)")
    plt.imshow(processed_vis)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test preprocessing with example images
    from src.config import FINGERPRINT_EX_CASIA, FINGERPRINT_EX_FVC2000

    print("Testing unified preprocessor...")

    # Test with FVC2000 example
    if os.path.exists(FINGERPRINT_EX_FVC2000):
        print(f"\nTesting with FVC2000 image: {FINGERPRINT_EX_FVC2000}")
        tensor = preprocess_fingerprint(
            FINGERPRINT_EX_FVC2000, train=False, dataset_type="fvc2000"
        )
        print(f"Output shape: {tensor.shape}")
        print(f"Output range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    # Test with CASIA example
    if os.path.exists(FINGERPRINT_EX_CASIA):
        print(f"\nTesting with CASIA image: {FINGERPRINT_EX_CASIA}")
        tensor = preprocess_fingerprint(
            FINGERPRINT_EX_CASIA, train=False, dataset_type="casia"
        )
        print(f"Output shape: {tensor.shape}")
        print(f"Output range: [{tensor.min():.3f}, {tensor.max():.3f}]")

    print("\nPreprocessor test complete!")
