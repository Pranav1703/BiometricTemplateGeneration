import cv2
import numpy as np
import torch
from PIL import Image

def load_image(path):
    """
    Load an image from a file path as grayscale.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    return img

def resize_image(img, size=(224, 224)):
    """
    Resize the image to given size (width, height).
    """
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def normalize_image(img):
    """
    Normalize image pixels to range [0,1].
    """
    img = img.astype(np.float32) / 255.0
    return img

def preprocess_image(path, size=(224, 224)):
    """
    Full preprocessing pipeline:
      1. Load grayscale image
      2. Resize
      3. Normalize
      4. Convert to torch tensor with channel dim [1, H, W]
    """
    img = load_image(path)
    img = resize_image(img, size)
    img = normalize_image(img)

    # Convert to tensor with channel dim
    img_tensor = torch.from_numpy(img).float().unsqueeze(0)  # [1, H, W]
    return img_tensor

def preprocess_and_save(input_path, output_path, size=(224, 224)):
    """
    Preprocess an image and save it as .png for inspection/debugging.
    """
    img_tensor = preprocess_image(input_path, size)  # [1, H, W]
    img_np = (img_tensor.squeeze(0).numpy() * 255).astype(np.uint8)  # back to uint8
    Image.fromarray(img_np).save(output_path)
