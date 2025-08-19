import cv2
import numpy as np
import torch
from PIL import Image
from src.config import IRIS_EX_1_0, IRIS_EX_2_0
import matplotlib.pyplot as plt

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

def Gaussian_Blur(img):
    """
    Adding Blur removes the "distracting" small edges while keeping the "important" big ones
    """
    return cv2.GaussianBlur(img, (5,5), 0)

def Equalize_Hist(img):
    """
    Equalize the img because Sometimes the unwrapped iris can be a bit dark or have low contrast
    """
    return cv2.equalizeHist(img)

def unwrap_iris(image, pupil_circle, iris_circle, width, height):
    """
    Unwraps the iris region into a rectangular image.

    Args:
        image (np.array): The original grayscale eye image.
        pupil_circle (tuple): A tuple (x, y, r) for the pupil.
        iris_circle (tuple): A tuple (x, y, r) for the iris.
        width (int): The desired width of the output rectangle.
        height (int): The desired height of the output rectangle.

    Returns:
        np.array: The unwrapped, rectangular iris image.
    """
    # 1. Get the center coordinates from the pupil circle
    center_x, center_y, _ = pupil_circle
    
    # 2. Get the radius of the pupil and the iris
    pupil_radius = pupil_circle[2]
    iris_radius = iris_circle[2]
    
    # 3. Create arrays for the angle and radius for every point in the output image
    # Angle (theta) goes from 0 to 2*pi (a full circle)
    thetas = np.linspace(0, 2 * np.pi, width) 
    # Radius (r) goes from the pupil's edge to the iris's edge
    radii = np.linspace(pupil_radius, iris_radius, height)
    
    # 4. Create the mapping grids
    # Create a grid of theta and r values
    theta_grid, r_grid = np.meshgrid(thetas, radii)
    
    # 5. Convert from polar (r, theta) to Cartesian (x, y) coordinates
    # This is the core transformation using sine and cosine
    x_map = center_x + r_grid * np.cos(theta_grid)
    y_map = center_y + r_grid * np.sin(theta_grid)

    # 6. Convert maps to the correct data type for cv2.remap
    x_map_float = x_map.astype(np.float32)
    y_map_float = y_map.astype(np.float32)
    
    # 7. Remap the pixels from the original image to the new rectangular one
    unwrapped = cv2.remap(image, x_map_float, y_map_float, cv2.INTER_LINEAR)
    
    return unwrapped

def hough_circles(img):
    """
    Detect iris and pupil in an eye image using Hough Circle Transform.
    Pupil detection is constrained within the iris.
    """
    img_copy = img.copy()
    iris_circle = None
    pupil_circle = None


    # ---- Detect Iris (outer circle) ----
    iris_circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=45,
        param1=50,
        param2=25,
        minRadius=35,
        maxRadius=70
    )

    if iris_circles is not None:
        iris_circles = np.around(iris_circles[0, :]).astype(int)
        iris_circle = max(iris_circles, key=lambda c: c[2])  # take largest circle
        x_i, y_i, r_i = iris_circle
        cv2.circle(img_copy, (x_i, y_i), r_i, (0, 255, 0), 2)  # green = iris
        cv2.circle(img_copy, (x_i, y_i), 2, (0, 255, 0), 3)

        # ---- Mask image to iris region for pupil detection ----
        mask = np.zeros_like(img)
        cv2.circle(mask, (x_i, y_i), r_i, 255, -1)  # fill iris circle
        iris_roi = cv2.bitwise_and(img, img, mask=mask)

        # ---- Detect Pupil inside iris ----
        pupil_circles = cv2.HoughCircles(
            iris_roi,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=20,     # lower threshold for smaller/darker circle
            minRadius=10,
            maxRadius=int(r_i*0.6)  # pupil should be smaller than iris
        )

        if pupil_circles is not None:
            pupil_circles = np.around(pupil_circles[0, :]).astype(int)
            pupil_circle = min(pupil_circles, key=lambda c: c[2])  # take smallest circle
            x_p, y_p, r_p = pupil_circle
            cv2.circle(img_copy, (x_p, y_p), r_p, (0, 0, 255), 2)  # red = pupil
            cv2.circle(img_copy, (x_p, y_p), 2, (0, 0, 255), 3)

    return img_copy, iris_circle, pupil_circle

def preprocess_image(path, size=(224, 224)):
    org_img = load_image(path)
    blur_img = Gaussian_Blur(org_img)
    _, iris_circle, pupil_circle = hough_circles(blur_img)

    # if iris_circle is None or pupil_circle is None:
    #     print(f"Skipping {path}: circles not detected")
    #     return None  # or raise an exception

    if iris_circle is None:
        res_img = resize_image(org_img, size)

        # Convert to torch tensor
        img_tensor = torch.from_numpy(res_img).float().unsqueeze(0)  # [1,H,W]
        return img_tensor
    
    # Fallback if pupil not detected
    if pupil_circle is None:
        x_i, y_i, r_i = iris_circle
        pupil_circle = (x_i, y_i, int(r_i*0.3))  # approximate pupil

    # Unwrap and resize
    img = unwrap_iris(org_img, pupil_circle, iris_circle, 360, 64)
    res_img = resize_image(img, size)

    # Convert to torch tensor
    img_tensor = torch.from_numpy(res_img).float().unsqueeze(0)  # [1,H,W]
    return img_tensor


def preprocess_and_save(input_path, output_path, size=(224, 224)):
    """
    Preprocess an image and save it as .png for inspection/debugging.
    """
    img_tensor = preprocess_image(input_path, size)  # [1, H, W]
    img_np = (img_tensor.squeeze(0).numpy() * 255).astype(np.uint8)  # back to uint8
    Image.fromarray(img_np).save(output_path)

if __name__ == "__main__":

    org_img = load_image(IRIS_EX_2_0)
    blur_img = Gaussian_Blur(org_img)
    hough_img, iris_circle, pupil_circle  = hough_circles(blur_img)
    
    if iris_circle is None:
        raise ValueError(f"Iris not detected for ")
    
    # Fallback if pupil not detected
    if pupil_circle is None:
        x_i, y_i, r_i = iris_circle
        pupil_circle = (x_i, y_i, int(r_i*0.3))  # approximate pupil

    img = unwrap_iris(org_img,pupil_circle,iris_circle,360,64)
    res_img = resize_image(img, (224,224))
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(org_img, cmap="gray")

    plt.subplot(1, 2, 2)
    plt.title("Edge Img")
    plt.imshow(hough_img, cmap="gray")
    plt.show()
