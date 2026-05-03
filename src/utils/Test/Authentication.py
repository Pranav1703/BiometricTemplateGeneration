import os
import sys
import hashlib
import numpy as np
import cv2
import torch
from pathlib import Path

# Make sure the project root is in the Python path so you can import src
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Your existing modules
from src.config import (
    CASIA_DIR, CASIA_LABELS_DIR, CASIA_TRAIN_CSV, SAVED_MODELS_DIR, SCORES_DIR, PLOTS_DIR
)
from src.fingerprint.preprocess_fingerprint import preprocess_fingerprint
from src.fingerprint.train import FingerprintEmbeddingNet, EMBEDDING_DIM, DEVICE
from src.fingerprint.gen_embeddings import generate_chaotic_projection, apply_bio_hash
from src.Dataset_Loader import FingerprintDataset

# BCH library (install with: pip install bchlib)
import bchlib

# ----------------------------------------------------------------------
# 1. BCH configuration (511 bits, matches standard BCH(511, t=30))
# ----------------------------------------------------------------------
BCH_POLY = 529               # generator polynomial for BCH(511,30)
BCH_T = 30                   # maximum correctable errors
BCH_DATA_BYTES = 30          # bytes of data (2*30 = 240 bits of info)
BCH_CODE_LENGTH = 511        # codeword length (2**9 - 1)

bch = bchlib.BCH(BCH_POLY, BCH_DATA_BYTES)

# ----------------------------------------------------------------------
# 2. Helper: generate a random secret key, hash, and BCH encode
# ----------------------------------------------------------------------
def generate_secret_key(num_bytes=16):
    """Random 128‑bit key."""
    return os.urandom(num_bytes)

def hash_secret(s: bytes) -> str:
    return hashlib.sha256(s).hexdigest()

def encode_secret(s: bytes) -> np.ndarray:
    """BCH‑encode a secret key into a 511‑bit codeword."""
    # Pad key to the BCH data length (30 bytes)
    data_bytes = s.ljust(BCH_DATA_BYTES, b'\x00')
    ecc = bch.encode(data_bytes)               # returns error correction bytes
    codeword_bytes = data_bytes + ecc          # total length: 30 + ecc_bytes
    # Convert to bit array of length 511
    bits = np.unpackbits(np.frombuffer(codeword_bytes, dtype=np.uint8))[:BCH_CODE_LENGTH]
    return bits.astype(np.uint8)

def decode_secret(codeword_bits: np.ndarray):
    """BCH‑decode a 511‑bit codeword, returning (secret_bytes, num_errors, success)."""
    # Convert bits back to bytes
    codeword_bytes = np.packbits(codeword_bits.astype(np.uint8)).tobytes()
    try:
        data_bytes, _, nerr = bch.decode(codeword_bytes)
        # Trim to original 16 bytes (the rest is padding)
        s = data_bytes[:16]
        return s, nerr, True
    except Exception:
        # Decoding failure (too many errors)
        return None, None, False

# ----------------------------------------------------------------------
# 3. Model loading (CASIA backbone only)
# ----------------------------------------------------------------------
def load_casia_model():
    """Load the frozen CASIA ResNet‑50 backbone."""
    model_path = os.path.join(SAVED_MODELS_DIR, "casia_arcface_model_quantized_100_128bs.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train it first.")
    backbone = FingerprintEmbeddingNet(EMBEDDING_DIM).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    backbone.load_state_dict(checkpoint["backbone"])
    backbone.eval()
    return backbone

# ----------------------------------------------------------------------
# 4. Obtain the protected template T_prot from an image path
# ----------------------------------------------------------------------
def get_protected_template(image_path: str, user_key: str, backbone: FingerprintEmbeddingNet):
    """
    Preprocess image, extract raw binary bits, then apply L2FE‑Hash (cancelable).
    Returns a 511‑bit binary template (first 511 of the 512‑bit output).
    """
    # Preprocess (using dataset_type='casia' for correct CLAHE settings)
    img_tensor = preprocess_fingerprint(
        image_path, train=False, dataset_type="casia"
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        quantized_bits, _ = backbone(img_tensor)   # (1, 512)
    raw_bits = quantized_bits.cpu().squeeze().numpy().astype(np.uint8)

    # Cancelable transform: chaotic projection + median threshold
    R_key = generate_chaotic_projection(user_key)   # Orthogonal matrix on GPU
    # Move raw bits to GPU for matrix multiplication
    raw_tensor = quantized_bits.float()             # (1,512)
    projected = torch.matmul(R_key, raw_tensor.T)   # (512,1)
    prot_bits = apply_bio_hash(projected.T).squeeze().cpu().numpy().astype(np.uint8)

    # Take first 511 bits (BCH works with 511)
    return prot_bits[:BCH_CODE_LENGTH]

# ----------------------------------------------------------------------
# 5. Distortion functions: cuts (lines) and dirt (dots)
# ----------------------------------------------------------------------
def add_cuts(image: np.ndarray, num_lines: int, line_thickness: int = 2):
    """
    Draw random dark lines on a grayscale fingerprint to simulate cuts.
    image: grayscale (H,W) uint8
    """
    out = image.copy()
    h, w = image.shape
    for _ in range(num_lines):
        pt1 = (np.random.randint(0, w), np.random.randint(0, h))
        pt2 = (np.random.randint(0, w), np.random.randint(0, h))
        cv2.line(out, pt1, pt2, (0,), thickness=line_thickness)   # black line
    return out

def add_dots(image: np.ndarray, num_dots: int, dot_radius: int = 2):
    """
    Draw random black dots (dirt) on a grayscale fingerprint.
    """
    out = image.copy()
    h, w = image.shape
    for _ in range(num_dots):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        cv2.circle(out, (cx, cy), dot_radius, (0,), thickness=-1)  # filled circle
    return out

# ----------------------------------------------------------------------
# 6. Main demonstration
# ----------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # 6.1 Choose a sample CASIA image (first available)
    # ------------------------------------------------------------------
    print("Looking for a CASIA image...")
    # Let's pick one from the validation set (or training)
    val_csv = CASIA_TRAIN_CSV   # or CASIA_VAL_CSV, both exist
    if not os.path.exists(val_csv):
        raise FileNotFoundError(f"Label file not found: {val_csv}. Run gen_labels.py first.")
    dataset = FingerprintDataset(val_csv, train=False, dataset_type="casia")
    if len(dataset) == 0:
        raise RuntimeError("No samples in dataset.")
    # Get the first sample’s path and label
    first_sample = dataset.samples[0]
    img_path = first_sample[0]               # full path
    user_label = first_sample[1]             # integer ID
    print(f"Using image: {img_path}\n")

    # Load model
    backbone = load_casia_model()

    # ------------------------------------------------------------------
    # 6.2 Enrolment
    # ------------------------------------------------------------------
    # Define a user‑specific key (e.g., derived from the person’s ID + a master password)
    user_key = f"casia_user_{user_label}_secret_key"
    print("1. Enrolment")
    T_prot = get_protected_template(img_path, user_key, backbone)
    print(f"   Protected template (first 50 bits): {T_prot[:50]}...")

    # Generate secret key S and encode
    S = generate_secret_key(16)
    S_hash = hash_secret(S)
    C = encode_secret(S)                     # 511‑bit codeword
    H = C ^ T_prot                           # helper data
    print(f"   Secret key hash: {S_hash[:16]}...")
    print(f"   Helper data H stored.\n")

    # ------------------------------------------------------------------
    # 6.3 Authentication with increasing image distortions
    # ------------------------------------------------------------------
    # We'll save distorted images into a subdirectory
    output_dir = os.path.join(PLOTS_DIR, "casia_distortion_demo")
    os.makedirs(output_dir, exist_ok=True)

    # Test various levels: number of cuts (or dots). We'll use cuts for clarity.
    # You can change the function to add_dots if you prefer.
    test_levels = [0, 5, 10, 15, 20, 25, 30, 35, 40]   # number of cuts
    print(f"2. Authentication tests (BCH capacity t = {BCH_T})")
    print("-" * 60)

    # Load original grayscale image (not preprocessed) to apply distortions
    orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if orig_img is None:
        raise RuntimeError(f"Could not read image {img_path}")

    for num_cuts in test_levels:
        # Apply distortion
        dist_img = add_cuts(orig_img, num_cuts, line_thickness=2)
        # Save temporary file for preprocessing
        temp_path = os.path.join(output_dir, f"distorted_{num_cuts:02d}_cuts.tif")
        cv2.imwrite(temp_path, dist_img)

        # Get protected template from the distorted image
        T_prot_prime = get_protected_template(temp_path, user_key, backbone)

        # Fuzzy commitment recovery
        C_prime = H ^ T_prot_prime
        S_prime, nerr, success = decode_secret(C_prime)

        # Check hash
        if success:
            valid = (hash_secret(S_prime) == S_hash)
        else:
            valid = False

        print(f"  Cuts added: {num_cuts:2d}  ->  Authentication {'✔ SUCCESS' if valid else '✘ FAILED'} ", end="")
        if success and nerr is not None:
            print(f"(corrected errors: {nerr})")
        else:
            print()

        # We do not clean up the temp file so you can inspect it later.

    print("\nDistorted images saved in:", output_dir)
    print("Done. Increase 'num_cuts' beyond ~30 to see failure due to exceeding BCH error correction.")

if __name__ == "__main__":
    main()