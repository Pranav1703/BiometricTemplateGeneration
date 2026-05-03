import os
import sys
import hashlib
import random
import numpy as np
import cv2
import torch
from bch import BCH

# Add project root to path (if needed)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    CASIA_DIR,
    SAVED_MODELS_DIR,
    EMBEDDING_DIM
)
from src.fingerprint.preprocess_fingerprint import preprocess_fingerprint
from src.fingerprint.train import FingerprintEmbeddingNet  # Your model class

# ---------------------------------------------------------------------
# 1. CONFIGURATION – tweak these values
# ---------------------------------------------------------------------
DATASET_TYPE = "casia"
ENROLL_IMG = os.path.join(CASIA_DIR, "000", "L0", "000_L0_0.bmp")   # first image
AUTH_IMG   = os.path.join(CASIA_DIR, "000", "L0", "000_L0_1.bmp")   # second image (same finger)
USER_KEY   = "AliceSecureToken123"
BCH_T      = 20                    # BCH error‑correcting capacity (max 56 for m=9)
ERROR_TYPE = "cuts"                # "cuts", "dots", or "mixed"
MAX_ERROR  = 30                    # maximum severity to test
STEP_ERROR = 5                     # increment step
DEVICE     = "cuda"
# ---------------------------------------------------------------------
# 2. LOAD THE FROZEN BACKBONE
# ---------------------------------------------------------------------
def load_model():
    model_path = os.path.join(SAVED_MODELS_DIR, "casia_arcface_model_quantized_100_128bs.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    net = FingerprintEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    net.load_state_dict(state["backbone"])
    net.eval()
    return net

# ---------------------------------------------------------------------
# 3. L2FE‑HASH: chaotic orthogonal projection + median threshold
# ---------------------------------------------------------------------
def generate_chaotic_projection(user_key: str) -> torch.Tensor:
    """Deterministic 512×512 orthogonal matrix from user key."""
    seed_val = int(hashlib.sha256(user_key.encode()).hexdigest(), 16) % (2**32)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed_val)
    R = torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), generator=gen, device=DEVICE)
    Q, _ = torch.linalg.qr(R)
    return Q

def extract_protected_template(raw_bio: torch.Tensor, R_key: torch.Tensor) -> torch.Tensor:
    """Project raw binary template and binarise with median → protected template."""
    projected = torch.matmul(R_key, raw_bio)
    t = torch.median(projected)
    return (projected >= t).float()

# ---------------------------------------------------------------------
# 4. ADD SYNTHETIC ERRORS (cuts/dots) to the raw image
# ---------------------------------------------------------------------
def add_cuts(img: np.ndarray, num_cuts: int, thickness: int = 2) -> np.ndarray:
    """Draw random black lines on a grayscale image."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(num_cuts):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        cv2.line(out, (x1, y1), (x2, y2), color=0, thickness=thickness)
    return out

def add_dots(img: np.ndarray, num_dots: int, radius: int = 3) -> np.ndarray:
    """Draw random black dots."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(num_dots):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(out, (x, y), radius, color=0, thickness=-1)
    return out

# ---------------------------------------------------------------------
# 5. BCH FUZZY COMMITMENT (m=9, n=511)
# ---------------------------------------------------------------------
def setup_bch(t: int):
    """
    Create a BCH(m=9, t) object.
    m = 9  → codeword length n = 2^9 - 1 = 511 bits.
    Returns (bch_instance, data_bytes, code_bytes).
    """
    # For m=9, t ≤ 56.  Primitive polynomial is auto‑chosen.
    bch = BCH(9, t)                 # all parameters in one call!

    # Determine the size of the secret key (in bytes) that can be protected
    k_bits = bch.n - 9 * t          # number of data bits
    data_bytes = (k_bits + 7) // 8

    # Codeword length in bytes (ceil(511/8) = 64)
    code_bytes = (bch.n + 7) // 8
    return bch, int(data_bytes), int(code_bytes)

def bits2bytes(bits: np.ndarray) -> bytes:
    """Convert a 1D binary numpy array (0/1) to bytes, MSB first, padded to byte boundary."""
    padded = np.pad(bits, (0, (8 - len(bits) % 8) % 8), constant_values=0)
    return bytes(int("".join(str(b) for b in padded[i:i+8]), 2)
                 for i in range(0, len(padded), 8))

def bytes2bits(data: bytes, nbits: int) -> np.ndarray:
    """Convert bytes back to a binary array of exactly nbits."""
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    return bits[:nbits]

# ---------------------------------------------------------------------
# 6. ENROLLMENT
# ---------------------------------------------------------------------
def enroll(img_path: str, user_key: str, bch, data_bytes, model):
    """
    Enroll a user.
    Returns: helper_data (H) as 511‑bit array, secret_key_bytes, R_key.
    """
    # Preprocess image
    img_tensor = preprocess_fingerprint(img_path, train=False, dataset_type=DATASET_TYPE)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    # Raw binary template R_bio (512 bits)
    with torch.no_grad():
        raw_bio, _ = model(img_tensor)  # quantized bits from STE
    raw_bio = raw_bio.squeeze(0)  # [512]

    # L2FE‑Hash → protected template T_prot (512 bits, then keep first 511)
    R_key = generate_chaotic_projection(user_key)
    T_prot = extract_protected_template(raw_bio, R_key).cpu().numpy().astype(int)
    T_prot_511 = T_prot[:bch.n].astype(int)    # list or array of 0/1

    # Generate random secret key S
    secret = os.urandom(data_bytes)

    # BCH encode: returns the full codeword as bytes (data + ecc)
    codeword_bytes = bch.encode(secret)         # <-- NEW API
    codeword_bits = bytes2bits(codeword_bytes, bch.n)

    # Helper data H = C XOR T_prot
    H = codeword_bits ^ T_prot_511

    return H, secret, R_key

# ---------------------------------------------------------------------
# 7. AUTHENTICATION (with damaged query image)
# ---------------------------------------------------------------------
def authenticate(img_path: str, user_key: str, bch, data_bytes, H, secret, R_key, model):
    """
    Attempt authentication with a (possibly corrupted) fingerprint.
    Returns True if secret key recovered, False otherwise.
    """
    # Preprocess query image
    img_tensor = preprocess_fingerprint(img_path, train=False, dataset_type=DATASET_TYPE)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

    # Extract raw binary template
    with torch.no_grad():
        raw_bio, _ = model(img_tensor)
    raw_bio = raw_bio.squeeze(0)

    # Generate protected template T'_prot (using same R_key)
    T_prot_prime = extract_protected_template(raw_bio, R_key).cpu().numpy().astype(int)
    T_prot_prime_511 = T_prot_prime[:bch.n].astype(int)

    # Recover corrupted codeword
    C_prime_bits = H ^ T_prot_prime_511
    C_prime_bytes = bits2bytes(C_prime_bits)   # still 64 bytes

    # BCH decode – the library corrects errors and returns the original data (S)
    try:
        recovered_secret = bch.decode(C_prime_bytes)
    except Exception:          # decode failure (too many errors)
        return False

    # Compare recovered secret with original
    return recovered_secret == secret

# ---------------------------------------------------------------------
# 8. MAIN DEMO
# ---------------------------------------------------------------------
def main():
    print("Loading model...")
    model = load_model()
    print("Model loaded.\n")

    # Setup BCH
    bch, data_bytes, code_bytes = setup_bch(BCH_T)
    print(f"BCH parameters: m=9, n=511, t={BCH_T}, secret key size = {data_bytes*8} bits\n")

    # Enroll using the original (clean) image
    print("=== ENROLLMENT ===")
    H, secret, R_key = enroll(ENROLL_IMG, USER_KEY, bch, data_bytes, model)
    print(f"Helper data H created. Secret key (hex) = {secret.hex()}\n")

    # Test with increasing corruption levels
    print(f"=== AUTHENTICATION with {ERROR_TYPE} ===")
    # Load the clean auth image as grayscale (to be modified each time)
    base_auth_img = cv2.imread(AUTH_IMG, cv2.IMREAD_GRAYSCALE)
    if base_auth_img is None:
        raise FileNotFoundError(f"Cannot read {AUTH_IMG}")

    for severity in range(0, MAX_ERROR+1, STEP_ERROR):
        corrupted = base_auth_img.copy()
        if ERROR_TYPE == "cuts":
            corrupted = add_cuts(corrupted, num_cuts=severity)
        elif ERROR_TYPE == "dots":
            corrupted = add_dots(corrupted, num_dots=severity)
        else:  # mixed
            corrupted = add_cuts(corrupted, num_cuts=severity//2)
            corrupted = add_dots(corrupted, num_dots=severity//2)

        # Save temporary corrupted image
        tmp_path = f"temp_corrupted_{severity}.bmp"
        cv2.imwrite(tmp_path, corrupted)

        success = authenticate(
            tmp_path, USER_KEY, bch, data_bytes, H, secret, R_key, model
        )
        print(f"Severity {severity:3d}  →  Authentication {'SUCCESS' if success else 'FAILED'}")
        os.remove(tmp_path)

if __name__ == "__main__":
    main()