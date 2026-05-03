#!/usr/bin/env python3
"""
Demo: Enrollment & Authentication with Cancelable Biometrics + Custom C-BCH Vault
- Uses a real CASIA fingerprint image.
- Uses our custom compiled C-library to handle m=12, t=150.
- Uses SHAKE256 padding to match 512-bit templates to large codewords.
- Saves corrupted query images to disk for visual verification.
"""

import os
import sys
import hashlib
import numpy as np
import cv2
import torch
import ctypes

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    CASIA_DIR,
    SAVED_MODELS_DIR,
    EMBEDDING_DIM
)
from src.fingerprint.preprocess_fingerprint import preprocess_fingerprint
from src.fingerprint.train import FingerprintEmbeddingNet

# ---------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------
DATASET_TYPE = "casia"
ENROLL_IMG = os.path.join(CASIA_DIR, "000", "L", "000_L0_0.bmp")   # First image
AUTH_IMG   = os.path.join(CASIA_DIR, "000", "L", "000_L0_1.bmp")   # Second image (same finger)
USER_KEY   = "AliceSecureToken123"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# Massive Error Correction Capacity
BCH_M      = 12                    # Galois Field size (up to 4095 bits)
BCH_T      = 150                   # Correct up to 150 bit errors!

ERROR_TYPE = "cuts"                # "cuts", "dots", or "mixed"
MAX_ERROR  = 50                    # maximum severity to test
STEP_ERROR = 5                     # increment step

# ---------------------------------------------------------------------
# 2. CUSTOM C-EXTENSION WRAPPER
# ---------------------------------------------------------------------
class CustomBCH:
    def __init__(self, m, t):
        self.m = m
        self.t = t
        bits = m * t
        self.ecc_bytes = (bits // 8) + 1 if bits % 8 != 0 else bits // 8
        
        # Load the compiled shared library safely from src/utils/Test/
        lib_ext = '.dll' if os.name == 'nt' else '.so'
        lib_prefix = '' if os.name == 'nt' else 'lib'
        lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{lib_prefix}libcustombch{lib_ext}')
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Missing C Library at: {lib_path}\nPlease ensure it is compiled in src/utils/Test/")
            
        self.lib = ctypes.CDLL(lib_path)
        
        # C-types definitions
        self.lib.bch_init_custom.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.bch_init_custom.restype = ctypes.c_void_p
        self.lib.bch_encode_custom.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint, ctypes.POINTER(ctypes.c_uint8)]
        self.lib.bch_decode_custom.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint, ctypes.POINTER(ctypes.c_uint8)]
        self.lib.bch_decode_custom.restype = ctypes.c_int
        self.lib.bch_free_custom.argtypes = [ctypes.c_void_p]
        
        self.ctx = self.lib.bch_init_custom(m, t)
        if not self.ctx:
            raise RuntimeError("Failed to initialize C BCH context.")

    def encode(self, data: bytes):
        data_len = len(data)
        c_data = (ctypes.c_uint8 * data_len)(*data)
        c_ecc = (ctypes.c_uint8 * self.ecc_bytes)()
        self.lib.bch_encode_custom(self.ctx, c_data, data_len, c_ecc)
        return bytes(c_ecc)

    def decode(self, data: bytearray, ecc: bytearray):
        data_len = len(data)
        c_data = (ctypes.c_uint8 * data_len)(*data)
        c_ecc = (ctypes.c_uint8 * len(ecc))(*ecc)
        errors = self.lib.bch_decode_custom(self.ctx, c_data, data_len, c_ecc)
        return errors, bytearray(c_data)

    def __del__(self):
        if hasattr(self, 'ctx') and self.ctx:
            self.lib.bch_free_custom(self.ctx)

# ---------------------------------------------------------------------
# 3. PYTORCH MODEL & L2FE-HASH
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

def apply_l2fe_hash(raw_bio: torch.Tensor, user_key: str) -> np.ndarray:
    """Project raw binary template and binarise with median → protected template."""
    seed_val = int(hashlib.sha256(user_key.encode()).hexdigest(), 16) % (2**32)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed_val)
    R_key = torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), generator=gen, device=DEVICE)
    Q, _ = torch.linalg.qr(R_key)
    
    projected = torch.matmul(Q, raw_bio.float())
    t_median = torch.median(projected)
    T_prot = (projected >= t_median).int().cpu().numpy()
    return T_prot

# ---------------------------------------------------------------------
# 4. CRYPTOGRAPHIC PADDING (Crucial for high BCH capacity)
# ---------------------------------------------------------------------
def expand_template(T_prot: np.ndarray, password: str, target_length_bytes: int) -> np.ndarray:
    """Securely expands the 64-byte template to match the Codeword length."""
    T_prot_bytes = np.packbits(T_prot).tobytes()
    pad_length = target_length_bytes - len(T_prot_bytes)
    
    hasher = hashlib.shake_256()
    hasher.update(password.encode() + b"secure_pad")
    secure_pad = hasher.digest(pad_length)
    
    expanded_bytes = T_prot_bytes + secure_pad
    return np.unpackbits(np.frombuffer(expanded_bytes, dtype=np.uint8))

# ---------------------------------------------------------------------
# 5. SYNTHETIC ERRORS
# ---------------------------------------------------------------------
def add_cuts(img: np.ndarray, num_cuts: int, thickness: int = 2) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(num_cuts):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        cv2.line(out, (x1, y1), (x2, y2), color=0, thickness=thickness)
    return out

def add_dots(img: np.ndarray, num_dots: int, radius: int = 3) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(num_dots):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(out, (x, y), radius, color=0, thickness=-1)
    return out

# ---------------------------------------------------------------------
# 6. ENROLLMENT & AUTHENTICATION PROTOCOLS
# ---------------------------------------------------------------------
def enroll(img_path: str, user_key: str, bch: CustomBCH, model):
    img_tensor = preprocess_fingerprint(img_path, train=False, dataset_type=DATASET_TYPE).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        raw_bio, _ = model(img_tensor)
    raw_bio = raw_bio.squeeze(0)

    # Protected 512-bit template
    T_prot = apply_l2fe_hash(raw_bio, user_key)
    
    # Generate random 32-byte (256-bit) secret key
    secret_key_S = os.urandom(32)
    ecc_bytes = bch.encode(secret_key_S)
    
    # Codeword = Key + Parity
    codeword_C_bytes = secret_key_S + ecc_bytes
    codeword_length = len(codeword_C_bytes)
    
    # Align template size and XOR
    T_padded_bits = expand_template(T_prot, user_key, codeword_length)
    codeword_C_bits = np.unpackbits(np.frombuffer(codeword_C_bytes, dtype=np.uint8))
    
    H_bits = np.bitwise_xor(codeword_C_bits, T_padded_bits)
    hashed_S = hashlib.sha256(secret_key_S).hexdigest()
    
    return H_bits, hashed_S, codeword_length

def authenticate(img_path: str, user_key: str, bch: CustomBCH, H_bits, stored_hashed_S, codeword_length, model):
    img_tensor = preprocess_fingerprint(img_path, train=False, dataset_type=DATASET_TYPE).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        raw_bio, _ = model(img_tensor)
    raw_bio = raw_bio.squeeze(0)

    T_prot_prime = apply_l2fe_hash(raw_bio, user_key)
    T_padded_prime_bits = expand_template(T_prot_prime, user_key, codeword_length)
    
    # Retrieve Candidate Codeword
    C_prime_bits = np.bitwise_xor(H_bits, T_padded_prime_bits)
    C_prime_bytes = np.packbits(C_prime_bits).tobytes()
    
    data_prime = bytearray(C_prime_bytes[:32])
    ecc_prime = bytearray(C_prime_bytes[32:])
    
    # Decode using Custom C-Library
    errors, corrected_data = bch.decode(data_prime, ecc_prime)
    
    if errors != -1:
        if hashlib.sha256(corrected_data).hexdigest() == stored_hashed_S:
            return True, errors
            
    return False, errors

# ---------------------------------------------------------------------
# 7. MAIN DEMO
# ---------------------------------------------------------------------
def main():
    print("Loading ResNet-50 Feature Extractor...")
    model = load_model()
    
    print("Initializing Custom C-Library BCH Vault...")
    bch_vault = CustomBCH(BCH_M, BCH_T)
    print(f"BCH parameters: m={BCH_M}, t={BCH_T} (Max error capacity)")

    print("\n=== ENROLLMENT ===")
    if not os.path.exists(ENROLL_IMG):
        print(f"Warning: Ensure {ENROLL_IMG} exists.")
        return

    H_bits, stored_hashed_S, codeword_length = enroll(ENROLL_IMG, USER_KEY, bch_vault, model)
    print(f"Helper data H created. Vault payload size: {codeword_length} bytes.")

    print(f"\n=== AUTHENTICATION (Testing with {ERROR_TYPE}) ===")
    base_auth_img = cv2.imread(AUTH_IMG, cv2.IMREAD_GRAYSCALE)
    if base_auth_img is None:
        raise FileNotFoundError(f"Cannot read {AUTH_IMG}")

    # Create an output folder for saving damaged images
    os.makedirs("outputs", exist_ok=True)

    for severity in range(0, MAX_ERROR + 1, STEP_ERROR):
        corrupted = base_auth_img.copy()
        if ERROR_TYPE == "cuts":
            corrupted = add_cuts(corrupted, num_cuts=severity)
        elif ERROR_TYPE == "dots":
            corrupted = add_dots(corrupted, num_dots=severity*10) # multiply dots so they are visible
        else: 
            corrupted = add_cuts(corrupted, num_cuts=severity//2)
            corrupted = add_dots(corrupted, num_dots=(severity//2)*10)

        # Save corrupted image for review
        save_path = os.path.join("outputs", f"auth_damaged_{ERROR_TYPE}_{severity:03d}.bmp")
        cv2.imwrite(save_path, corrupted)

        success, errors = authenticate(
            save_path, USER_KEY, bch_vault, H_bits, stored_hashed_S, codeword_length, model
        )
        
        status = "SUCCESS" if success else "FAILED"
        err_msg = f"Corrected {errors} bit errors" if success else f"Bit flips exceeded limit ({BCH_T})"
        print(f"Severity {severity:3d} | Status: {status} | {err_msg} | Image: {save_path}")

if __name__ == "__main__":
    main()