import ctypes
import os
import cv2
import numpy as np
import torch
import hashlib
from torchvision import transforms

# --- IMPORT YOUR EXISTING PIPELINE ---
from src.fingerprint.preprocess_fingerprint import preprocess_fingerprint
from src.fingerprint.train import FingerprintEmbeddingNet  
from src.config import SAVED_MODELS_DIR, EMBEDDING_DIM

# --- 1. CUSTOM BCH C-EXTENSION WRAPPER ---
class CustomBCH:
    def __init__(self, m, t):
        self.m = m
        self.t = t
        # Calculate exactly how many parity bytes the C library will generate
        bits = m * t
        self.ecc_bytes = (bits // 8) + 1 if bits % 8 != 0 else bits // 8
        
        # Load the compiled shared library
        lib_ext = '.dll' if os.name == 'nt' else '.so'
        lib_prefix = '' if os.name == 'nt' else 'lib'
        lib_path = os.path.join(os.path.dirname(__file__), f'{lib_prefix}custombch{lib_ext}')
        self.lib = ctypes.CDLL(lib_path)
        
        # Define argument types for memory safety
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

# --- CONFIGURATION ---
BCH_M = 12       # Galois Field size (Handles up to 4095 bits)
BCH_T = 150      # Our aggressive error correction capacity!
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bch = CustomBCH(BCH_M, BCH_T)

# --- 2. CRYPTOGRAPHIC ALIGNMENT PADDING ---
def expand_template(T_prot, password, target_length_bytes):
    """Securely expands the 64-byte template to match the Codeword length."""
    T_prot_bytes = np.packbits(T_prot).tobytes()
    pad_length = target_length_bytes - len(T_prot_bytes)
    
    # Use SHAKE256 to generate a deterministic, cryptographically secure pad
    hasher = hashlib.shake_256()
    hasher.update(password.encode() + b"secure_pad")
    secure_pad = hasher.digest(pad_length)
    
    expanded_bytes = T_prot_bytes + secure_pad
    return np.unpackbits(np.frombuffer(expanded_bytes, dtype=np.uint8))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

# --- 1. PREPROCESSING & ERROR INJECTION ---


def add_fingerprint_damage(image_path, error_level):
    """
    Loads an image and adds simulated cuts (lines) and dots (noise)
    based on the error_level multiplier.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    damaged_img = img.copy()
    h, w = damaged_img.shape
    
    # Add random black cuts across the fingerprint
    num_cuts = int(error_level * 2)
    for _ in range(num_cuts):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        thickness = np.random.randint(1, 4)
        cv2.line(damaged_img, (x1, y1), (x2, y2), (0, 0, 0), thickness)
        
    # Add white dots (sensor noise/dust)
    num_dots = int(error_level * 50)
    for _ in range(num_dots):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        cv2.circle(damaged_img, (x, y), 1, (255, 255, 255), -1)
        
    return damaged_img

# --- 2. MODEL & L2FE-HASH (CANCELABILITY) ---

def extract_raw_template(model, preprocessed_tensor):
    """Passes image through ResNet-50 and STE to get 512-bit raw template."""
    model.eval()
    with torch.no_grad():
        output = model(preprocessed_tensor) # Expected shape: (1, 512)
        # Assuming your STE outputs strictly 0.0 and 1.0
        binary_template = output.squeeze().cpu().numpy().astype(int)
    return binary_template

def apply_l2fe_hash(R_bio, password):
    """Projects raw template with user password to create revocable template."""
    # 1. User key -> deterministic seed
    seed_bytes = hashlib.sha256(password.encode()).digest()
    seed_int = int.from_bytes(seed_bytes[:4], 'big') # Simplify seed for numpy
    
    # 2. Generate random matrix (Simulating the chaotic orthogonal matrix R_key)
    np.random.seed(seed_int)
    random_matrix = np.random.randn(512, 512)
    Q, _ = np.linalg.qr(random_matrix) # QR decomposition to make it orthogonal
    
    # 3. Projection & Binarisation
    P = np.dot(R_bio, Q)
    median_val = np.median(P)
    T_prot = (P >= median_val).astype(int)
    
    return T_prot

# --- 3. VAULT LOGIC (HELPER FUNCTIONS) ---

def bit_array_to_bytes(bit_array):
    """Packs an array of 1s and 0s into bytes."""
    return np.packbits(bit_array).tobytes()

def bytes_to_bit_array(byte_data):
    """Unpacks bytes into an array of 1s and 0s."""
    return np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))

# --- 4. ENROLLMENT & AUTHENTICATION ---
def enroll(model, image_path, password):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    tensor_img = preprocess_fingerprint(img).to(DEVICE)
    R_bio = extract_raw_template(model, tensor_img)
    T_prot = apply_l2fe_hash(R_bio, password)
    
    # Generate 32-byte (256-bit) Secret Key
    secret_key_S = os.urandom(32) 
    ecc_bytes = bch.encode(secret_key_S)
    
    # Codeword is Key + Parity
    codeword_C_bytes = secret_key_S + ecc_bytes
    codeword_length = len(codeword_C_bytes)
    
    # Expand Template to match Codeword length safely
    T_padded_bits = expand_template(T_prot, password, codeword_length)
    
    codeword_C_bits = np.unpackbits(np.frombuffer(codeword_C_bytes, dtype=np.uint8))
    
    # Bind template to key: H = C XOR T_padded
    H_bits = np.bitwise_xor(codeword_C_bits, T_padded_bits)
    hashed_S = hashlib.sha256(secret_key_S).hexdigest()
    
    print(f"Enrollment Success! Key: 32b | ECC: {len(ecc_bytes)}b | Total Vault Size: {codeword_length} bytes")
    return H_bits, hashed_S, codeword_length

def authenticate(model, damaged_img, password, H_bits, stored_hashed_S, codeword_length):
    tensor_img = preprocess_fingerprint(damaged_img).to(DEVICE)
    R_bio_prime = extract_raw_template(model, tensor_img)
    T_prot_prime = apply_l2fe_hash(R_bio_prime, password)
    
    # Expand query template
    T_padded_prime_bits = expand_template(T_prot_prime, password, codeword_length)
    
    # Retrieve Codeword Candidate: C' = H XOR T'_padded
    C_prime_bits = np.bitwise_xor(H_bits, T_padded_prime_bits)
    C_prime_bytes = np.packbits(C_prime_bits).tobytes()
    
    # Split into data and parity
    data_prime = bytearray(C_prime_bytes[:32])
    ecc_prime = bytearray(C_prime_bytes[32:])
    
    # Send to our Custom C Decoder!
    errors, corrected_data = bch.decode(data_prime, ecc_prime)
    
    if errors != -1:
        if hashlib.sha256(corrected_data).hexdigest() == stored_hashed_S:
            return True, errors
            
    return False, -1

# --- 5. MAIN EXECUTION LOOP ---

if __name__ == "__main__":
    # --- LOAD YOUR PYTORCH MODEL HERE ---
    model = load_model()
    
    # Setup Paths
    image_path = "C:/Users/lappy-002/ML/BiometricTemplateGeneration/datasets/CASIA-dataset/000/L/000_L0_0.bmp" # UPDATE WITH REAL PATH
    password = "MySecurePassword123"
    
    # We create a dummy black square just to make the script runnable 
    # without a real image if you are copy/pasting.
    if not os.path.exists(image_path):
        cv2.imwrite(image_path, np.ones((300, 300), dtype=np.uint8) * 127)
    
    # 1. ENROLL
    H_bits, stored_hashed_S = enroll(model, image_path, password)
    
    # 2. TEST AUTHENTICATION UNDER INCREASING ERROR
    print("\n--- AUTHENTICATION TESTING ---")
    print(f"BCH Error Correction Capacity (t): {BCH_T} bits\n")
    
    # Increase error level from 0 to 10
    for error_level in range(0, 11):
        # Generate damaged image
        damaged_img = add_fingerprint_damage(image_path, error_level)
        
        # Save the damaged image so you can visualize the cuts/noise
        output_filename = f"auth_attempt_error_lvl_{error_level}.png"
        cv2.imwrite(output_filename, damaged_img)
        
        # Authenticate
        success, bitflips = authenticate(model, damaged_img, password, H_bits, stored_hashed_S)
        
        status = "SUCCESS" if success else "FAILED "
        flips_text = f"Corrected {bitflips} errors" if success else "Errors exceeded capacity (>t)"
        print(f"Error Level {error_level:2d} | Status: {status} | {flips_text} | Image saved: {output_filename}")