import numpy as np
import os
import argparse
import csv
from scipy.special import comb
import math
import ctypes
import hashlib
import random

from src.config import SCORES_DIR, PLOTS_DIR

# Config
EMBEDDING_DIM = 512
RESULTS_DIR = PLOTS_DIR

# =====================================================================
# 1. THE CUSTOM C-EXTENSION WRAPPER
# =====================================================================
class CustomBCH:
    def __init__(self, m, t):
        self.m = m
        self.t = t
        bits = m * t
        self.ecc_bytes = (bits // 8) + 1 if bits % 8 != 0 else bits // 8
        
        # Safely locate the DLL
        lib_ext = '.dll' if os.name == 'nt' else '.so'
        lib_prefix = '' if os.name == 'nt' else 'lib'
        
        # Update this path if your DLL is located somewhere else!
        lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                'utils', 'Test', f'{lib_prefix}custombch{lib_ext}')
        
        if not os.path.exists(lib_path):
            # Fallback path if running directly in the Test folder
            lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{lib_prefix}custombch{lib_ext}')
            if not os.path.exists(lib_path):
                raise FileNotFoundError(f"Missing C Library at: {lib_path}")
            
        self.lib = ctypes.CDLL(lib_path)
        self.lib.bch_init_custom.argtypes = [ctypes.c_int, ctypes.c_int]
        self.lib.bch_init_custom.restype = ctypes.c_void_p
        self.lib.bch_encode_custom.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint, ctypes.POINTER(ctypes.c_uint8)]
        self.lib.bch_decode_custom.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint, ctypes.POINTER(ctypes.c_uint8)]
        self.lib.bch_decode_custom.restype = ctypes.c_int
        self.lib.bch_free_custom.argtypes = [ctypes.c_void_p]
        
        self.ctx = self.lib.bch_init_custom(m, t)

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

# =====================================================================
# 2. CRYPTO HELPER FUNCTIONS
# =====================================================================
def flip_exact_bits(bit_array, num_flips):
    """Physically flips an exact number of bits to simulate dataset errors."""
    corrupted = bit_array.copy()
    flip_indices = random.sample(range(len(corrupted)), num_flips)
    for idx in flip_indices:
        corrupted[idx] = 1 - corrupted[idx]
    return corrupted

def secure_pad_template(T_prot_bytes, target_length_bytes):
    """Expands template to match codeword length securely."""
    pad_length = target_length_bytes - len(T_prot_bytes)
    hasher = hashlib.shake_256()
    hasher.update(b"static_eval_seed") 
    secure_pad = hasher.digest(pad_length)
    return T_prot_bytes + secure_pad

# =====================================================================
# 3. THE EVALUATION ENGINE
# =====================================================================
class CryptographicVaultBenchmark:
    def __init__(self, t_capacity):
        self.dim = EMBEDDING_DIM
        self.t = t_capacity
        self.bch = CustomBCH(m=12, t=t_capacity)

    def calculate_bit_errors(self, similarity_array):
        # Rounding handles tiny floating-point inaccuracies
        return np.round((1.0 - similarity_array) * self.dim).astype(int)

    def _physically_verify_decoder(self, errors_array):
        """
        Instead of simulating (errors <= t), this extracts the unique errors 
        from the dataset and forces the C-DLL to actually attempt to decode them.
        """
        unique_errs, counts = np.unique(errors_array, return_counts=True)
        total_successes = 0
        
        # 1. Setup a Master Vault
        T_prot = np.random.randint(0, 2, size=512, dtype=np.uint8)
        T_prot_bytes = np.packbits(T_prot).tobytes()
        secret_key_S = os.urandom(32)
        ecc_bytes = self.bch.encode(secret_key_S)
        codeword_C_bytes = secret_key_S + ecc_bytes
        
        T_padded_bytes = secure_pad_template(T_prot_bytes, len(codeword_C_bytes))
        T_padded_bits = np.unpackbits(np.frombuffer(T_padded_bytes, dtype=np.uint8))
        codeword_C_bits = np.unpackbits(np.frombuffer(codeword_C_bytes, dtype=np.uint8))
        H_bits = np.bitwise_xor(codeword_C_bits, T_padded_bits)

        # 2. Test the C-Decoder against the dataset's unique error counts
        for err, count in zip(unique_errs, counts):
            if err > 250:
                # Optimization: Hardware limit. Anything above 250 errors will mathematically fail.
                continue
                
            # Physically corrupt the template
            T_prot_corrupted = flip_exact_bits(T_prot, err)
            T_prot_corrupted_bytes = np.packbits(T_prot_corrupted).tobytes()
            T_padded_prime_bytes = secure_pad_template(T_prot_corrupted_bytes, len(codeword_C_bytes))
            T_padded_prime_bits = np.unpackbits(np.frombuffer(T_padded_prime_bytes, dtype=np.uint8))
            
            # XOR to retrieve candidate codeword
            C_prime_bits = np.bitwise_xor(H_bits, T_padded_prime_bits)
            C_prime_bytes = np.packbits(C_prime_bits).tobytes()
            
            data_prime = bytearray(C_prime_bytes[:32])
            ecc_prime = bytearray(C_prime_bytes[32:])
            
            # FIRE THE C-LIBRARY
            errors_fixed, corrected_data = self.bch.decode(data_prime, ecc_prime)
            
            if errors_fixed != -1 and bytes(corrected_data) == secret_key_S:
                # The C-library successfully fixed this dataset error!
                total_successes += count
                
        return total_successes

    def evaluate_vault(self, genuine_sims, impostor_sims, stolen_key_sims):
        gen_errors = self.calculate_bit_errors(genuine_sims)
        imp_errors = self.calculate_bit_errors(impostor_sims)
        stolen_errors = self.calculate_bit_errors(stolen_key_sims)

        # Physically pass the dataset to the compiled C-library
        successful_unlocks = self._physically_verify_decoder(gen_errors)
        gar = successful_unlocks / len(gen_errors) if len(gen_errors) > 0 else 0.0
        
        false_unlocks = self._physically_verify_decoder(imp_errors)
        far = false_unlocks / len(imp_errors) if len(imp_errors) > 0 else 0.0
        
        stolen_unlocks = self._physically_verify_decoder(stolen_errors)
        stolen_far = stolen_unlocks / len(stolen_errors) if len(stolen_errors) > 0 else 0.0

        return gar, far, stolen_far

    def calculate_security_level(self):
        sphere_volume = sum([comb(self.dim, i, exact=True) for i in range(self.t + 1)])
        security_loss = math.log2(sphere_volume)
        actual_security = self.dim - security_loss
        return actual_security

# =====================================================================
# 4. MAIN LOOP
# =====================================================================
def main(dataset_name):
    print(f"\n{'='*55}")
    print(f" C-Library Verified Vault Benchmark: {dataset_name.upper()} ")
    print(f"{'='*55}")
    
    try:
        prot_gen = np.load(os.path.join(SCORES_DIR, f"{dataset_name}_prot_gen.npy"))
        prot_imp = np.load(os.path.join(SCORES_DIR, f"{dataset_name}_prot_imp.npy"))
        stolen_imp = np.load(os.path.join(SCORES_DIR, f"{dataset_name}_stolen_imp.npy"))
    except FileNotFoundError:
        print(f"Error: Scores for {dataset_name} not found.")
        return

    print(f"Loaded Pairs -> Genuine: {len(prot_gen):,}, Impostor: {len(prot_imp):,}, Stolen: {len(stolen_imp):,}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_file_path = os.path.join(RESULTS_DIR, "vault_benchmarks_verified.csv")
    
    write_header = not os.path.exists(csv_file_path)
    
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Dataset", "BCH Capacity (t)", "GAR (%)", "Zero-Effort FAR (%)", "Stolen Token FAR (%)", "Security Level (Bits)"])

        # Test higher capacity BCH error correcting boundaries
        for test_t in [60, 80, 100, 110, 120, 130, 140, 150]:
            vault = CryptographicVaultBenchmark(t_capacity=test_t)
            
            gar, far, stolen_far = vault.evaluate_vault(prot_gen, prot_imp, stolen_imp)
            sec_level = vault.calculate_security_level()
            
            print(f"--- C-Library Decoding: t = {test_t} bits ---")
            print(f"GAR (Key Recovery): {gar:.4%}")
            print(f"Zero-Effort FAR:    {far:.4%}")
            print(f"Stolen Token FAR:   {stolen_far:.4%}")
            print(f"Crypto Security:    {sec_level:.1f} bits (AES-{int(sec_level)} equivalent)\n")
            
            writer.writerow([
                dataset_name.upper(),
                test_t,
                f"{gar*100:.4f}",
                f"{far*100:.4f}",
                f"{stolen_far*100:.4f}",
                f"{sec_level:.1f}"
            ])

    print(f"Successfully saved physically verified results to {csv_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Vault with actual C-Library")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["casia", "fvc2000", "fvc2004", "cmbd"],
                        help="The dataset to evaluate in the Cryptographic Vault")
    args = parser.parse_args()
    main(args.dataset)