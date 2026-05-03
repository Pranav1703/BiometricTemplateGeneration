import numpy as np
import os
import argparse
import ctypes
import hashlib
import random
import time
import csv
from tqdm import tqdm

from src.config import SCORES_DIR, PLOTS_DIR, CUS_BCH_DLL
from src.config import EMBEDDING_DIM

RESULTS_DIR = PLOTS_DIR

# =====================================================================
# 1. CUSTOM C-EXTENSION WRAPPER
# =====================================================================
class CustomBCH:
    def __init__(self, m, t):
        self.m = m
        self.t = t
        bits = m * t
        self.ecc_bytes = (bits // 8) + 1 if bits % 8 != 0 else bits // 8
        
        lib_ext = '.dll' if os.name == 'nt' else '.so'
        lib_prefix = '' if os.name == 'nt' else 'lib'
        
        # Safely locate the DLL
        lib_path = CUS_BCH_DLL
        
        if not os.path.exists(lib_path):
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
# 2. HELPER FUNCTIONS
# =====================================================================
def flip_exact_bits(bit_array, num_flips):
    corrupted = bit_array.copy()
    flip_indices = random.sample(range(len(corrupted)), num_flips)
    for idx in flip_indices:
        corrupted[idx] = 1 - corrupted[idx]
    return corrupted

def secure_pad_template(T_prot_bytes, target_length_bytes):
    pad_length = target_length_bytes - len(T_prot_bytes)
    hasher = hashlib.shake_256()
    hasher.update(b"static_unoptimized_seed") 
    secure_pad = hasher.digest(pad_length)
    return T_prot_bytes + secure_pad

# =====================================================================
# 3. UNOPTIMIZED BRUTE-FORCE ENGINE
# =====================================================================
class BruteForceVaultBenchmark:
    def __init__(self, t_capacity):
        self.dim = EMBEDDING_DIM
        self.t = t_capacity
        self.bch = CustomBCH(m=12, t=t_capacity)

    def calculate_bit_errors(self, similarity_array):
        return np.round((1.0 - similarity_array) * self.dim).astype(int)

    def _brute_force_decode(self, errors_array, label):
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

        # 2. Iterate through EVERY SINGLE PAIR
        for err in tqdm(errors_array, desc=f"Decoding {label} Pairs (t={self.t})", unit="pair"):
            if err > 250:
                continue
                
            T_prot_corrupted = flip_exact_bits(T_prot, err)
            T_prot_corrupted_bytes = np.packbits(T_prot_corrupted).tobytes()
            T_padded_prime_bytes = secure_pad_template(T_prot_corrupted_bytes, len(codeword_C_bytes))
            T_padded_prime_bits = np.unpackbits(np.frombuffer(T_padded_prime_bytes, dtype=np.uint8))
            
            C_prime_bits = np.bitwise_xor(H_bits, T_padded_prime_bits)
            C_prime_bytes = np.packbits(C_prime_bits).tobytes()
            
            data_prime = bytearray(C_prime_bytes[:32])
            ecc_prime = bytearray(C_prime_bytes[32:])
            
            errors_fixed, corrected_data = self.bch.decode(data_prime, ecc_prime)
            
            if errors_fixed != -1 and bytes(corrected_data) == secret_key_S:
                total_successes += 1
                
        return total_successes

# =====================================================================
# 4. MAIN SCRIPT
# =====================================================================
def main(dataset_name, t_input):
    print(f"\n{'='*65}")
    print(f" UNOPTIMIZED BRUTE-FORCE VERIFICATION: {dataset_name.upper()} ")
    print(f"{'='*65}")
    
    # Parse the t_input argument
    if t_input.lower() == "all":
        t_values = [60, 80, 100, 110, 120, 130, 140, 150]
    else:
        try:
            t_values = [int(x.strip()) for x in t_input.split(',')]
        except ValueError:
            print("Error: Invalid --t argument. Use 'all', a single number, or a comma-separated list.")
            return

    try:
        prot_gen = np.load(os.path.join(SCORES_DIR, f"{dataset_name}_prot_gen.npy"))
        prot_imp = np.load(os.path.join(SCORES_DIR, f"{dataset_name}_prot_imp.npy"))
    except FileNotFoundError:
        print(f"Error: Scores for {dataset_name} not found.")
        return

    print(f"Loaded Pairs -> Genuine: {len(prot_gen):,}, Impostor: {len(prot_imp):,}")
    print(f"Capacities to test: {t_values}\n")

    # CSV Setup
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_file_path = os.path.join(RESULTS_DIR, "vault_benchmarks_unoptimized.csv")
    write_header = not os.path.exists(csv_file_path)

    # Initialize a dummy vault just to access the calculate_bit_errors method
    dummy_vault = BruteForceVaultBenchmark(t_capacity=150)
    gen_errors = dummy_vault.calculate_bit_errors(prot_gen)
    imp_errors = dummy_vault.calculate_bit_errors(prot_imp)

    # Open CSV in append mode so we can write row-by-row
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Dataset", "BCH Capacity (t)", "GAR (%)", "Zero-Effort FAR (%)", "Time Taken (s)"])

        # Loop over every requested capacity
        for target_t in t_values:
            print(f"\n--- Starting Evaluation for t = {target_t} ---")
            vault = BruteForceVaultBenchmark(t_capacity=target_t)
            
            start_time = time.time()

            # Process Data
            successful_gen = vault._brute_force_decode(gen_errors, "Genuine")
            gar = successful_gen / len(gen_errors) if len(gen_errors) > 0 else 0.0
            
            successful_imp = vault._brute_force_decode(imp_errors, "Impostor")
            far = successful_imp / len(imp_errors) if len(imp_errors) > 0 else 0.0

            end_time = time.time()
            elapsed_time = end_time - start_time
            
            print("\nRESULTS:")
            print(f"  GAR             : {gar:.4%}")
            print(f"  Zero-Effort FAR : {far:.4%}")
            print(f"  Time Taken      : {elapsed_time:.2f} seconds")

            # Write immediately to CSV so we don't lose data if we cancel the script midway
            writer.writerow([
                dataset_name.upper(),
                target_t,
                f"{gar*100:.4f}",
                f"{far*100:.4f}",
                f"{elapsed_time:.2f}"
            ])
            
            # Flush the file buffer to guarantee it's written to disk immediately
            file.flush()

    print(f"\nSuccessfully saved brute-force results to {csv_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brute Force Unoptimized Vault Test")
    parser.add_argument("--dataset", type=str, required=True, choices=["casia", "fvc2000", "fvc2004", "cmbd"])
    parser.add_argument("--t", type=str, default="150", help="Use 'all' to run standard range, a single number (e.g., '150'), or comma-separated ('60,100,150')")
    
    args = parser.parse_args()
    main(args.dataset, args.t)