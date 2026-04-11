import numpy as np
import os
import argparse
import csv
from scipy.special import comb
import math
from src.config import SCORES_DIR, PLOTS_DIR

# Config (Match this to your embeddings)
EMBEDDING_DIM = 512
RESULTS_DIR = PLOTS_DIR

class CryptographicVaultBenchmark:
    def __init__(self, t_capacity):
        """
        t_capacity: The maximum number of bit errors the BCH code can correct.
        """
        self.dim = EMBEDDING_DIM
        self.t = t_capacity

    def calculate_bit_errors(self, similarity_array):
        """
        Converts Hamming Similarity (0.0 to 1.0) back to actual Bit Errors (0 to 512).
        Math: Bit_Errors = (1.0 - Similarity) * 512
        """
        # Rounding handles tiny floating-point inaccuracies
        bit_errors = np.round((1.0 - similarity_array) * self.dim).astype(int)
        return bit_errors

    def evaluate_vault(self, genuine_sims, impostor_sims, stolen_key_sims):
        """
        Evaluates the Fuzzy Commitment (Secure Sketch) Scheme.
        """
        gen_errors = self.calculate_bit_errors(genuine_sims)
        imp_errors = self.calculate_bit_errors(impostor_sims)
        stolen_errors = self.calculate_bit_errors(stolen_key_sims)

        # 1. Genuine Accept Rate (GAR) / Key Recovery Rate
        successful_unlocks = np.sum(gen_errors <= self.t)
        gar = successful_unlocks / len(gen_errors) if len(gen_errors) > 0 else 0.0
        
        # 2. False Accept Rate (FAR) - Normal Impostor
        false_unlocks = np.sum(imp_errors <= self.t)
        far = false_unlocks / len(imp_errors) if len(imp_errors) > 0 else 0.0
        
        # 3. False Accept Rate (FAR) - Stolen Key Hacker
        stolen_unlocks = np.sum(stolen_errors <= self.t)
        stolen_far = stolen_unlocks / len(stolen_errors) if len(stolen_errors) > 0 else 0.0

        return gar, far, stolen_far

    def calculate_security_level(self):
        """
        Calculates the Information Theoretic Security of the Vault in bits.
        Formula: Security = N - log2(Sphere_Volume(N, t))
        """
        sphere_volume = sum([comb(self.dim, i, exact=True) for i in range(self.t + 1)])
        security_loss = math.log2(sphere_volume)
        actual_security = self.dim - security_loss
        return actual_security

def main(dataset_name):
    print(f"\n{'='*55}")
    print(f" Cryptographic Vault Benchmark: {dataset_name.upper()} ")
    print(f"{'='*55}")
    
    try:
        # Load the scores saved by gen_embeddings.py
        prot_gen = np.load(os.path.join(SCORES_DIR, f"{dataset_name}_prot_gen.npy"))
        prot_imp = np.load(os.path.join(SCORES_DIR, f"{dataset_name}_prot_imp.npy"))
        stolen_imp = np.load(os.path.join(SCORES_DIR, f"{dataset_name}_stolen_imp.npy"))
    except FileNotFoundError:
        print(f"Error: Scores for {dataset_name} not found in {SCORES_DIR}.")
        print(f"Please run 'python -m src.fingerprint.gen_embeddings --dataset {dataset_name}' first.")
        return

    print(f"Loaded Pairs -> Genuine: {len(prot_gen):,}, Impostor: {len(prot_imp):,}, Stolen: {len(stolen_imp):,}\n")

    # --- Setup CSV Logging ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_file_path = os.path.join(RESULTS_DIR, "vault_benchmarks.csv")
    
    # Check if we need to write the header row (if file is new)
    write_header = not os.path.exists(csv_file_path)
    
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Dataset", "BCH Capacity (t)", "GAR (%)", "Zero-Effort FAR (%)", "Stolen Token FAR (%)", "Security Level (Bits)"])

        # Test different BCH error correcting capacities
        # Test higher capacity BCH error correcting boundaries
        for test_t in [60, 80, 100, 110, 120, 130, 140, 150]:
            vault = CryptographicVaultBenchmark(t_capacity=test_t)
            gar, far, stolen_far = vault.evaluate_vault(prot_gen, prot_imp, stolen_imp)
            sec_level = vault.calculate_security_level()
            
            # Print to Terminal
            print(f"--- BCH Capacity: t = {test_t} bits ---")
            print(f"GAR (Key Recovery): {gar:.4%}")
            print(f"Zero-Effort FAR:    {far:.4%}")
            print(f"Stolen Token FAR:   {stolen_far:.4%}")
            print(f"Crypto Security:    {sec_level:.1f} bits (AES-{int(sec_level)} equivalent)\n")
            
            # Write row to CSV
            writer.writerow([
                dataset_name.upper(),
                test_t,
                f"{gar*100:.4f}",
                f"{far*100:.4f}",
                f"{stolen_far*100:.4f}",
                f"{sec_level:.1f}"
            ])

    print(f"Successfully saved results to {csv_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate L2FE-Hash / Secure Sketch Vault")
    
    # We now support all 4 datasets via command line
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["casia", "fvc2000", "fvc2004", "cmbd"],
                        help="The dataset to evaluate in the Cryptographic Vault")
    
    args = parser.parse_args()
    main(args.dataset)