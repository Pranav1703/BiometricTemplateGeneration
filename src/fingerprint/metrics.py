import numpy as np
import os
from scipy.stats import ks_2samp, norm

# ---------------------------
# Configuration
# ---------------------------
SCORE_DIR = "artifacts/scores"
FAR_TARGETS = [0.01, 0.001, 0.0001] # 1%, 0.1%, and 0.01% FAR

# ---------------------------
# d-prime (d')
# ---------------------------
def calculate_d_prime(genuine_scores, impostor_scores):
    """
    Calculates d-prime, a measure of separability between two distributions.
    A higher d-prime value indicates better separation.
    """
    mu_genuine = np.mean(genuine_scores)
    mu_impostor = np.mean(impostor_scores)
    std_genuine = np.std(genuine_scores)
    std_impostor = np.std(impostor_scores)
    
    # Calculate pooled standard deviation
    std_pooled = np.sqrt(0.5 * (std_genuine**2 + std_impostor**2))
    
    # Calculate d-prime
    d_prime = (mu_genuine - mu_impostor) / std_pooled
    return d_prime

# ---------------------------
# KS Test (Kolmogorov-Smirnov)
# ---------------------------
def calculate_ks_test(genuine_scores, impostor_scores):
    """
    Performs the 2-sample KS test.
    Returns the statistic (D), which is the max distance between the CDFs (0-1).
    A value closer to 1 means the distributions are more different.
    """
    # Note: ks_2samp can be slow on very large datasets. 
    # We can downsample the impostor scores if needed, but we'll try full first.
    if len(impostor_scores) > 500000:
        # Downsample to speed up KS test if we have millions of scores
        downsampled_impostors = np.random.choice(impostor_scores, 500000, replace=False)
    else:
        downsampled_impostors = impostor_scores
        
    ks_result = ks_2samp(genuine_scores, downsampled_impostors)
    return ks_result.statistic, ks_result.pvalue

# ---------------------------
# GAR (Genuine Acceptance Rate) at specific FAR levels
# ---------------------------
def calculate_gar_at_far(genuine_scores, impostor_scores, far_levels):
    """
    Calculates the Genuine Acceptance Rate (GAR = 1 - FRR) at given FAR levels.
    """
    results = {}
    
    # Sort impostor scores once to efficiently find thresholds
    impostor_scores_sorted = np.sort(impostor_scores)
    num_impostors = len(impostor_scores_sorted)

    for far in far_levels:
        # Find the index corresponding to the FAR percentile
        # e.g., for FAR=0.01 (1%), we need the 99th percentile threshold
        index = int((1.0 - far) * num_impostors)
        
        # Ensure index is within bounds
        if index >= num_impostors:
            index = num_impostors - 1
            
        threshold = impostor_scores_sorted[index]
        
        # Calculate GAR (1 - FRR) at this threshold
        gar = np.mean(genuine_scores >= threshold)
        
        results[far] = {
            'GAR': gar,
            'Threshold': threshold
        }
        
    return results

# ---------------------------
# Main execution
# ---------------------------
def analyze_system(system_name, genuine_path, impostor_path):
    """
    Loads scores and runs all advanced metric calculations.
    """
    print(f"\n--- Analyzing Advanced Metrics for: {system_name} System ---")
    try:
        genuine = np.load(genuine_path)
        impostor = np.load(impostor_path)
    except FileNotFoundError as e:
        print(f"Error loading score files: {e}. Skipping.")
        return

    # 1. d-prime
    d_prime = calculate_d_prime(genuine, impostor)
    print(f"  [1] d-prime (Separability): {d_prime:.4f}")
    print(f"      (Measures distribution separation. Higher is better.)")

    # 2. KS Test
    ks_stat, p_value = calculate_ks_test(genuine, impostor)
    print(f"  [2] KS Statistic (D):       {ks_stat:.4f}")
    print(f"      (Max difference between distributions, 0-1. Higher is more different.)")
    print(f"      (p-value: {p_value})") # Will be 0.0, confirming they are different

    # 3. GAR at FAR
    print(f"  [3] GAR (%) at specific FAR levels:")
    gar_results = calculate_gar_at_far(genuine, impostor, FAR_TARGETS)
    
    for far, data in gar_results.items():
        gar_percent = data['GAR'] * 100
        print(f"      - FAR @ {far*100:6.3f}%  ->  GAR = {gar_percent:6.3f}% (Threshold = {data['Threshold']:.4f})")

if __name__ == "__main__":
    # Analyze the Raw (unprotected) system
    analyze_system(
        "Raw",
        os.path.join(SCORE_DIR, "raw_genuine.npy"),
        os.path.join(SCORE_DIR, "raw_impostor.npy")
    )
    
    # Analyze the Protected system
    analyze_system(
        "Protected",
        os.path.join(SCORE_DIR, "protected_genuine.npy"),
        os.path.join(SCORE_DIR, "protected_impostor.npy")
    )
