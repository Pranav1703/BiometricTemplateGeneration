import numpy as np
import os
import argparse
import csv
from scipy.stats import ks_2samp
from src.config import SCORES_DIR, PLOTS_DIR

FAR_TARGETS = [0.01, 0.001, 0.0001]
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
    if len(impostor_scores) > 500000:
        # Downsample to speed up KS test if we have millions of scores
        downsampled_impostors = np.random.choice(impostor_scores, 500000, replace=False)
    else:
        downsampled_impostors = impostor_scores
        
    ks_result = ks_2samp(genuine_scores, downsampled_impostors)
    return ks_result.statistic

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
# Core Analysis Logic
# ---------------------------
def analyze_dataset_scenarios(dataset_name, csv_writer):
    """
    Loads the score files for a dataset, computes d' and KS stat for the
    three main scenarios, and writes the results to the CSV.
    """
    print(f"\n{'='*55}")
    print(f" Analyzing Separability Metrics: {dataset_name.upper()} ")
    print(f"{'='*55}")

    # File paths based on the gen_embeddings.py output format
    raw_gen_path = os.path.join(SCORES_DIR, f"{dataset_name}_raw_gen.npy")
    raw_imp_path = os.path.join(SCORES_DIR, f"{dataset_name}_raw_imp.npy")
    prot_gen_path = os.path.join(SCORES_DIR, f"{dataset_name}_prot_gen.npy")
    prot_imp_path = os.path.join(SCORES_DIR, f"{dataset_name}_prot_imp.npy")
    stolen_imp_path = os.path.join(SCORES_DIR, f"{dataset_name}_stolen_imp.npy")

    # Check if files exist
    if not os.path.exists(raw_gen_path):
        print(f"Error: Scores for {dataset_name} not found in {SCORES_DIR}.")
        print(f"Please run 'python -m src.fingerprint.gen_embeddings --dataset {dataset_name}' first.")
        return

    # Load arrays
    raw_gen = np.load(raw_gen_path)
    raw_imp = np.load(raw_imp_path)
    prot_gen = np.load(prot_gen_path)
    prot_imp = np.load(prot_imp_path)
    stolen_imp = np.load(stolen_imp_path)

    # Define the 3 evaluation scenarios
    scenarios = [
        ("Raw Backbone", raw_gen, raw_imp),
        ("Protected (Bio-Hashed)", prot_gen, prot_imp),
        ("Stolen Key Scenario", prot_gen, stolen_imp)
    ]

    for scenario_name, gen_scores, imp_scores in scenarios:
        d_prime = calculate_d_prime(gen_scores, imp_scores)
        ks_stat = calculate_ks_test(gen_scores, imp_scores)
        gar_results = calculate_gar_at_far(gen_scores, imp_scores, FAR_TARGETS)
        
        print(f"--- {scenario_name} ---")
        print(f"  d-prime (Separability): {d_prime:.4f}")
        print(f"  KS Statistic (D):       {ks_stat:.4f}\n")

        for far, data in gar_results.items():
            gar_percent = data['GAR'] * 100
            print(f"      - FAR @ {far*100:6.3f}%  ->  GAR = {gar_percent:6.3f}% (Threshold = {data['Threshold']:.4f})")

        # Write to CSV
        csv_writer.writerow([
            dataset_name.upper(),
            scenario_name,
            f"{d_prime:.4f}",
            f"{ks_stat:.4f}",
            f"{gar_results[0.01]['GAR'] * 100:.3f}%",
            f"{gar_results[0.001]['GAR'] * 100:.3f}%",
            f"{gar_results[0.0001]['GAR'] * 100:.3f}%"
        ])

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and store Distribution Separability Metrics")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["casia", "fvc2000", "fvc2004", "cmbd", "all"],
                        help="The dataset to analyze, or 'all' to process all available.")
    
    args = parser.parse_args()

    # Determine which datasets to run
    datasets_to_run = ["casia", "fvc2000", "fvc2004", "cmbd"] if args.dataset == "all" else [args.dataset]

    # Setup CSV output
    os.makedirs(PLOTS_DIR, exist_ok=True)
    csv_file_path = os.path.join(PLOTS_DIR, "distribution_separability_metrics.csv")
    
    # Check if we need to write the header row
    write_header = not os.path.exists(csv_file_path)
    
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Dataset", "Evaluation Scenario", "d' (Separability Index)", "KS Statistic (D)", "GAR @ FAR=1%", "GAR @ FAR=0.1%", "GAR @ FAR=0.01%"])

        for ds in datasets_to_run:
            analyze_dataset_scenarios(ds, writer)

    print(f"\nSuccess! Separability metrics saved to {csv_file_path}")