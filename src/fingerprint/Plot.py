import numpy as np
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from src.config import ANALYSIS_LOG_DIR, SCORES_DIR
import matplotlib.pyplot as plt

# ---------------------------
# EER and Curve Calculation Function
# ---------------------------
def calculate_curves_and_eer(genuine_scores, impostor_scores):
    """Calculates the full FAR/FRR curves and finds the EER."""
    thresholds = np.linspace(0, 1, 1001)
    far_list, frr_list = [], []
    for t in thresholds:
        far = np.mean(impostor_scores >= t)
        frr = np.mean(genuine_scores < t)
        far_list.append(far)
        frr_list.append(frr)
    
    far_list, frr_list = np.array(far_list), np.array(frr_list)
    diff = np.abs(far_list - frr_list)
    eer_index = np.argmin(diff)
    eer_threshold = thresholds[eer_index]
    eer = (far_list[eer_index] + frr_list[eer_index]) / 2.0
    return eer, eer_threshold, far_list, frr_list, thresholds

# ---------------------------
# Helper logging function for individual metrics
# ---------------------------
def log_individual_metrics(writer, system_name, genuine, impostor, eer, threshold):
    """Logs the histograms, scalars, and ROC curve for a single system."""
    print(f"--- Logging metrics for: {system_name} ---")
    print(f"EER: {eer:.3%}, Threshold: {threshold:.3f}\n")
    
    writer.add_histogram(f"{system_name}/Score_Dist/Genuine", genuine, 0)
    writer.add_histogram(f"{system_name}/Score_Dist/Impostor", impostor, 0)
    
    metrics_dict = {'EER': eer, 'EER_Threshold': threshold}
    writer.add_scalars(f"Performance_Summary/{system_name}", metrics_dict, 0)
    
    # 1 for Genuine (Match), 0 for Impostor (Non-Match)
    labels = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    predictions = np.concatenate([genuine, impostor])
    
    # Downsample for PR Curve if dataset is too large to prevent TensorBoard lag
    if len(labels) > 200000:
        indices = np.random.choice(len(labels), 200000, replace=False)
        labels, predictions = labels[indices], predictions[indices]
    
    writer.add_pr_curve(f'Precision_Recall_Curve/{system_name}', labels, predictions, global_step=0)

def save_overlapping_histogram(genuine, impostor, system_name, eer, threshold, dataset_name):
    """Generates and saves a presentation-ready overlapping distribution graph."""
    plt.figure(figsize=(10, 6))
    
    # Plot Impostor (Red) and Genuine (Green) with 60% opacity (alpha=0.6)
    plt.hist(impostor, bins=100, alpha=0.6, color='red', label='Impostor (Different Fingers)', density=True)
    plt.hist(genuine, bins=100, alpha=0.6, color='green', label='Genuine (Same Finger)', density=True)
    
    # Add a vertical dashed line where your EER threshold is
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'EER Threshold ({threshold:.2f})')
    
    plt.title(f'Score Distribution: {dataset_name.upper()} - {system_name}\nEER = {eer:.2%}', fontsize=14)
    plt.xlabel('Similarity Score (0.0 to 1.0)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(loc='upper center')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs("artifacts/plots", exist_ok=True)
    filename = f"artifacts/plots/{dataset_name}_{system_name}_Distribution.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved overlapping graph to {filename}")
# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Fingerprint Scores")
    parser.add_argument("--dataset", type=str, required=True, choices=["casia", "fvc2000", "fvc2004", "cmbd"],
                        help="The dataset to analyze (casia, fvc2000, or fvc2004)")
    args = parser.parse_args()

    # Dynamic Path Setup
    score_prefix = args.dataset
    run_name = f"{args.dataset}_bio_hashing_eval"
    writer = SummaryWriter(os.path.join(ANALYSIS_LOG_DIR, run_name))

    print(f"Starting Analysis for Dataset: {args.dataset.upper()}")

    # --- 1. Load score files dynamically ---
    try:
        raw_gen = np.load(os.path.join(SCORES_DIR, f"{score_prefix}_raw_gen.npy"))
        raw_imp = np.load(os.path.join(SCORES_DIR, f"{score_prefix}_raw_imp.npy"))
        
        prot_gen = np.load(os.path.join(SCORES_DIR, f"{score_prefix}_prot_gen.npy"))
        prot_imp = np.load(os.path.join(SCORES_DIR, f"{score_prefix}_prot_imp.npy"))
        
        # NEW: Load Stolen Impostor Scores
        stolen_imp = np.load(os.path.join(SCORES_DIR, f"{score_prefix}_stolen_imp.npy"))
    except FileNotFoundError as e:
        print(f"Error: Could not find scores for {args.dataset}. Run gen_embedding.py first.")
        exit()

    # --- 2. Calculate curves ---
    raw_eer, raw_thresh, raw_far, raw_frr, thresholds = calculate_curves_and_eer(raw_gen, raw_imp)
    prot_eer, prot_thresh, prot_far, prot_frr, _ = calculate_curves_and_eer(prot_gen, prot_imp)
    
    # NEW: Calculate Stolen Token EER (Genuine User vs Hacker with Stolen Key)
    stolen_eer, stolen_thresh, stolen_far, stolen_frr, _ = calculate_curves_and_eer(prot_gen, stolen_imp)

    # --- 3. Log individual metrics ---
    log_individual_metrics(writer, "1_Raw_Backbone", raw_gen, raw_imp, raw_eer, raw_thresh)
    save_overlapping_histogram(raw_gen, raw_imp, "1_Raw_Backbone", raw_eer, raw_thresh, args.dataset)
    
    log_individual_metrics(writer, "2_Bio_Hashed_Normal", prot_gen, prot_imp, prot_eer, prot_thresh)
    save_overlapping_histogram(prot_gen, prot_imp, "2_Bio_Hashed_Normal", prot_eer, prot_thresh, args.dataset)
    
    # NEW: Log Stolen Scenario
    log_individual_metrics(writer, "3_Bio_Hashed_Stolen_Key", prot_gen, stolen_imp, stolen_eer, stolen_thresh)
    save_overlapping_histogram(prot_gen, stolen_imp, "3_Bio_Hashed_Stolen_Key", stolen_eer, stolen_thresh, args.dataset)
    
    # --- 4. Log Combined Comparison ---
    print("--- Logging Comparison Curves ---")
    for i, t in enumerate(thresholds):
        writer.add_scalars(
            'Comparison/FAR_vs_FRR', 
            {
                'FAR_Raw': raw_far[i],
                'FRR_Raw': raw_frr[i],
                'FAR_Prot': prot_far[i],
                'FRR_Prot': prot_frr[i]
            },
            global_step=i
        )

    writer.close()
    print(f"\nAnalysis complete. Run: tensorboard --logdir={ANALYSIS_LOG_DIR}")