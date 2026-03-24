import numpy as np
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
from src.config import ANALYSIS_LOG_DIR, SCORES_DIR

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

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Fingerprint Scores")
    parser.add_argument("--dataset", type=str, required=True, choices=["casia", "fvc2000", "fvc2004"],
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
    except FileNotFoundError as e:
        print(f"Error: Could not find scores for {args.dataset}. Run gen_embedding.py first.")
        exit()

    # --- 2. Calculate curves ---
    raw_eer, raw_thresh, raw_far, raw_frr, thresholds = calculate_curves_and_eer(raw_gen, raw_imp)
    prot_eer, prot_thresh, prot_far, prot_frr, _ = calculate_curves_and_eer(prot_gen, prot_imp)

    # --- 3. Log individual metrics ---
    log_individual_metrics(writer, "1_Raw_Backbone", raw_gen, raw_imp, raw_eer, raw_thresh)
    log_individual_metrics(writer, "2_Bio_Hashed_Protected", prot_gen, prot_imp, prot_eer, prot_thresh)
    
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