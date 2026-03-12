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
def log_individual_metrics(writer, system_name, genuine, impostor, eer, threshold, hparams=None):
    """Logs the histograms, scalars, and ROC curve for a single system."""
    print(f"--- Logging metrics for: {system_name} ---")
    print(f"EER: {eer:.3%}, Threshold: {threshold:.3f}\n")
    
    writer.add_histogram(f"{system_name}/Score_Dist/Genuine", genuine, 0)
    writer.add_histogram(f"{system_name}/Score_Dist/Impostor", impostor, 0)
    
    metrics_dict = {'EER': eer, 'EER_Threshold': threshold}
    writer.add_scalars(f"Performance_Summary/{system_name}", metrics_dict, 0)
    
    labels = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    predictions = np.concatenate([genuine, impostor])
    
    # Downsample for PR Curve if dataset is too large
    if len(labels) > 200000:
        indices = np.random.choice(len(labels), 200000, replace=False)
        labels, predictions = labels[indices], predictions[indices]
    
    writer.add_pr_curve(f'ROC_Curve/{system_name}', labels, predictions, global_step=0)
    
    if hparams:
        flat_metrics = {f'hparam/{k}': v for k, v in metrics_dict.items()}
        writer.add_hparams(hparams, flat_metrics)

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Fingerprint Scores")
    parser.add_argument("--dataset", type=str, required=True, choices=["casia", "fvc2004"],
                        help="The dataset to analyze (casia or fvc2004)")
    parser.add_argument("--alpha", type=float, default=0.6, help="Alpha value used for protection")
    args = parser.parse_args()

    # Dynamic Path Setup
    # Matches the format: {dataset}_{mode}_{type}.npy
    score_prefix = f"{args.dataset}"
    run_name = f"{args.dataset}_alpha_{args.alpha}"
    writer = SummaryWriter(os.path.join(ANALYSIS_LOG_DIR, run_name))

    print(f"Starting Analysis for Dataset: {args.dataset.upper()}")

    # --- 1. Load score files dynamically ---
    try:
        raw_gen = np.load(os.path.join(SCORES_DIR, f"{score_prefix}_raw_gen.npy"))
        raw_imp = np.load(os.path.join(SCORES_DIR, f"{score_prefix}_raw_imp.npy"))
        prot_gen = np.load(os.path.join(SCORES_DIR, f"{score_prefix}_prot_gen.npy"))
        prot_imp = np.load(os.path.join(SCORES_DIR, f"{score_prefix}_prot_imp.npy"))
    except FileNotFoundError as e:
        print(f"Error: Could not find scores for {args.dataset}. Ensure you ran evaluation first.")
        print(f"Details: {e}")
        exit()

    # --- 2. Calculate curves ---
    raw_eer, raw_thresh, raw_far, raw_frr, thresholds = calculate_curves_and_eer(raw_gen, raw_imp)
    prot_eer, prot_thresh, prot_far, prot_frr, _ = calculate_curves_and_eer(prot_gen, prot_imp)

    # --- 3. Log individual metrics ---
    log_individual_metrics(writer, "Raw", raw_gen, raw_imp, raw_eer, raw_thresh)
    
    hparams = {"dataset": args.dataset, "alpha": args.alpha, "protected": True}
    log_individual_metrics(writer, "Protected", prot_gen, prot_imp, prot_eer, prot_thresh, hparams)
    
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