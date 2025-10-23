# Save this as your updated analysis script (e.g., Plot.py)
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

# ---------------------------
# Configuration
# ---------------------------
SCORE_DIR = "artifacts/scores"
LOG_DIR = "logs/analysis"
ALPHA_USED = 0.6 

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
    print(f"--- Logging individual metrics for: {system_name} System ---")
    print(f"EER: {eer:.3%}, Threshold: {threshold:.3f}\n")
    
    # Log Histograms, individual EER, ROC Curve, and HParams
    writer.add_histogram(f"{system_name}/Genuine_Scores", genuine, 0)
    writer.add_histogram(f"{system_name}/Impostor_Scores", impostor, 0)
    metrics_dict = {'EER': eer, 'EER_Threshold': threshold}
    writer.add_scalars(f"Performance/{system_name}", metrics_dict, 0)
    labels = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    predictions = np.concatenate([genuine, impostor])
    if len(labels) > 200000:
        indices = np.random.choice(len(labels), 200000, replace=False)
        labels, predictions = labels[indices], predictions[indices]
    writer.add_pr_curve(f'ROC_Curve/{system_name}', labels, predictions, global_step=0)
    if hparams:
        flat_metrics = {f'hparam/{k}': v for k, v in metrics_dict.items()}
        writer.add_hparams(hparams, flat_metrics)

# ---------------------------
# Main Execution (RESTRUCTURED FOR COMBINED PLOT)
# ---------------------------
if __name__ == "__main__":
    run_name = f"alpha_{ALPHA_USED}_comparison_plot"
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))

    # --- 1. Load all score files ---
    try:
        raw_genuine = np.load(os.path.join(SCORE_DIR, "raw_genuine.npy"))
        raw_impostor = np.load(os.path.join(SCORE_DIR, "raw_impostor.npy"))
        prot_genuine = np.load(os.path.join(SCORE_DIR, "protected_genuine.npy"))
        prot_impostor = np.load(os.path.join(SCORE_DIR, "protected_impostor.npy"))
    except FileNotFoundError as e:
        print(f"Error loading score files: {e}")
        exit()

    # --- 2. Calculate curves for both systems ---
    raw_eer, raw_thresh, raw_far, raw_frr, thresholds = calculate_curves_and_eer(raw_genuine, raw_impostor)
    prot_eer, prot_thresh, prot_far, prot_frr, _ = calculate_curves_and_eer(prot_genuine, prot_impostor)

    # --- 3. Log individual metrics for each system (like before) ---
    log_individual_metrics(writer, "Raw", raw_genuine, raw_impostor, raw_eer, raw_thresh)
    hparams = {"alpha": ALPHA_USED, "protection": True}
    log_individual_metrics(writer, "Protected", prot_genuine, prot_impostor, prot_eer, prot_thresh, hparams)
    
    # --- 4. Log the combined FAR/FRR plot with clear labels ---
    print("--- Logging combined FAR/FRR comparison plot ---")
    for i, t in enumerate(thresholds):
        writer.add_scalars(
            'Comparison/FAR_vs_FRR_Curves',  # The title of the new combined graph
            {
                'FAR (Raw)': raw_far[i],
                'FRR (Raw)': raw_frr[i],
                'FAR (Protected)': prot_far[i],
                'FRR (Protected)': prot_frr[i]
            },
            global_step=i
        )

    writer.close()
    print("\nAnalysis complete. View results in TensorBoard.")
    print(f"Run 'tensorboard --logdir={LOG_DIR}' in your terminal.")