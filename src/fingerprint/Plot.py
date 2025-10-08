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
# EER and Curve Calculation Function (MODIFIED TO RETURN CURVES)
# ---------------------------
def calculate_curves_and_eer(genuine_scores, impostor_scores):
    """
    Calculates the full FAR/FRR curves and finds the EER.
    Now returns all the data needed for plotting.
    """
    thresholds = np.linspace(0, 1, 1001)
    far_list = []
    frr_list = []

    for t in thresholds:
        far = np.mean(impostor_scores >= t)
        frr = np.mean(genuine_scores < t)
        far_list.append(far)
        frr_list.append(frr)
    
    far_list = np.array(far_list)
    frr_list = np.array(frr_list)

    # Find the EER point
    diff = np.abs(far_list - frr_list)
    eer_index = np.argmin(diff)
    eer_threshold = thresholds[eer_index]
    eer = (far_list[eer_index] + frr_list[eer_index]) / 2.0
    
    return eer, eer_threshold, far_list, frr_list, thresholds

# ---------------------------
# Main Analysis and Logging Function (UPDATED)
# ---------------------------
def analyze_and_log(writer, system_name, genuine_path, impostor_path, hparams=None):
    """Loads scores, performs analysis, and logs to TensorBoard."""
    print(f"--- Analyzing & Logging: {system_name} System ---")
    try:
        genuine = np.load(genuine_path)
        impostor = np.load(impostor_path)
    except FileNotFoundError:
        print(f"Error: Score files for '{system_name}' not found. Skipping.\n")
        return

    print(f"Loaded {len(genuine)} genuine and {len(impostor)} impostor scores.")

    # Get all the metrics and curve data
    eer, threshold, far_list, frr_list, thresholds = calculate_curves_and_eer(genuine, impostor)
    
    print(f"Equal Error Rate (EER): {eer:.3%}")
    print(f"Threshold at EER: {threshold:.3f}\n")

    # --- Log to TensorBoard ---
    
    # 1. Log Histograms of score distributions
    writer.add_histogram(f"{system_name}/Genuine_Scores", genuine, 0)
    writer.add_histogram(f"{system_name}/Impostor_Scores", impostor, 0)

    # 2. Log single-value metrics
    metrics_dict = {'EER': eer, 'EER_Threshold': threshold}
    writer.add_scalars(f"Performance/{system_name}", metrics_dict, 0)
    
    # 3. Log an ROC Curve 
    labels = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    predictions = np.concatenate([genuine, impostor])
    if len(labels) > 200000:
        indices = np.random.choice(len(labels), 200000, replace=False)
        labels, predictions = labels[indices], predictions[indices]
    writer.add_pr_curve(f'ROC_Curve/{system_name}', labels, predictions, global_step=0)

    # 4. Log Hyperparameters
    if hparams:
        flat_metrics = {f'hparam/{k}': v for k, v in metrics_dict.items()}
        writer.add_hparams(hparams, flat_metrics)

    # 5. --- NEW: Log FAR and FRR curves to a single graph ---
    # We loop through each threshold and log both FAR and FRR at each step.
    # The 'global_step' must be an integer, so we use the loop index 'i'.
    for i, t in enumerate(thresholds):
        writer.add_scalars(
            f'FAR_FRR_Curve/{system_name}',  # This will be the title of the graph
            {'FAR': far_list[i], 'FRR': frr_list[i]},
            global_step=i  # Use the integer index 'i' as the step
        )

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    run_name = f"alpha_{ALPHA_USED}_full_curves" # Give it a new name
    writer = SummaryWriter(os.path.join(LOG_DIR, run_name))

    raw_genuine_path = os.path.join(SCORE_DIR, "raw_genuine.npy")
    raw_impostor_path = os.path.join(SCORE_DIR, "raw_impostor.npy")
    analyze_and_log(writer, "Raw", raw_genuine_path, raw_impostor_path)
    
    prot_genuine_path = os.path.join(SCORE_DIR, "protected_genuine.npy")
    prot_impostor_path = os.path.join(SCORE_DIR, "protected_impostor.npy")
    hparams = {"alpha": ALPHA_USED, "protection": True}
    analyze_and_log(writer, "Protected", prot_genuine_path, prot_impostor_path, hparams=hparams)

    writer.close()
    print("Analysis complete. View results in TensorBoard.")
    print(f"Run 'tensorboard --logdir={LOG_DIR}' in your terminal.")