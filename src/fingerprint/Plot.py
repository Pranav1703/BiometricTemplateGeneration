import numpy as np
import os
import wandb 
import traceback # <-- Import traceback

# ---------------------------
# Configuration
# ---------------------------
SCORE_DIR = "artifacts/scores"
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
# Helper logging function for system-specific plots
# ---------------------------
def log_system_plots(system_name, genuine, impostor):
    """Logs the histograms and ROC curve for a single system."""
    print(f"--- Logging plots for: {system_name} System ---")
    
    wandb.log({
        f'2_Score_Distributions/Genuine_Scores/{system_name}': wandb.Histogram(genuine),
        f'2_Score_Distributions/Impostor_Scores/{system_name}': wandb.Histogram(impostor)
    })

    labels = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
    predictions = np.concatenate([genuine, impostor])
    
    if len(labels) > 200000:
        indices = np.random.choice(len(labels), 200000, replace=False)
        labels, predictions = labels[indices], predictions[indices]
    
    # wandb.plot.roc_curve expects probabilities for each class
    # We pass [P(impostor), P(genuine)] which is [1-score, score]
    wandb.log({
        f'4_ROC_Curves/GAR_vs_FAR/{system_name}': wandb.plot.roc_curve(
            labels, 
            np.stack([1-predictions, predictions], axis=1)
            # Removed classes_to_plot argument to fix ValueError
        )
    })

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    run_name = f"alpha_{ALPHA_USED}_final_analysis"
    
    # --- 1. Initialize W&B Run ---
    try:
        wandb.init(
            project="Biometric_Analysis", 
            name=run_name,
            config={
                "alpha": ALPHA_USED,
                "score_dir": SCORE_DIR
            }
        )
        print("--- W&B Init Succeeded. Starting analysis... ---")
        
    except Exception as e:
        print(f"Error initializing wandb: {e}")
        exit()

    
    # --- NEW: MASTER ERROR CATCHER ---
    # This will catch ANY error during the main process
    try:
        # --- 2. Load all score files ---
        print("--- Loading score files... ---")
        raw_genuine = np.load(os.path.join(SCORE_DIR, "raw_genuine.npy"))
        raw_impostor = np.load(os.path.join(SCORE_DIR, "raw_impostor.npy"))
        prot_genuine = np.load(os.path.join(SCORE_DIR, "protected_genuine.npy"))
        prot_impostor = np.load(os.path.join(SCORE_DIR, "protected_impostor.npy"))
        print("--- Score files loaded successfully. ---")

        # --- 3. Calculate curves for both systems ---
        print("--- Calculating EER and curves... ---")
        raw_eer, raw_thresh, raw_far, raw_frr, thresholds = calculate_curves_and_eer(raw_genuine, raw_impostor)
        prot_eer, prot_thresh, prot_far, prot_frr, _ = calculate_curves_and_eer(prot_genuine, prot_impostor)

        # --- 4. Log System-Specific Plots (Histograms, ROC) ---
        log_system_plots("Raw", raw_genuine, raw_impostor)
        # !!! CRITICAL FIX HERE !!!
        # Was: log_system_plots("Protected", prot_impostor, prot_impostor)
        log_system_plots("Protected", prot_genuine, prot_impostor) 
        
        # --- 5. Log Performance Summary (EER & Threshold) ---
        print("\n--- Logging Performance Summary to W&B Summary ---")
        # wandb.summary is for metrics that summarize the *entire* run
        wandb.summary["Raw_EER"] = raw_eer
        wandb.summary["Raw_EER_Threshold"] = raw_thresh
        wandb.summary["Protected_EER"] = prot_eer
        wandb.summary["Protected_EER_Threshold"] = prot_thresh

        # --- 6. Log the FAR/FRR plots as separate graphs ---
        print("--- Logging separate FAR/FRR comparison plots ---")
        
        # --- 6a. Log Raw FAR/FRR plot ---
        wandb.log({
            "3_Error_Rate_Curves/FAR_vs_FRR_(Raw)": wandb.plot.line_series(
                xs=thresholds,
                ys=[raw_far, raw_frr],
                keys=["FAR (Raw)", "FRR (Raw)"],
                title="FAR/FRR vs. Decision Threshold (Raw System)",
                xname="Threshold"
            )
        })
        
        # --- 6b. Log Protected FAR/FRR plot ---
        wandb.log({
            "3_Error_Rate_Curves/FAR_vs_FRR_(Protected)": wandb.plot.line_series(
                xs=thresholds,
                ys=[prot_far, prot_frr],
                keys=["FAR (Protected)", "FRR (Protected)"],
                title="FAR/FRR vs. Decision Threshold (Protected System)",
                xname="Threshold"
            )
        })

        # --- 7. Finish the run ---
        wandb.finish() 
        print("\nAnalysis complete. View results in Weights & Biases.")

    except Exception as e:
        # --- THIS WILL CATCH THE ERROR ---
        print("\n" + "="*50)
        print("!!! AN ERROR OCCURRED !!!")
        print("="*50 + "\n")
        print(f"Error Type: {type(e)}")
        print(f"Error Details: {e}")
        print("\nFull Traceback:")
        traceback.print_exc() # Prints the full error traceback
        
        # Log the error to W&B before crashing
        wandb.log({"error": str(e), "traceback": traceback.format_exc()})
        wandb.finish(exit_code=1) # Finish the run with a non-zero exit code
        print("\nScript terminated with an error. Error details logged to W&B.")

