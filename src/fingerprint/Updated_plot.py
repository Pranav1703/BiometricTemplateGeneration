import numpy as np
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.config import ANALYSIS_LOG_DIR, SCORES_DIR

# ---------------------------
# Custom Color Palette
# ---------------------------
COLOR_GENUINE = '#0072B2'    
COLOR_IMPOSTOR = '#D55E00'   
COLOR_BACKGROUND = "#FFFFFF" 
COLOR_TEXT = '#333333'       
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
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLOR_BACKGROUND)
    ax.set_facecolor(COLOR_BACKGROUND)
    
    # Plot Impostor and Genuine with 60% opacity (alpha=0.6)
    ax.hist(impostor, bins=100, alpha=0.6, color=COLOR_IMPOSTOR, label='Impostor (Different Fingers)', density=True)
    ax.hist(genuine, bins=100, alpha=0.6, color=COLOR_GENUINE, label='Genuine (Same Finger)', density=True)
    
    # Add a vertical dashed line where your EER threshold is
    ax.axvline(x=threshold, color=COLOR_TEXT, linestyle='--', linewidth=2, label=f'EER Threshold ({threshold:.2f})')
    
    # Styling text and grid
    ax.set_title(f'Score Distribution: {dataset_name.upper()} - {system_name}\nEER = {eer:.2%}', 
                 fontsize=14, color=COLOR_TEXT, fontweight='bold')
    ax.set_xlabel('Similarity Score (0.0 to 1.0)', fontsize=12, color=COLOR_TEXT)
    ax.set_ylabel('Density', fontsize=12, color=COLOR_TEXT)
    
    legend = ax.legend(loc='upper center', facecolor=COLOR_BACKGROUND, edgecolor=COLOR_TEXT)
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT)
        
    ax.grid(True, color=COLOR_TEXT, alpha=0.2)
    ax.tick_params(colors=COLOR_TEXT)
    for spine in ax.spines.values():
        spine.set_color(COLOR_TEXT)
    
    # Save the plot
    os.makedirs("artifacts/plots", exist_ok=True)
    filename = f"artifacts/plots/{dataset_name}_{system_name}_Distribution.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved overlapping graph to {filename}")

# ---------------------------
# Centered 3-Plot Grid (Top: 2, Bottom: 1 Centered)
# ---------------------------
def save_combined_grid_plot(scenario_data, title, save_filename, expected_datasets):
    """
    Generates a custom grid plotting 3 datasets using the custom palette.
    Top Row: casia, fvc2000
    Bottom Row: fvc2004 (centered)
    """
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(COLOR_BACKGROUND)
    
    # Create a 2x4 grid to allow centering the bottom plot
    gs = gridspec.GridSpec(2, 4, figure=fig)
    
    # Assign specific grid positions
    ax1 = fig.add_subplot(gs[0, 0:2]) # Top Left (Spans cols 0 and 1)
    ax2 = fig.add_subplot(gs[0, 2:4]) # Top Right (Spans cols 2 and 3)
    ax3 = fig.add_subplot(gs[1, 1:3]) # Bottom Middle (Spans cols 1 and 2)
    
    axes = [ax1, ax2, ax3]

    for i, dataset_name in enumerate(expected_datasets):
        if i >= 3: 
            break # Safety break if array size changes
            
        ax = axes[i]
        ax.set_facecolor(COLOR_BACKGROUND)
        
        if dataset_name in scenario_data:
            genuine = scenario_data[dataset_name]['gen']
            impostor = scenario_data[dataset_name]['imp']
            eer = scenario_data[dataset_name]['eer']
            threshold = scenario_data[dataset_name]['thresh']

            ax.hist(impostor, bins=100, alpha=0.6, color=COLOR_IMPOSTOR, label='Impostor', density=True)
            ax.hist(genuine, bins=100, alpha=0.6, color=COLOR_GENUINE, label='Genuine', density=True)
            
            # Draw the threshold line
            ax.axvline(x=threshold, color=COLOR_TEXT, linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')
            
            # Styling text, grid, and borders
            ax.set_title(f"{dataset_name.upper()} - EER: {eer:.2%}", fontsize=14, fontweight='bold', color=COLOR_TEXT)
            ax.set_xlabel('Similarity Score (0.0 to 1.0)', fontsize=12, color=COLOR_TEXT)
            ax.set_ylabel('Density', fontsize=12, color=COLOR_TEXT)
            
            legend = ax.legend(loc='upper center', facecolor=COLOR_BACKGROUND, edgecolor=COLOR_TEXT)
            for text in legend.get_texts():
                text.set_color(COLOR_TEXT)
                
            ax.grid(True, color=COLOR_TEXT, alpha=0.2)
            ax.tick_params(colors=COLOR_TEXT)
            for spine in ax.spines.values():
                spine.set_color(COLOR_TEXT)
        else:
            ax.set_title(f"{dataset_name.upper()} - Missing Data", fontsize=14, fontweight='bold', color=COLOR_TEXT)
            ax.text(0.5, 0.5, 'Data not found', horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, color=COLOR_TEXT)

    plt.suptitle(title, fontsize=20, fontweight='bold', color=COLOR_TEXT)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs("artifacts/plots", exist_ok=True)
    filepath = os.path.join("artifacts/plots", save_filename)
    # Ensure the background color persists in the saved PDF
    plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved custom styled 3-plot grid to {filepath}")

# ---------------------------
# Main Execution
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Fingerprint Scores")
    parser.add_argument("--dataset", type=str, choices=["casia", "fvc2000", "fvc2004"],
                        help="The dataset to analyze (casia, fvc2000, fvc2004)")
    parser.add_argument("--combined", action="store_true", 
                        help="Generate the 3 combined custom PDF grids for the e-portfolio")
    args = parser.parse_args()

    if args.combined:
        print("--- Generating Centered 3-Plot PDF Grids with Custom Palette ---")
        # Ensure array order exactly matches desired layout: casia, fvc2000, fvc2004
        all_datasets = ["casia", "fvc2000", "fvc2004"] 
        raw_data, biohashed_data, stolen_data = {}, {}, {}

        for ds in all_datasets:
            try:
                # Load scores for the current dataset
                raw_gen = np.load(os.path.join(SCORES_DIR, f"{ds}_raw_gen.npy"))
                raw_imp = np.load(os.path.join(SCORES_DIR, f"{ds}_raw_imp.npy"))
                prot_gen = np.load(os.path.join(SCORES_DIR, f"{ds}_prot_gen.npy"))
                prot_imp = np.load(os.path.join(SCORES_DIR, f"{ds}_prot_imp.npy"))
                stolen_imp = np.load(os.path.join(SCORES_DIR, f"{ds}_stolen_imp.npy"))

                # Calculate metrics for the combined plots
                raw_eer, raw_thresh, _, _, _ = calculate_curves_and_eer(raw_gen, raw_imp)
                prot_eer, prot_thresh, _, _, _ = calculate_curves_and_eer(prot_gen, prot_imp)
                stolen_eer, stolen_thresh, _, _, _ = calculate_curves_and_eer(prot_gen, stolen_imp)

                # Store arrays and metrics in dictionaries
                raw_data[ds] = {'gen': raw_gen, 'imp': raw_imp, 'eer': raw_eer, 'thresh': raw_thresh}
                biohashed_data[ds] = {'gen': prot_gen, 'imp': prot_imp, 'eer': prot_eer, 'thresh': prot_thresh}
                stolen_data[ds] = {'gen': prot_gen, 'imp': stolen_imp, 'eer': stolen_eer, 'thresh': stolen_thresh}
                
            except FileNotFoundError:
                print(f"Warning: Missing .npy files for {ds}. Generating grid with missing panels.")

        # Generate the 3 requested PDF files
        save_combined_grid_plot(raw_data, "Raw Biometric Distribution", "Combined_Raw_Distribution.pdf", all_datasets)
        save_combined_grid_plot(biohashed_data, "BioHashed Distribution", "Combined_BioHashed_Distribution.pdf", all_datasets)
        save_combined_grid_plot(stolen_data, "Stolen Key Scenario Distribution", "Combined_StolenKey_Distribution.pdf", all_datasets)
        
        print("\nCombined grids successfully generated in artifacts/plots/")

    elif args.dataset:
        # Dynamic Path Setup for single dataset mode
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
            
            stolen_imp = np.load(os.path.join(SCORES_DIR, f"{score_prefix}_stolen_imp.npy"))
        except FileNotFoundError as e:
            print(f"Error: Could not find scores for {args.dataset}. Run gen_embedding.py first.")
            exit()

        # --- 2. Calculate curves ---
        raw_eer, raw_thresh, raw_far, raw_frr, thresholds = calculate_curves_and_eer(raw_gen, raw_imp)
        prot_eer, prot_thresh, prot_far, prot_frr, _ = calculate_curves_and_eer(prot_gen, prot_imp)
        stolen_eer, stolen_thresh, stolen_far, stolen_frr, _ = calculate_curves_and_eer(prot_gen, stolen_imp)

        # --- 3. Log individual metrics ---
        log_individual_metrics(writer, "1_Raw_Backbone", raw_gen, raw_imp, raw_eer, raw_thresh)
        save_overlapping_histogram(raw_gen, raw_imp, "1_Raw_Backbone", raw_eer, raw_thresh, args.dataset)
        
        log_individual_metrics(writer, "2_Bio_Hashed_Normal", prot_gen, prot_imp, prot_eer, prot_thresh)
        save_overlapping_histogram(prot_gen, prot_imp, "2_Bio_Hashed_Normal", prot_eer, prot_thresh, args.dataset)
        
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
        
    else:
        print("Error: You must provide either --dataset <name> or --combined")