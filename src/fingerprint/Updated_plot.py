import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def create_combined_distribution_plot(scenario_data, title, save_filename):
    """
    Generates a 2x2 grid plot for 4 different datasets.
    
    :param scenario_data: Dictionary containing the datasets and their scores.
                          Format: {'Dataset1': {'genuine': [...], 'imposter': [...]}, ...}
    :param title: The main title for the entire figure.
    :param save_filename: The output filename (e.g., 'Combined_Raw_Distribution.pdf').
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, (dataset_name, scores) in enumerate(scenario_data.items()):
        if i >= 4:
            break # Ensure we only plot up to 4 datasets in the 2x2 grid
            
        ax = axes[i]
        genuine_scores = scores.get('genuine', [])
        imposter_scores = scores.get('imposter', [])

        # Plot Genuine Scores
        if len(genuine_scores) > 0:
            sns.kdeplot(genuine_scores, ax=ax, color='green', fill=True, 
                        label='Genuine', alpha=0.5, linewidth=2)
            
        # Plot Imposter Scores
        if len(imposter_scores) > 0:
            sns.kdeplot(imposter_scores, ax=ax, color='red', fill=True, 
                        label='Imposter', alpha=0.5, linewidth=2)

        # Subplot formatting
        ax.set_title(f"{dataset_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel('Score / Distance', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='upper right', fontsize=11)

    # Main figure formatting
    plt.suptitle(title, fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to fit the main title
    
    # Save as high-quality PDF
    plt.savefig(save_filename, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Successfully saved: {save_filename}")
    plt.close()

# ==========================================
# Data Integration & Execution
# ==========================================
if __name__ == "__main__":
    # TODO: Replace the random data generators below with your actual data arrays
    # loaded from your biometric template generation pipeline (e.g., from ComputeScores.py).
    
    def get_dummy_data(scenario_offset):
        """Helper to generate dummy distributions representing your datasets."""
        datasets = ['FVC2000', 'FVC2002', 'FVC2004', 'CASIA']
        data = {}
        for ds in datasets:
            # Shift distributions slightly to simulate different scenarios and datasets
            data[ds] = {
                'genuine': np.random.normal(loc=0.2 + scenario_offset, scale=0.08, size=1000),
                'imposter': np.random.normal(loc=0.8 - scenario_offset, scale=0.1, size=5000)
            }
        return data

    # 1. Generate Raw Distribution
    raw_data = get_dummy_data(scenario_offset=0.0)
    create_combined_distribution_plot(
        scenario_data=raw_data,
        title="Raw Biometric Distribution across Datasets",
        save_filename="Combined_Raw_Distribution.pdf"
    )

    # 2. Generate BioHashed Distribution (Expected to have better separation)
    biohashed_data = get_dummy_data(scenario_offset=-0.1) 
    create_combined_distribution_plot(
        scenario_data=biohashed_data,
        title="BioHashed Distribution across Datasets",
        save_filename="Combined_BioHashed_Distribution.pdf"
    )

    # 3. Generate Stolen Key Distribution (Expected to overlap significantly)
    stolen_key_data = get_dummy_data(scenario_offset=0.25)
    create_combined_distribution_plot(
        scenario_data=stolen_key_data,
        title="Stolen Key Scenario Distribution across Datasets",
        save_filename="Combined_StolenKey_Distribution.pdf"
    )