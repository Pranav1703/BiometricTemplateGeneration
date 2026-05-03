import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# Set up an academic, publication-ready plot style
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (8, 6),
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--'
})

# Path to your npy files
SCORES_DIR = "artifacts/scores"

# Datasets to plot
datasets = [
    ('fvc2000', 'FVC2000', '#ff7f0e'),  # Orange
    ('casia', 'CASIA', '#1f77b4'),      # Blue
    ('fvc2004', 'FVC2004', '#2ca02c')   # Green
]

fig, ax = plt.subplots()
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

for ds_prefix, name, color in datasets:
    try:
        # Load the actual genuine and impostor scores
        genuine = np.load(os.path.join(SCORES_DIR, f"{ds_prefix}_raw_gen.npy"))
        impostor = np.load(os.path.join(SCORES_DIR, f"{ds_prefix}_raw_imp.npy"))
        
        # Create labels: 1 for genuine, 0 for impostor
        y_true = np.concatenate([np.ones_like(genuine), np.zeros_like(impostor)])
        y_scores = np.concatenate([genuine, impostor])
        
        # Calculate ROC metrics
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Plot the curve
        ax.plot(fpr, tpr, label=f'{name} | AUC = {roc_auc:.4f}', color=color, linewidth=2.5)
        print(f"Processed {name} successfully.")
    except FileNotFoundError:
        print(f"Warning: Data for {name} not found in {SCORES_DIR}. Skipping.")

# Plot the random guess line
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random Guess')

# Focus the plot on the critical biometric operational area (Top Left)
ax.set_xlim([0.0, 0.15])
ax.set_ylim([0.85, 1.005])

ax.set_xlabel('False Positive Rate (FAR)')
ax.set_ylabel('True Positive Rate (GAR)')
ax.set_title('ROC Curve: Raw ResNet-50 Backbone')
ax.legend(loc='lower right', frameon=True, framealpha=0.9, edgecolor='black')

# Save as PDF for the LaTeX report
os.makedirs("figures", exist_ok=True)
plt.savefig('figures/ROC_Curve_Raw_Backbone.pdf', dpi=300, bbox_inches='tight')
print("\nSuccess! Saved ROC curve to figures/ROC_Curve_Raw_Backbone.pdf")
plt.show()