import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------
# 1. THE REFINED PALETTE
# ---------------------------
COLOR_PRIMARY = '#006D77'    # Deep Teal (CASIA)
COLOR_SECONDARY = '#E23838'  # Crimson (FVC2000)
COLOR_TERTIARY = '#457B9D'   # Slate Blue (FVC2004)
COLOR_BACKGROUND = '#F1FAEE' # Off-White/Ice
COLOR_TEXT = '#1D3557'       # Dark Navy

# Force reset styles to prevent environment hijacking
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('default')

def smooth_curve(scalars, weight=0.85):
    last = scalars.iloc[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_final_report_loss():
    # Use your specific absolute paths here as you did in your code
    file_casia = r"D:\Btech\Btech-4\Sem-8\BiometricTemplateGeneration\artifacts\plots\casia_Quantized_20260322-143401.csv"
    file_fvc2000 = r"D:\Btech\Btech-4\Sem-8\BiometricTemplateGeneration\artifacts\plots\fvc2000_Quantized_20260322-220217.csv"
    file_fvc2004 = r"D:\Btech\Btech-4\Sem-8\BiometricTemplateGeneration\artifacts\plots\fvc2004_Quantized_20260322-230045.csv"

    try:
        df_casia = pd.read_csv(file_casia)
        df_fvc2000 = pd.read_csv(file_fvc2000)
        df_fvc2004 = pd.read_csv(file_fvc2004)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    fig.patch.set_facecolor(COLOR_BACKGROUND)
    ax.set_facecolor(COLOR_BACKGROUND)

    sw = 0.85

    # Plotting with Z-Order to keep smooth lines on top
    for df, color, label in [(df_casia, COLOR_PRIMARY, 'CASIA'), 
                             (df_fvc2000, COLOR_SECONDARY, 'FVC2000'), 
                             (df_fvc2004, COLOR_TERTIARY, 'FVC2004')]:
        ax.plot(df['Step'], df['Value'], color=color, alpha=0.1, zorder=1)
        ax.plot(df['Step'], smooth_curve(df['Value'], sw), color=color, 
                linewidth=3, label=label, zorder=2)

    # TITLES & LABELS
    ax.set_title("ArcFace Training Loss Convergence", color=COLOR_TEXT, fontsize=18, fontweight='bold', pad=25)
    ax.set_xlabel("Global Step", color=COLOR_TEXT, fontsize=13, fontweight='bold')
    ax.set_ylabel("Loss Magnitude", color=COLOR_TEXT, fontsize=13, fontweight='bold')

    # MODERN AXIS STYLING
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_TEXT)
    ax.spines['bottom'].set_color(COLOR_TEXT)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    ax.tick_params(colors=COLOR_TEXT, labelsize=11)
    ax.grid(True, color=COLOR_TEXT, alpha=0.08, linestyle='--', zorder=0)

    # CLEAN LEGEND (No Box)
    legend = ax.legend(frameon=False, fontsize=12, loc='upper right')
    for text in legend.get_texts():
        text.set_color(COLOR_TEXT)
        text.set_weight('bold')

    plt.tight_layout()
    output = r"D:\Btech\Btech-4\Sem-8\BiometricTemplateGeneration\artifacts\plots\Final_Professional_Loss.pdf"
    plt.savefig(output, format='pdf', facecolor=COLOR_BACKGROUND)
    print(f"Success: {output}")

if __name__ == "__main__":
    plot_final_report_loss()