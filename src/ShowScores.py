import numpy as np

# ---------------------------
# Load scores
# ---------------------------
raw_genuine = np.load("raw_genuine.npy")
raw_impostor = np.load("raw_impostor.npy")
prot_genuine = np.load("protected_genuine.npy")
prot_impostor = np.load("protected_impostor.npy")

# ---------------------------
# Function to print stats
# ---------------------------
def print_score_stats(name, scores):
    print(f"--- {name} ---")
    print(f"Number of scores: {len(scores)}")
    print(f"Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")
    print(f"Mean: {np.mean(scores):.4f}, Std: {np.std(scores):.4f}")
    print(f"First 10 scores: {scores[:10]}")
    print()

# ---------------------------
# Print raw scores
# ---------------------------
print_score_stats("Raw Genuine Scores", raw_genuine)
print_score_stats("Raw Impostor Scores", raw_impostor)

# ---------------------------
# Print protected scores
# ---------------------------
print_score_stats("Protected Genuine Scores", prot_genuine)
print_score_stats("Protected Impostor Scores", prot_impostor)
