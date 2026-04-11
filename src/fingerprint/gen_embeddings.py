import os
import torch
import argparse
import numpy as np
import hashlib
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.Dataset_Loader import FingerprintDataset
from src.config import SAVED_MODELS_DIR, SCORES_DIR
from src.fingerprint.train import FingerprintEmbeddingNet, EMBEDDING_DIM, DEVICE, get_dataset_config

# ---------------------------
# Bio-Hashing Functions
# ---------------------------
def get_chaotic_sequence(seed_str: str, length: int, r: float = 3.99) -> torch.Tensor:
    """Generates a chaotic sequence based on a user key."""
    hash_val = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16)
    x = (hash_val % 10**10) / 10**10 
    
    sequence = torch.zeros(length, device=DEVICE)
    for i in range(length):
        x = r * x * (1 - x)
        sequence[i] = x
    return sequence

# ---------------------------
# Bio-Hashing Functions
# ---------------------------
def generate_chaotic_projection(user_key: str) -> torch.Tensor:
    """Generates an Orthogonal Projection Matrix (R) instantly on the GPU."""
    # 1. Map string key to a deterministic seed integer
    seed_val = int(hashlib.sha256(user_key.encode()).hexdigest(), 16) % (2**32)
    
    # 2. Seed a PyTorch generator (keeps it isolated from global random state)
    gen = torch.Generator(device=DEVICE)
    gen.manual_seed(seed_val)
    
    # 3. Generate the entire matrix instantly in C++/CUDA
    R_chaotic = torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), generator=gen, device=DEVICE)
    
    # 4. Orthonormalize using QR decomposition
    Q, _ = torch.linalg.qr(R_chaotic)
    return Q 

def apply_bio_hash(projected_embedding: torch.Tensor) -> torch.Tensor:
    # Use median to ensure 50% probability for each bit
    t = torch.median(projected_embedding)
    return (projected_embedding >= t).float()

# ---------------------------
# Main Execution
# ---------------------------
def main(dataset_name):
    config = get_dataset_config(dataset_name)
    model_path = os.path.join(SAVED_MODELS_DIR, config["model_name"])

    print(f"Loading {dataset_name} dataset for evaluation...")
    val_dataset = FingerprintDataset(config["train_csv"], train=False, dataset_type=dataset_name)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Load Backbone
    backbone = FingerprintEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    backbone.load_state_dict(checkpoint["backbone"])
    backbone.eval()

    all_raw_embeddings = []
    all_labels = []

    # 1. Extract Raw Embeddings
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Extracting Raw Features"):
            imgs = imgs.to(DEVICE)
            
            # FIX: Unpack the tuple to get the quantized bits
            quantized_bits, _ = backbone(imgs)
            
            all_raw_embeddings.append(quantized_bits.cpu())
            all_labels.append(labels.cpu())

    raw_embeddings = torch.cat(all_raw_embeddings, dim=0).to(DEVICE) # [N, 512]
    labels = torch.cat(all_labels, dim=0).to(DEVICE) # [N]

    # 2. Apply Bio-Hashing (Dynamic Projection)
    print("Pre-generating Bio-Hashing matrices...")
    unique_labels = torch.unique(labels)
    user_matrices = {}
    
    # Pre-generate the chaotic matrices for each unique identity WITH A LOADING BAR
    for label in tqdm(unique_labels, desc="Generating User Keys"):
        user_key = f"{dataset_name}_user_{label.item()}_secret_key"
        user_matrices[label.item()] = generate_chaotic_projection(user_key)

    prot_embeddings = torch.zeros_like(raw_embeddings)
    
    for i in tqdm(range(len(labels)), desc="Projecting Templates"):
        lbl = labels[i].item()
        R_matrix = user_matrices[lbl]
        projected = torch.matmul(R_matrix, raw_embeddings[i])
        prot_embeddings[i] = apply_bio_hash(projected)
    
    # ==========================================

    # 3. Compute Pairwise Scores (RAM-Safe Matrix Math)
    print("Computing millions of pairwise comparisons...")
    
    # Raw = Cosine Similarity
    raw_embeddings_norm = torch.nn.functional.normalize(raw_embeddings, p=2, dim=1)
    raw_sim_matrix = torch.matmul(raw_embeddings_norm, raw_embeddings_norm.T)

    # Protected = Hamming Similarity (Vectorized to avoid Out-Of-Memory errors)
    matches = torch.matmul(prot_embeddings, prot_embeddings.T) + torch.matmul(1 - prot_embeddings, 1 - prot_embeddings.T)
    prot_sim_matrix = matches / EMBEDDING_DIM

    # 4. Extract Genuine and Impostor Scores
    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
    
    # Remove the diagonal (don't compare an image to itself)
    label_matrix.fill_diagonal_(False)
    imp_mask = ~label_matrix
    imp_mask.fill_diagonal_(False)

    raw_gen = raw_sim_matrix[label_matrix].cpu().numpy()
    raw_imp = raw_sim_matrix[imp_mask].cpu().numpy()
    
    prot_gen = prot_sim_matrix[label_matrix].cpu().numpy()
    prot_imp = prot_sim_matrix[imp_mask].cpu().numpy()

        # ==========================================
    # NEW: 2.5 The Stolen Token Scenario
    # ==========================================
    print("Simulating Stolen Token (Hacker) Scenario...")
    stolen_prot_embeddings = torch.zeros_like(raw_embeddings)
    
    # We force EVERY fingerprint to be projected using User 0's Key
    stolen_hacker_matrix = user_matrices[unique_labels[0].item()]
    
    for i in range(len(labels)):
        projected = torch.matmul(stolen_hacker_matrix, raw_embeddings[i])
        stolen_prot_embeddings[i] = apply_bio_hash(projected)

    # Calculate Stolen Scores
    stolen_matches = torch.matmul(stolen_prot_embeddings, stolen_prot_embeddings.T) + torch.matmul(1 - stolen_prot_embeddings, 1 - stolen_prot_embeddings.T)
    stolen_sim_matrix = stolen_matches / EMBEDDING_DIM
    
    # We only care about the impostor scores here (Hacker vs Victim)
    stolen_imp = stolen_sim_matrix[imp_mask].cpu().numpy()

    # 5. Save Score Arrays
    os.makedirs(SCORES_DIR, exist_ok=True)
    prefix = os.path.join(SCORES_DIR, dataset_name)
    
    np.save(f"{prefix}_raw_gen.npy", raw_gen)
    np.save(f"{prefix}_raw_imp.npy", raw_imp)
    np.save(f"{prefix}_prot_gen.npy", prot_gen)
    np.save(f"{prefix}_prot_imp.npy", prot_imp)
    np.save(f"{prefix}_stolen_imp.npy", stolen_imp)

    print(f"Success! Saved scores to {SCORES_DIR}")
    print(f"Genuine Pairs: {len(raw_gen):,} | Impostor Pairs: {len(raw_imp):,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Embeddings & Bio-Hash Scores")
    parser.add_argument("--dataset", type=str, required=True, choices=["casia", "fvc2000", "fvc2004", "cmbd"])
    args = parser.parse_args()
    main(args.dataset)