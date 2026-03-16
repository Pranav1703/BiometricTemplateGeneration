import torch
import torch.nn.functional as F
import numpy as np
import hashlib
import os

from src.fingerprint.train import FingerprintEmbeddingNet
from src.fingerprint.preprocess_fingerprint import preprocess_fingerprint
from src.config import SAVED_MODELS_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 512  

# ---------------------------
# 1. Load trained backbone
# ---------------------------
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "fingerprint_arcface_model.pth")
backbone = FingerprintEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE,  weights_only=True)
backbone.load_state_dict(checkpoint["backbone"])
backbone.eval()


def get_chaotic_sequence(seed_str: str, length: int, r: float = 3.99) -> torch.Tensor:
    """
    Generates a chaotic sequence based on a user key.
    r must be between 3.57 and 4.0 for chaotic behavior.
    """
    # 1. Map string key to a float between 0 and 1
    hash_val = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16)
    x = (hash_val % 10**10) / 10**10  # Initial state x0
    
    sequence = torch.zeros(length, device=DEVICE)
    for i in range(length):
        x = r * x * (1 - x)
        sequence[i] = x
        
    return sequence

def generate_chaotic_projection(embedding: torch.Tensor, user_key: str) -> torch.Tensor:
    # Generate enough chaotic numbers for a DIM x DIM matrix
    num_elements = EMBEDDING_DIM * EMBEDDING_DIM
    chaotic_seq = get_chaotic_sequence(user_key, num_elements)
    
    # Reshape into matrix R
    R_chaotic = chaotic_seq.view(EMBEDDING_DIM, EMBEDDING_DIM)
    
    # Orthonormalize R (Crucial for Bio-Hashing to preserve distance)
    # Using QR decomposition to make R an orthogonal matrix
    Q, _ = torch.linalg.qr(R_chaotic)
    
    return Q # (512, 512)
def apply_bio_hash(projected_embedding: torch.Tensor):
    """
    Transforms the continuous projected vector into a binary template.
    0.0 is the standard threshold for normalized embeddings.
    """
    # Thresholding: 1 if > 0, else 0
    binary_template = (projected_embedding > 0).float()
    return binary_template

def calculate_hamming_similarity(template1: torch.Tensor, template2: torch.Tensor):
    """
    Returns the percentage of matching bits. 
    1.0 means identical, 0.0 means completely opposite.
    """
    # XOR gives 1 where bits differ, so we count zeros
    matching_bits = torch.sum(template1 == template2).item()
    return matching_bits / template1.numel()

@torch.no_grad()
def get_protected_template(img_path: str, user_key: str, R_fixed: torch.Tensor):
    # 1. Feature Extraction
    img_tensor = preprocess_fingerprint(img_path, train=False).unsqueeze(0).to(DEVICE)
    embedding = backbone(img_tensor).squeeze(0) 

    # 2. Dynamic Projection using the passed R_fixed
    # Ensure R_fixed is on the same device as embedding
    projected = torch.matmul(R_fixed.to(DEVICE), embedding)

    # 3. Bio-Hashing (Binarization)
    binary_template = apply_bio_hash(projected)

    return embedding.cpu(), binary_template.cpu()


# Optimized Workflow
if __name__ == "__main__":
    img1_path = "D:/code/Projects/biometric-template-gen/data/CASIA-dataset/000/L/000_L0_0.bmp"
    img2_path = "D:/code/Projects/biometric-template-gen/data/CASIA-dataset/000/L/000_L0_0.bmp"
    user_key = "user_000_session_1"

    # 1. PRE-GENERATE R (Do this once per user/session)
    # This saves massive compute time if processing multiple images
    R_fixed = generate_chaotic_projection(None, user_key) 

    # 2. Get Templates
    # (Update get_protected_template to accept R_fixed instead of generating it inside)
    emb1_raw, emb1_prot = get_protected_template(img1_path, user_key, R_fixed)
    emb2_raw, emb2_prot = get_protected_template(img2_path, user_key, R_fixed)

    # 3. Compare using Hamming Similarity
    hamming_sim = calculate_hamming_similarity(emb1_prot, emb2_prot)
    
    print(f"Raw Cosine Similarity: {F.cosine_similarity(emb1_raw.unsqueeze(0), emb2_raw.unsqueeze(0)).item():.3f}")
    print(f"Protected Hamming Similarity: {hamming_sim:.3f}")