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
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "casia_arcface_model.pth")
backbone = FingerprintEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE,  weights_only=True)
backbone.load_state_dict(checkpoint["backbone"])
backbone.eval()


# ---------------------------
# 2. Dynamic Projection Function
# ---------------------------
def generate_dynamic_projection(embedding: torch.Tensor, user_key: str, alpha: float = 0.5) -> torch.Tensor:
    # 1. Key-based random matrix (Directly on GPU)
    seed = int(hashlib.sha256(user_key.encode()).hexdigest(), 16) % (2**32)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    # Generate R_key on GPU
    R_key = torch.randn((EMBEDDING_DIM, EMBEDDING_DIM), generator=generator, device=DEVICE)
    R_key = F.normalize(R_key, p=2, dim=1)

    # 2. Biometric-driven matrix (Directly on GPU)
    E_norm = F.normalize(embedding, p=2, dim=0)
    R_bio = torch.outer(E_norm, E_norm)
    R_bio = F.normalize(R_bio, p=2, dim=1)

    # 3. Hybrid Combine
    R_dyn = alpha * R_bio + (1 - alpha) * R_key
    R_dyn = F.normalize(R_dyn, p=2, dim=1)

    return R_dyn # stays on GPU


# ---------------------------
# 3. Embedding + Dynamic Protection
# ---------------------------
@torch.no_grad()
def get_embeddings(img_path: str, user_key: str):
    # Load and preprocess
    img_tensor = preprocess_fingerprint(img_path, train=False).unsqueeze(0).to(DEVICE)

    # Forward pass on GPU
    embedding = backbone(img_tensor).squeeze(0)  # stays on GPU (512,)

    # Generate dynamic projection directly on GPU
    R_dyn = generate_dynamic_projection(embedding, user_key=user_key, alpha=0.6).to(DEVICE)

    # Compute protected embedding on GPU
    protected = torch.matmul(R_dyn, embedding)

    # Move results to CPU only if needed
    return embedding.cpu(), protected.cpu()

# ---------------------------
# 4. Example usage
# ---------------------------
if __name__ == "__main__":
    img1_path = "D:/code/Projects/biometric-template-gen/data/CASIA-dataset/000/L/000_L0_0.bmp"
    img2_path = "D:/code/Projects/biometric-template-gen/data/CASIA-dataset/002/R/002_R1_4.bmp"
    
    # Each user/session can have a unique key
    user_key = "user_000_session_1"

    emb1_raw, emb1_protected = get_embeddings(img1_path, user_key)
    emb2_raw, emb2_protected = get_embeddings(img2_path, user_key)

    print("Raw embedding1 (first 10):", emb1_raw[:10].numpy())
    print("Protected embedding1 (first 10):", emb1_protected[:10].numpy())

    cos_sim_raw = F.cosine_similarity(emb1_raw.unsqueeze(0), emb2_raw.unsqueeze(0)).item()
    cos_sim_prot = F.cosine_similarity(emb1_protected.unsqueeze(0), emb2_protected.unsqueeze(0)).item()

    # print(f"Raw cosine similarity between img1 and img2: {cos_sim_raw:.3f}")
    print(f"Protected cosine similarity img1 and img2: {cos_sim_prot:.3f}")
