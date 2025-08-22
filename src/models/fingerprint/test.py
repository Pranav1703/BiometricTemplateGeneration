import torch
import torch.nn.functional as F
from src.preprocess.fingerprint import preprocess_fingerprint
from src.models.fingerprint.train import EmbeddingNet  # ✅ Import your trained model architecture
from src.config import FINGERPRINT_EX_1_0,FINGERPRINT_EX_1_1,SAVED_MODELS_DIR
import os

# ====== CONFIG ======
MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "fingerprint_embedding_model.pth")
IMG1_PATH = FINGERPRINT_EX_1_0
IMG2_PATH = FINGERPRINT_EX_1_1

# ====== 1. Load the model ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmbeddingNet()  # create model
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ====== 2. Preprocess images ======
img1_tensor = torch.tensor(preprocess_fingerprint(IMG1_PATH, train=False)).unsqueeze(0).to(device)
img2_tensor = torch.tensor(preprocess_fingerprint(IMG2_PATH, train=False)).unsqueeze(0).to(device)

# ====== 3. Get embeddings ======
with torch.no_grad():
    emb1 = model(img1_tensor)
    emb2 = model(img2_tensor)

# ====== 4. Log embeddings ======
print("\n📌 Embedding for Image 1:\n", emb1.cpu().numpy())
print("\n📌 Embedding for Image 2:\n", emb2.cpu().numpy())

# ====== 5. Compute similarity ======
similarity = F.cosine_similarity(emb1, emb2).item()
print(f"\n🔍 Cosine Similarity: {similarity:.4f}")

# ====== 6. Simple decision ======
THRESHOLD = 0.99
if similarity >= THRESHOLD:
    print("✅ Likely same finger")
else:
    print("❌ Likely different fingers")
