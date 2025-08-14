import torch
import torch.nn.functional as F
from preprocess import preprocess_fingerprint
from train import EmbeddingNet  # ✅ Import your trained model architecture

# ====== CONFIG ======
MODEL_PATH = "output/fingerprint_embedding_model.pth"
IMG1_PATH = "data/IRIS_and_FINGERPRINT_DATASET/1/Fingerprint/1__M_Left_middle_finger.BMP"
IMG2_PATH = "data/IRIS_and_FINGERPRINT_DATASET/1/Fingerprint/1__M_Right_index_finger.BMP"

# ====== 1. Load the model ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmbeddingNet()  # create model
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ====== 2. Preprocess images ======
img1_tensor = preprocess_fingerprint(IMG1_PATH, train=False).unsqueeze(0).to(device)
img2_tensor = preprocess_fingerprint(IMG2_PATH, train=False).unsqueeze(0).to(device)

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
THRESHOLD = 0.8
if similarity >= THRESHOLD:
    print("✅ Likely same finger")
else:
    print("❌ Likely different fingers")
