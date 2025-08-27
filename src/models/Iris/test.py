import torch
import torch.nn.functional as F
from src.preprocess.iris import preprocess_image   # ✅ Import iris preprocessing
from src.models.Iris.train import IrisEmbeddingNet    # ✅ Import your iris model architecture
from src.config import IRIS_MODEL_PATH

# ====== CONFIG ======
MODEL_PATH = IRIS_MODEL_PATH
IMG1_PATH = "data/IRIS_and_FINGERPRINT_DATASET/1/left/aeval1.bmp"
IMG2_PATH = "data/IRIS_and_FINGERPRINT_DATASET/2/right/bryanr2.bmp"

# ====== 1. Load the model ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = IrisEmbeddingNet()  # create model
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ====== 2. Preprocess images ======
img1_tensor = preprocess_image(IMG1_PATH).unsqueeze(0).to(device)
img2_tensor = preprocess_image(IMG2_PATH).unsqueeze(0).to(device)

# ====== 3. Get embeddings ======
with torch.no_grad():
    emb1 = model(img1_tensor)
    emb2 = model(img2_tensor)

# ====== 4. Log embeddings ======
print("\n Embedding for Iris Image 1 Shape:", emb1.shape)
print(" Embedding for Iris Image 2 Shape:", emb2.shape)

print("\n Embedding for Iris Image 1:\n", emb1.cpu().numpy())
print("\n Embedding for Iris Image 2:\n", emb2.cpu().numpy())


# ====== 5. Compute similarity ======
similarity = F.cosine_similarity(emb1, emb2).item()
print(f"\n🔍 Cosine Similarity: {similarity:.4f}")

# ====== 6. Simple decision ======
THRESHOLD = 0.90   # 👈 Iris features may need a slightly lower threshold
if similarity >= THRESHOLD:
    print("✅ Likely same iris")
else:
    print("❌ Likely different iris")
