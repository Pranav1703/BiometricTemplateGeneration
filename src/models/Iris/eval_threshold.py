import csv
import torch
import torch.nn.functional as F
from itertools import combinations
from preprocess.iris import preprocess_image   # <- your iris preprocessing
from train import EmbeddingNet           # Import trained iris architecture
from pathlib import Path
import numpy as np

# ====== CONFIG ======
MODEL_PATH = "output/iris_embedding_model.pth"   # <- iris model
VAL_CSV = Path("labels/iris_val.csv")            # <- iris validation csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Load model ======
model = EmbeddingNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# ====== Load validation data ======
samples = []
with open(VAL_CSV, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        samples.append((row['filepath'], int(row['person_id'])))

# ====== Cache embeddings ======
embeddings_cache = {}
for img_path, label in samples:
    img_tensor = preprocess_image(img_path).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(img_tensor).cpu()
    embeddings_cache[img_path] = (emb, label)

# ====== Generate pairs ======
pairs = []
for (img1, label1), (img2, label2) in combinations(samples, 2):
    emb1, _ = embeddings_cache[img1]
    emb2, _ = embeddings_cache[img2]
    sim = F.cosine_similarity(emb1, emb2).item()
    is_same = 1 if label1 == label2 else 0
    pairs.append((sim, is_same))

# ====== Find best threshold ======
best_acc = 0
best_thresh = 0
sims = np.array([p[0] for p in pairs])
labels = np.array([p[1] for p in pairs])

for thresh in np.linspace(0, 1, 101):  # 0.00 → 1.00 in steps of 0.01
    preds = (sims >= thresh).astype(int)
    acc = (preds == labels).mean()
    if acc > best_acc:
        best_acc = acc
        best_thresh = thresh

print(f"✅ Best Threshold (Iris): {best_thresh:.2f} | Accuracy: {best_acc*100:.2f}%")
