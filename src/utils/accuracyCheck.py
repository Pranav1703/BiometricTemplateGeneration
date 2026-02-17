import torch
import torch.nn.functional as F

# Load your embeddings & labels
data = torch.load("artifacts/embeddings/test_embeddings.pt", weights_only=True)
embeddings = data["embeddings"]  # [N, D]
labels = data["labels"]          # [N]

# Normalize for cosine similarity
embeddings = F.normalize(embeddings, dim=1)

# Compute similarity matrix
similarity_matrix = torch.matmul(embeddings, embeddings.T)  # [N, N]

# For each sample, find the index of the most similar embedding (excluding itself)
N = embeddings.size(0)
preds = []
for i in range(N):
    sim = similarity_matrix[i].clone()
    sim[i] = -1.0  # exclude itself
    pred_idx = torch.argmax(sim)
    preds.append(labels[pred_idx])

preds = torch.tensor(preds)

# Accuracy
accuracy = (preds == labels).float().mean().item()
print(f"Top-1 Accuracy: {accuracy*100:.2f}%")

#This is a verification-style evaluation (1-vs-all).