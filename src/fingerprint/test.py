import torch
import torch.nn.functional as F

# load saved embeddings
data = torch.load("artifacts/embeddings/test_embeddings.pt")
embeddings = data["embeddings"]
labels = data["labels"]

print(f"Loaded embeddings: {embeddings.shape}, labels: {labels.shape}")

# Example: cosine similarity between first two
cos = F.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[0].unsqueeze(0))
print("Cosine similarity between sample 0 and 1:", cos.item())

# Example: compute same-ID vs different-ID mean cosine similarity
# same_id = []
# diff_id = []
# for i in range(100):  # sample a subset to keep it fast
#     for j in range(i+1, i+10):
#         if labels[i] == labels[j]:
#             same_id.append(F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item())
#         else:
#             diff_id.append(F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item())

# print(f"Mean cosine similarity (same ID): {sum(same_id)/len(same_id):.3f}")
# print(f"Mean cosine similarity (different ID): {sum(diff_id)/len(diff_id):.3f}")


'''
To confirm that your protection still preserves identity information:

Test intra-class similarity (same finger, different samples).

Test inter-class similarity (different fingers).
FAR, FRR

Plot ROC / DET or EER curves to see if genuine vs imposter distributions are separable even after projection.

If genuine pairs (like yours) consistently score ~0.84 and imposters <0.5,
then protection works and recognition accuracy remains good.
'''