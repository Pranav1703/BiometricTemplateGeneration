import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.Dataset_Loader import FingerprintDataset
from src.config import FINGERPRINT_VAL_CSV, SAVED_MODELS_DIR
from src.fingerprint.train import FingerprintEmbeddingNet, EMBEDDING_DIM, DEVICE

Model_path = os.path.join(SAVED_MODELS_DIR, "fingerprint_arcface_model.pth")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset & dataloader
    test_dataset = FingerprintDataset(FINGERPRINT_VAL_CSV, train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # load backbone
    backbone = FingerprintEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(device)
    checkpoint = torch.load(Model_path, map_location=device)
    backbone.load_state_dict(checkpoint["backbone"])
    backbone.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Extracting embeddings"):
            imgs = imgs.to(device)
            embeddings = backbone(imgs)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    os.makedirs("artifacts/embeddings", exist_ok=True)
    torch.save({
        "embeddings": all_embeddings,
        "labels": all_labels
    }, "artifacts/embeddings/test_embeddings.pt")

    print("Saved embeddings to artifacts/embeddings/test_embeddings.pt")

if __name__ == "__main__":
    main()
