import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from dataset import FingerprintDataset  # Your Dataset class
import numpy as np
from pathlib import Path
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 256
BATCH_SIZE = 32
EPOCHS = 20
MARGIN = 1.0  # For triplet loss
LEARNING_RATE = 1e-4

# Modified ResNet-18 for embeddings
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        # Use new weights argument instead of deprecated pretrained
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)
    
    def forward(self, x):
        x = self.model(x)
        return nn.functional.normalize(x, p=2, dim=1)

# Triplet mining utility (basic batch all triplets)
def get_triplets(embeddings, labels):
    """
    Generate triplets for triplet loss: anchor, positive, negative
    For simplicity, generate all valid triplets in batch.
    """
    triplets = []
    labels = labels.cpu().numpy()
    embeddings = embeddings.cpu().detach().numpy()
    
    for i in range(len(embeddings)):
        anchor_label = labels[i]
        anchor = embeddings[i]
        
        pos_indices = np.where(labels == anchor_label)[0]
        neg_indices = np.where(labels != anchor_label)[0]
        
        # skip anchor itself for positive
        pos_indices = pos_indices[pos_indices != i]
        
        for pos_idx in pos_indices:
            for neg_idx in neg_indices:
                triplets.append((i, pos_idx, neg_idx))
    return triplets

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc="Training", leave=False)
    for imgs, labels in loop:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        embeddings = model(imgs)
        triplets = get_triplets(embeddings, labels)
        if not triplets:
            continue

        anchor_idx, pos_idx, neg_idx = zip(*triplets)
        anchor = embeddings[list(anchor_idx)]
        positive = embeddings[list(pos_idx)]
        negative = embeddings[list(neg_idx)]

        loss = criterion(anchor, positive, negative)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    loop = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for imgs, labels in loop:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            embeddings = model(imgs)
            triplets = get_triplets(embeddings, labels)
            if not triplets:
                continue

            anchor_idx, pos_idx, neg_idx = zip(*triplets)
            anchor = embeddings[list(anchor_idx)]
            positive = embeddings[list(pos_idx)]
            negative = embeddings[list(neg_idx)]

            loss = criterion(anchor, positive, negative)
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
    avg_loss = total_loss / len(dataloader)
    return avg_loss



def main():
    SCRIPT_DIR = Path(__file__).parent.resolve()
    train_csv = SCRIPT_DIR.parent.parent / "labels" / "fingerprint_train.csv"
    val_csv = SCRIPT_DIR.parent.parent / "labels" / "fingerprint_val.csv"

    print(f"Train CSV path: {train_csv}")
    print(f"Val CSV path: {val_csv}")

    train_dataset = FingerprintDataset(str(train_csv))
    val_dataset = FingerprintDataset(str(val_csv))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = EmbeddingNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=MARGIN)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)


        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "fingerprint_embedding_model.pth")

if __name__ == "__main__":
    main()
