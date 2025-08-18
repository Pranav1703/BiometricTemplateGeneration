import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from src.utils.Dataset_Loader import IrisDataset  # <-- use CSV-based IrisDataset
import numpy as np
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
import os
from pathlib import Path
from src.config import IRIS_TRAIN_CSV, IRIS_VAL_CSV


# ==================
# Config
# ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 256
BATCH_SIZE = 32
EPOCHS = 20
MARGIN = 1.0  # For triplet loss
LEARNING_RATE = 1e-4
IMG_SIZE = 224


# ==================
# Model
# ==================
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = models.resnet18(weights=weights)

        # Convert to 1-channel input instead of RGB
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.model(x)
        return nn.functional.normalize(x, p=2, dim=1)


# ==================
# Triplet mining utility
# ==================
def get_triplets(embeddings, labels):
    """
    Generate triplets for triplet loss: anchor, positive, negative.
    """
    triplets = []
    labels = labels.cpu().numpy()
    embeddings = embeddings.cpu().detach().numpy()

    for i in range(len(embeddings)):
        anchor_label = labels[i]

        pos_indices = np.where(labels == anchor_label)[0]
        neg_indices = np.where(labels != anchor_label)[0]

        # skip anchor itself for positive
        pos_indices = pos_indices[pos_indices != i]

        for pos_idx in pos_indices:
            for neg_idx in neg_indices:
                triplets.append((i, pos_idx, neg_idx))

    return triplets


# ==================
# Training & Validation
# ==================
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


# ==================
# Main
# ==================
def main():
    train_csv = IRIS_TRAIN_CSV
    val_csv = IRIS_VAL_CSV

    print(f"Train CSV path: {train_csv}")
    print(f"Val CSV path: {val_csv}")

    # Load datasets
    train_dataset = IrisDataset(str(train_csv), train=True)
    val_dataset = IrisDataset(str(val_csv), train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, optimizer, loss
    model = EmbeddingNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=MARGIN)

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)

        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Save model
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "iris_embedding_model.pth"))


if __name__ == "__main__":
    main()
