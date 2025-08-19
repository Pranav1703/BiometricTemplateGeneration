import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision.models import ResNet18_Weights
from pathlib import Path
from tqdm import tqdm
from src.config import FINGERPRINT_VAL_CSV, FINGERPRINT_TRAIN_CSV, SAVED_MODELS_DIR, TENSORBOARD_DIR
from src.utils.Dataset_Loader import FingerprintDataset  # Your Dataset class
from src.utils.logger import get_logger
from datetime import datetime
import numpy as np
import os

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
    # Initializing Loger
    logger = get_logger("train")

    # Example usage
    # logger.debug(msg) → Detailed debug info
    # logger.info(msg) → General info (e.g. loss values, progress)
    # logger.warning(msg) → Something unexpected, but not breaking
    # logger.error(msg) → Serious issue
    # logger.critical(msg) → Program might crash

    train_csv = FINGERPRINT_TRAIN_CSV
    val_csv = FINGERPRINT_VAL_CSV

    print(f"Train CSV path: {train_csv}")
    print(f"Val CSV path: {val_csv}")

    train_dataset = FingerprintDataset(str(train_csv))
    val_dataset = FingerprintDataset(str(val_csv))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = EmbeddingNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=MARGIN)

    # Create unique run directory
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(TENSORBOARD_DIR, run_name)

    # Initialize the Tensorboard Writier
    writer = SummaryWriter(log_dir=log_dir)

    logger.info("Tensorboard is initialized run this command in another termenal to see the live graph tensorboard --logdir artifacts\\plots\\tensorboard")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)

        # Log scalar values
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)

        logger.info(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    writer.close()

    try:
        output_dir = SAVED_MODELS_DIR
        os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist
        torch.save(model.state_dict(), os.path.join(output_dir, "fingerprint_embedding_model.pth"))
        logger.info("Model Saved successfully")
    except:
        logger.error("Model can't save")


if __name__ == "__main__":
    main()
