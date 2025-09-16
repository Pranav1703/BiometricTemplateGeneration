import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math

from src.config import FINGERPRINT_TRAIN_CSV, FINGERPRINT_VAL_CSV, SAVED_MODELS_DIR, TENSORBOARD_DIR
from src.Dataset_Loader import FingerprintDataset
from src.utils.logger import get_logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 512
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 500  # <-- set to number of unique fingerprint IDs

# --------------------------
# Backbone: ResNet50 -> Embedding
# --------------------------
class FingerprintEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.model(x)
        # Normalize to unit hypersphere
        return nn.functional.normalize(x, p=2, dim=1)

# --------------------------
# ArcFace / ArcMarginProduct head
# --------------------------
class ArcMarginProduct(nn.Module):
    """
    Implements large margin arc distance:
    cos(theta + m)
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # input: (batch, embedding_dim)
        # label: (batch)
        cosine = nn.functional.linear(nn.functional.normalize(input),
                                      nn.functional.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        # combine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output  # logits
# --------------------------

def train_one_epoch(backbone, margin_head, dataloader, optimizer, criterion):
    backbone.train()
    margin_head.train()
    total_loss = 0
    loop = tqdm(dataloader, desc="Training", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        embeddings = backbone(imgs)
        logits = margin_head(embeddings, labels)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(backbone, margin_head, dataloader, criterion):
    backbone.eval()
    margin_head.eval()
    total_loss = 0
    loop = tqdm(dataloader, desc="Validation", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        embeddings = backbone(imgs)
        logits = margin_head(embeddings, labels)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

def main():
    logger = get_logger("train")

    # datasets
    train_dataset = FingerprintDataset(str(FINGERPRINT_TRAIN_CSV), train=True)
    val_dataset = FingerprintDataset(str(FINGERPRINT_VAL_CSV), train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # number of classes from dataset
    unique_ids = len(set(label for _, label in train_dataset.samples))
    logger.info(f"Detected {unique_ids} unique fingerprint IDs.")
    num_classes = unique_ids

    backbone = FingerprintEmbeddingNet().to(DEVICE)
    margin_head = ArcMarginProduct(EMBEDDING_DIM, num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(backbone.parameters()) + list(margin_head.parameters()),
                           lr=LEARNING_RATE, weight_decay=1e-4)

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(TENSORBOARD_DIR, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"Tensorboard initialized. Run: tensorboard --logdir {TENSORBOARD_DIR}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(backbone, margin_head, train_loader, optimizer, criterion)
        val_loss = validate(backbone, margin_head, val_loader, criterion)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        logger.info(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    writer.close()

    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    torch.save({
        "backbone": backbone.state_dict(),
        "margin_head": margin_head.state_dict()
    }, os.path.join(SAVED_MODELS_DIR, "fingerprint_arcface_model.pth"))
    logger.info("Model saved successfully.")

if __name__ == "__main__":
    main()
