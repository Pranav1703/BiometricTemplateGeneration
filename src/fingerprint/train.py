"""
Unified Training Script for Fingerprint Recognition
Supports: CASIA and FVC2000 datasets
Usage: python -m src.fingerprint.train --dataset casia
       python -m src.fingerprint.train --dataset fvc2000
"""

import argparse
import os
import math
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

from src.config import (
    CASIA_TRAIN_CSV,
    CASIA_VAL_CSV,
    FVC2000_TRAIN_CSV,
    FVC2000_VAL_CSV,
    SAVED_MODELS_DIR,
    TENSORBOARD_DIR,
    EMBEDDING_DIM,
)
from src.utils.Dataset_Loader import FingerprintDataset
from src.utils.logger import get_logger

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------
# Model Architecture
# --------------------------
class FingerprintEmbeddingNet(nn.Module):
    """ResNet50 backbone for fingerprint embedding extraction."""

    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.model(x)
        return nn.functional.normalize(x, p=2, dim=1)


class ArcMarginProduct(nn.Module):
    """ArcFace head for large margin classification."""

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
        cosine = nn.functional.linear(
            nn.functional.normalize(input), nn.functional.normalize(self.weight)
        )
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


# --------------------------
# Training Functions
# --------------------------
def train_one_epoch(backbone, margin_head, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    backbone.train()
    margin_head.train()
    total_loss = 0
    correct = 0
    total_samples = 0

    loop = tqdm(dataloader, desc="Training", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        embeddings = backbone(imgs)
        logits = margin_head(embeddings, labels)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    acc = correct / total_samples
    return avg_loss, acc


@torch.no_grad()
def validate(backbone, margin_head, dataloader, criterion, device):
    """Validate the model."""
    backbone.eval()
    margin_head.eval()
    total_loss = 0
    correct = 0
    total_samples = 0

    loop = tqdm(dataloader, desc="Validation", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        embeddings = backbone(imgs)
        logits = margin_head(embeddings, labels)
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader), correct / total_samples


# --------------------------
# Dataset Configuration
# --------------------------
DATASET_CONFIG = {
    "casia": {
        "train_csv": CASIA_TRAIN_CSV,
        "val_csv": CASIA_VAL_CSV,
        "batch_size": 16,
        "epochs": 20,
        "learning_rate": 1e-4,
        "num_classes": None,  # Will be auto-detected
        "model_name": "casia_arcface_model.pth",
    },
    "fvc2000": {
        "train_csv": FVC2000_TRAIN_CSV,
        "val_csv": FVC2000_VAL_CSV,
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 1e-4,
        "num_classes": 100,  # FVC2000 has 100 fingers
        "model_name": "fvc2000_arcface_model.pth",
    },
}


def get_dataset_config(dataset_name):
    """Get configuration for specified dataset."""
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Choose from: {list(DATASET_CONFIG.keys())}"
        )
    return DATASET_CONFIG[dataset_name]


# --------------------------
# Main Training Function
# --------------------------
def train(dataset_name, resume_from=None):
    """Train model on specified dataset.

    Args:
        dataset_name: Name of dataset ('casia' or 'fvc2000')
        resume_from: Path to checkpoint to resume training from (optional)
    """
    logger = get_logger(f"train_{dataset_name}")
    config = get_dataset_config(dataset_name)

    logger.info(f"Starting training on {dataset_name} dataset")
    logger.info(f"Device: {DEVICE}")

    # Load datasets
    train_dataset = FingerprintDataset(str(config["train_csv"]), train=True)
    val_dataset = FingerprintDataset(str(config["val_csv"]), train=False)

    if len(train_dataset) == 0:
        logger.error("Train dataset is empty!")
        return

    # Determine number of classes
    if config["num_classes"] is None:
        # Auto-detect for CASIA
        unique_ids = len(set(label for _, label in train_dataset.samples))
        config["num_classes"] = unique_ids
        logger.info(f"Auto-detected {unique_ids} unique classes")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(
        f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
    )
    logger.info(f"Number of Classes: {config['num_classes']}")

    # Initialize model
    backbone = FingerprintEmbeddingNet().to(DEVICE)
    margin_head = ArcMarginProduct(EMBEDDING_DIM, config["num_classes"]).to(DEVICE)

    # Load checkpoint if resuming
    start_epoch = 1
    if resume_from and os.path.exists(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=DEVICE)
        backbone.load_state_dict(checkpoint["backbone"])
        margin_head.load_state_dict(checkpoint["margin_head"])
        start_epoch = checkpoint.get("epoch", 1) + 1

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [
            {"params": backbone.parameters(), "lr": config["learning_rate"]},
            {"params": margin_head.parameters(), "lr": config["learning_rate"]},
        ],
        weight_decay=1e-4,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    # Setup logging
    run_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(TENSORBOARD_DIR, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logs: {log_dir}")
    logger.info(f"Monitor with: tensorboard --logdir={TENSORBOARD_DIR}")

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(start_epoch, config["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            backbone, margin_head, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_acc = validate(
            backbone, margin_head, val_loader, criterion, DEVICE
        )

        scheduler.step(val_loss)

        # Log metrics
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            f"Epoch {epoch}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
            save_path = os.path.join(SAVED_MODELS_DIR, config["model_name"])
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "margin_head": margin_head.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                save_path,
            )
            logger.info(f"  --> Best model saved (Val Loss: {val_loss:.4f})")

    writer.close()
    logger.info("Training complete!")
    logger.info(
        f"Best model saved at: {os.path.join(SAVED_MODELS_DIR, config['model_name'])}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train fingerprint embedding model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.fingerprint.train --dataset casia
  python -m src.fingerprint.train --dataset fvc2000
  python -m src.fingerprint.train --dataset casia --resume artifacts/models/casia_arcface_model.pth
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["casia", "fvc2000"],
        help="Dataset to train on (casia or fvc2000)",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    args = parser.parse_args()

    train(args.dataset, resume_from=args.resume)


if __name__ == "__main__":
    main()
