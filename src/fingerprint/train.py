"""
Unified Training Script for Fingerprint Recognition with Differentiable Quantization
Supports: CASIA and FVC2000 datasets
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
    FVC2004_TRAIN_CSV,
    FVC2004_VAL_CSV,
    SAVED_MODELS_DIR,
    TENSORBOARD_DIR,
)
from src.Dataset_Loader import FingerprintDataset
from src.utils.logger import get_logger
from src.utils.sampler import PKSampler

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Updated Architecture Config
EMBEDDING_DIM = 512
LAMBDA_ENTROPY = 0.1  # Weight coefficient for Entropy-Aware Loss


# --------------------------
# Model Architecture
# --------------------------
class DifferentiableQuantization(nn.Module):
    """
    Applies Straight-Through Estimator (STE) for differentiable binary quantization.
    Forward pass: Rounds to 0 or 1.
    Backward pass: Treats rounding as identity function to allow gradient flow.
    """
    def forward(self, x):
        # x is assumed to be in range [0, 1] (e.g., after Sigmoid)
        quantized = torch.round(x)
        # STE: Output quantized values during forward, but bypass rounding during backward
        return x + (quantized - x).detach()


class FingerprintEmbeddingNet(nn.Module):
    """ResNet50 backbone with integrated Fake Quantization for binary embeddings."""

    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.model = models.resnet50(weights=weights)
        
        # Replace final FC layer to output 512 dimensions
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)
        
        # Quantization components
        self.activation = nn.Sigmoid()  # Map features to [0, 1] probabilities
        self.quantizer = DifferentiableQuantization()

    def forward(self, x):
        features = self.model(x)
        
        # Get continuous probabilities (needed for entropy loss)
        continuous_probs = self.activation(features)
        
        # Get quantized bits (0s and 1s) via STE
        quantized_bits = self.quantizer(continuous_probs)
        
        # Note: ArcMarginProduct handles the L2 normalization for the hypersphere internally
        return quantized_bits, continuous_probs


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
        # Internally normalizes the input bits to the hypersphere
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
# Loss Functions
# --------------------------
def calculate_entropy_loss(continuous_probs, eps=1e-7):
    """
    Calculates Min-Entropy Loss to ensure 512 bits are uniformly distributed.
    Maximizing this prevents the model from mapping all templates to all 0s or all 1s.
    """
    # Calculate the mean probability of each bit being 1 across the batch
    # Shape: [512]
    p = continuous_probs.mean(dim=0)
    
    # Calculate Shannon Entropy for the Bernoulli distribution of each bit: -sum(p * log(p))
    entropy = - (p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
    
    # Return the mean entropy across all 512 dimensions
    return entropy.mean()


# --------------------------
# Training Functions
# --------------------------
def train_one_epoch(backbone, margin_head, dataloader, optimizer, criterion_arcface, current_lambda, device, scaler):
    """Train for one epoch with Annealed Multi-Term Loss."""
    backbone.train()
    margin_head.train()
    total_loss, total_arcface_loss, total_entropy_loss = 0, 0, 0
    correct, total_samples = 0, 0

    loop = tqdm(dataloader, desc="Training", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # Cast the forward pass and loss computation to FP16
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            embeddings, continuous_probs = backbone(imgs)
            logits = margin_head(embeddings, labels)
            
            loss_arcface = criterion_arcface(logits, labels)
            loss_entropy = calculate_entropy_loss(continuous_probs)
            loss = loss_arcface - (current_lambda * loss_entropy)

        # Scale gradients and call backward
        scaler.scale(loss).backward()
        
        # Step optimizer and update scaler
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_arcface_loss += loss_arcface.item()
        total_entropy_loss += loss_entropy.item()
        
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        loop.set_postfix(
            loss=loss.item(), 
            arc_loss=loss_arcface.item(), 
            ent_loss=loss_entropy.item()
        )

    return (total_loss / len(dataloader), correct / total_samples, 
            total_arcface_loss / len(dataloader), total_entropy_loss / len(dataloader))


@torch.no_grad()
def validate(backbone, margin_head, dataloader, criterion_arcface, current_lambda, device):
    """Validate the model and compute Bit Error Rate."""
    backbone.eval()
    margin_head.eval()
    total_loss = 0
    correct, total_samples = 0, 0
    
    batch_bers = []

    loop = tqdm(dataloader, desc="Validation", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        embeddings, continuous_probs = backbone(imgs)
        logits = margin_head(embeddings, labels)
        
        loss_arcface = criterion_arcface(logits, labels)
        loss_entropy = calculate_entropy_loss(continuous_probs)
        loss = loss_arcface - (current_lambda * loss_entropy)

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        # Calculate BER for this batch
        batch_ber = compute_batch_ber(embeddings, labels)
        if batch_ber is not None:
            batch_bers.append(batch_ber)

        loop.set_postfix(loss=loss.item())

    # Average BER across all valid batches
    avg_ber = sum(batch_bers) / len(batch_bers) if batch_bers else float('inf')

    return total_loss / len(dataloader), correct / total_samples, avg_ber
# --------------------------
# Research Utilities & Metrics
# --------------------------
def compute_batch_ber(quantized_embeddings, labels):
    """
    Computes the intra-class Bit Error Rate (BER) for a batch.
    Measures the normalized Hamming distance between templates of the same identity.
    """
    B, dim = quantized_embeddings.shape
    
    # Create a mask for positive pairs (same identity), excluding self-comparisons
    label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))
    label_mask.fill_diagonal_(False)
    
    # If no two images in this batch belong to the same person, we can't compute BER
    if not label_mask.any():
        return None 
        
    # Pairwise absolute difference acts as XOR for strict 0/1 bits
    diff = torch.abs(quantized_embeddings.unsqueeze(1) - quantized_embeddings.unsqueeze(0))
    hamming_distances = diff.sum(dim=-1) 
    
    # Calculate BER (Hamming / 512) for valid pairs
    ber_matrix = hamming_distances / dim
    valid_ber = ber_matrix[label_mask]
    
    return valid_ber.mean().item()


def get_annealed_lambda(epoch, max_epochs, max_lambda=0.1, warmup_epochs=10):
    """
    Linear warmup for Entropy Loss weight to allow identity feature learning 
    before enforcing strict uniform bit distribution.
    """
    if epoch <= warmup_epochs:
        return max_lambda * (epoch / warmup_epochs)
    return max_lambda


def generate_orthogonal_projection_matrix(dim=512, device="cpu"):
    """
    Generates a chaotic but orthogonal projection matrix using QR decomposition 
    (Gram-Schmidt process). This ensures the projection is distance-preserving.
    """
    random_matrix = torch.randn(dim, dim, device=device)
    # Q is the orthogonal matrix, R is upper triangular
    q, r = torch.linalg.qr(random_matrix)
    
    # Standardize the signs to ensure a uniform distribution
    d = torch.diag(r)
    ph = d.sign()
    q *= ph
    return q

# --------------------------
# Dataset Configuration
# --------------------------
DATASET_CONFIG = {
    "casia": {
        "train_csv": CASIA_TRAIN_CSV,
        "val_csv": CASIA_VAL_CSV,
        "batch_size": 64,
        "epochs": 50,
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
    "fvc2004": {
        "train_csv": FVC2004_TRAIN_CSV,
        "val_csv": FVC2004_VAL_CSV,
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 1e-4,
        "num_classes": 100,  # FVC2004 has 100 fingers
        "model_name": "fvc2004_arcface_model.pth",
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
    # ... [Keep setup, dataloader creation, and resuming logic exactly the same] ...
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
        unique_ids = len(set(label for _, label in train_dataset.samples))
        config["num_classes"] = unique_ids
        logger.info(f"Auto-detected {unique_ids} unique classes")

    # Create dataloaders
    # Use P=16, K=4 for a Batch Size of 64
    # Use P=32, K=4 for a Batch Size of 128 (Try this first on the 3060)
    pk_sampler = PKSampler(train_dataset, p_classes=32, k_samples=4)

    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=pk_sampler, # Replaces batch_size and shuffle
        num_workers=2,            # Keeps your 10GB RAM from crashing
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=2, pin_memory=True,
    )

    logger.info(
        f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
    )
    logger.info(f"Number of Classes: {config['num_classes']}")
    
    # Initialize model
    backbone = FingerprintEmbeddingNet(EMBEDDING_DIM).to(DEVICE)
    margin_head = ArcMarginProduct(EMBEDDING_DIM, config["num_classes"]).to(DEVICE)
    
    # Load checkpoint if resuming
    start_epoch = 1

    if resume_from and os.path.exists(resume_from):
            logger.info(f"Resuming from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=DEVICE)
            backbone.load_state_dict(checkpoint["backbone"])
            margin_head.load_state_dict(checkpoint["margin_head"])
            start_epoch = checkpoint.get("epoch", 1) + 1
            
            # Reload metric states so your saving logic doesn't reset
            best_val_loss = checkpoint.get("val_loss", float("inf"))
            best_val_ber = checkpoint.get("val_ber", float("inf"))
    criterion_arcface = nn.CrossEntropyLoss()
    
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

    run_name = f"{dataset_name}_Quantized_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(TENSORBOARD_DIR, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    best_val_loss = float("inf")
    best_val_ber = float("inf")

    # Initialize Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')

    # Training loop
    for epoch in range(start_epoch, config["epochs"] + 1):
            
        # 1. Calculate Annealed Lambda for this epoch
        current_lambda = get_annealed_lambda(epoch, config["epochs"], max_lambda=0.1, warmup_epochs=10)
        
        # 2. Pass current_lambda to train and val functions
        train_loss, train_acc, arc_loss, ent_loss = train_one_epoch(
            backbone, margin_head, train_loader, optimizer, criterion_arcface, current_lambda, DEVICE, scaler
        )
        val_loss, val_acc, val_ber = validate(
            backbone, margin_head, val_loader, criterion_arcface, current_lambda, DEVICE
        )

        scheduler.step(val_loss)

        # 3. Log Expanded Metrics
        writer.add_scalar("Loss/Total_Train", train_loss, epoch)
        writer.add_scalar("Loss/ArcFace_Train", arc_loss, epoch)
        writer.add_scalar("Loss/Entropy_Train", ent_loss, epoch)
        writer.add_scalar("Loss/Total_Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("Metrics/Validation_BER", val_ber, epoch) # Track BER
        writer.add_scalar("Hyperparameters/Lambda_Entropy", current_lambda, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)
        
        # 4. Update Logger text (Fixes the 'inf' formatting crash)
        ber_str = f"{val_ber:.2%}" if val_ber != float('inf') else "N/A (No Pairs)"

        logger.info(
            f"Epoch {epoch}/{config['epochs']} | "
            f"BER: {ber_str} | "
            f"Train Loss: {train_loss:.4f} (Arc: {arc_loss:.4f}) Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}"
        )

        # 5. Strategic Refinement: Save on Lowest BER or Lowest Val Loss
        is_best_ber = (val_ber != float('inf') and val_ber < best_val_ber)
        is_best_loss = (val_loss < best_val_loss)

        if is_best_loss or is_best_ber:
            # Update the trackers
            if is_best_loss:
                best_val_loss = val_loss
            if is_best_ber:
                best_val_ber = val_ber
                
            os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
            save_path = os.path.join(SAVED_MODELS_DIR, config["model_name"])
            
            # Figure out what triggered the save for the logs
            save_reason = "Loss & BER" if (is_best_loss and is_best_ber) else ("Val Loss" if is_best_loss else "BER")
            
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "margin_head": margin_head.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_ber": val_ber, # Save the BER state too
                },
                save_path,
            )
            logger.info(f"  --> Checkpoint saved (Trigger: {save_reason} | Val Loss: {val_loss:.4f}, BER: {ber_str})")

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
        choices=["casia", "fvc2000", "fvc2004"],
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
