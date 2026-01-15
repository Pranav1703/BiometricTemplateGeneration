import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from src.config import FVC2000_TRAIN_CSV, FVC2000_VAL_CSV, SAVED_MODELS_DIR, TENSORBOARD_DIR
from src.Dataset_Loader import FingerprintDataset
from src.utils.logger import get_logger

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 512
BATCH_SIZE = 32         # Increased to 32 (better for ArcFace stability)
EPOCHS = 50             # Increased to 50 (Transfer learning needs time)
LEARNING_RATE = 1e-4
NUM_CLASSES = 100       # CORRECT: FVC DB1 has 100 fingers

# --------------------------
# Backbone: ResNet50 -> Embedding
# --------------------------
class FingerprintEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        # Use ImageNet weights for transfer learning
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.model = models.resnet50(weights=weights)
        
        # Replace the final FC layer
        self.model.fc = nn.Linear(self.model.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.model(x)
        # L2 Normalize embeddings to the unit hypersphere
        return nn.functional.normalize(x, p=2, dim=1)

# --------------------------
# ArcFace Head
# --------------------------
class ArcMarginProduct(nn.Module):
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
        # 1. Cosine similarity
        cosine = nn.functional.linear(nn.functional.normalize(input),
                                      nn.functional.normalize(self.weight))
        # 2. Add margin
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 3. Create One-hot and apply margin only to the correct class
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

# --------------------------
# Training Loops
# --------------------------
def train_one_epoch(backbone, margin_head, dataloader, optimizer, criterion):
    backbone.train()
    margin_head.train()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    loop = tqdm(dataloader, desc="Training", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        embeddings = backbone(imgs)
        logits = margin_head(embeddings, labels)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Calculate rough accuracy
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        loop.set_postfix(loss=loss.item())
        
    avg_loss = total_loss / len(dataloader)
    acc = correct / total_samples
    return avg_loss, acc

@torch.no_grad()
def validate(backbone, margin_head, dataloader, criterion):
    backbone.eval()
    margin_head.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    loop = tqdm(dataloader, desc="Validation", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        
        embeddings = backbone(imgs)
        logits = margin_head(embeddings, labels)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader), correct / total_samples

def main():
    logger = get_logger("train")
    
    # 1. Setup Data
    train_dataset = FingerprintDataset(str(FVC2000_TRAIN_CSV), train=True)
    val_dataset = FingerprintDataset(str(FVC2000_VAL_CSV), train=False)
    
    # Check data size
    if len(train_dataset) == 0:
        logger.error("Train dataset is empty!")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    logger.info(f"Training on {len(train_dataset)} images, Validating on {len(val_dataset)} images.")
    logger.info(f"Number of Classes: {NUM_CLASSES}")

    # 2. Setup Model
    backbone = FingerprintEmbeddingNet().to(DEVICE)
    margin_head = ArcMarginProduct(EMBEDDING_DIM, NUM_CLASSES).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    
    # Separate parameters for backbone and head (optional, but good practice)
    optimizer = optim.Adam([
        {'params': backbone.parameters(), 'lr': LEARNING_RATE},
        {'params': margin_head.parameters(), 'lr': LEARNING_RATE}
    ], weight_decay=1e-4)

    # Scheduler: Reduce LR when loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 3. Logging
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(TENSORBOARD_DIR, run_name)
    writer = SummaryWriter(log_dir=log_dir)
    logger.info(f"TensorBoard logs will be saved to: {log_dir}")
    logger.info(f"run this cmd to monitor: tensorboard --logdir={TENSORBOARD_DIR}")

    best_val_loss = float('inf')

    # 4. Training Loop
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(backbone, margin_head, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(backbone, margin_head, val_loader, criterion)
        
        # Step the scheduler
        scheduler.step(val_loss)

        # Logging
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)
        
        logger.info(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
            torch.save({
                "backbone": backbone.state_dict(),
                "margin_head": margin_head.state_dict(),
                "epoch": epoch
            }, os.path.join(SAVED_MODELS_DIR, "best_fvc_model.pth"))
            logger.info(f"  --> New best model saved (Val Loss: {val_loss:.4f})")

    writer.close()
    logger.info("Training Complete.")

if __name__ == "__main__":
    main()