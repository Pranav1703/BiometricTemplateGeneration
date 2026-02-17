"""
Unified Testing Script for Fingerprint Recognition
Supports: CASIA and FVC2000 datasets
Usage: python src/fingerprint/test.py --dataset casia --mode evaluate
       python src/fingerprint/test.py --dataset fvc2000 --mode generate
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, accuracy_score
import matplotlib.pyplot as plt

from src.config import (
    CASIA_VAL_CSV,
    FVC2000_VAL_CSV,
    SAVED_MODELS_DIR,
    EMBEDDING_DIM,
)
from src.utils.Dataset_Loader import FingerprintDataset
from src.utils.logger import get_logger

# Import model architecture from train.py
from src.fingerprint.train import FingerprintEmbeddingNet, DEVICE


# --------------------------
# Dataset Configuration
# --------------------------
DATASET_CONFIG = {
    "casia": {
        "val_csv": CASIA_VAL_CSV,
        "model_name": "casia_arcface_model.pth",
        "batch_size": 32,
        "embeddings_file": "casia_test_embeddings.pt",
    },
    "fvc2000": {
        "val_csv": FVC2000_VAL_CSV,
        "model_name": "fvc2000_arcface_model.pth",
        "batch_size": 32,
        "embeddings_file": "fvc2000_test_embeddings.pt",
    },
}


def get_dataset_config(dataset_name):
    """Get configuration for specified dataset."""
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_CONFIG[dataset_name]


# --------------------------
# Embedding Generation
# --------------------------
def generate_embeddings(dataset_name, output_dir="artifacts/embeddings"):
    """Generate and save embeddings for test dataset.

    Args:
        dataset_name: Name of dataset ('casia' or 'fvc2000')
        output_dir: Directory to save embeddings
    """
    logger = get_logger(f"test_{dataset_name}")
    config = get_dataset_config(dataset_name)

    logger.info(f"Generating embeddings for {dataset_name} dataset")

    # Load dataset
    test_dataset = FingerprintDataset(str(config["val_csv"]), train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    # Load model
    model_path = os.path.join(SAVED_MODELS_DIR, config["model_name"])
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.error(
            "Please train the model first using: python src/fingerprint/train.py --dataset {dataset_name}"
        )
        return None

    backbone = FingerprintEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    backbone.load_state_dict(checkpoint["backbone"])
    backbone.eval()

    logger.info(f"Loaded model from: {model_path}")
    logger.info(f"Processing {len(test_dataset)} samples...")

    # Extract embeddings
    all_embeddings = []
    all_labels = []
    all_paths = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Extracting embeddings"):
            imgs = imgs.to(DEVICE)
            embeddings = backbone(imgs)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Save embeddings
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, config["embeddings_file"])
    torch.save(
        {
            "embeddings": all_embeddings,
            "labels": all_labels,
            "dataset": dataset_name,
        },
        output_path,
    )

    logger.info(f"Saved embeddings to: {output_path}")
    logger.info(f"Embeddings shape: {all_embeddings.shape}")
    logger.info(f"Labels shape: {all_labels.shape}")

    return output_path


# --------------------------
# Evaluation Functions
# --------------------------
def compute_similarity_matrix(embeddings):
    """Compute cosine similarity matrix between all pairs."""
    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    # Compute cosine similarity
    similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    return similarity_matrix


def evaluate_model(
    dataset_name, embeddings_file=None, metrics_output_dir="artifacts/metrics"
):
    """Evaluate model performance using generated embeddings.

    Args:
        dataset_name: Name of dataset ('casia' or 'fvc2000')
        embeddings_file: Path to embeddings file (auto-detected if None)
        metrics_output_dir: Directory to save evaluation metrics
    """
    logger = get_logger(f"evaluate_{dataset_name}")
    config = get_dataset_config(dataset_name)

    # Determine embeddings file path
    if embeddings_file is None:
        embeddings_file = os.path.join(
            "artifacts/embeddings", config["embeddings_file"]
        )

    if not os.path.exists(embeddings_file):
        logger.error(f"Embeddings file not found: {embeddings_file}")
        logger.error(
            "Please generate embeddings first using: python src/fingerprint/test.py --dataset {dataset_name} --mode generate"
        )
        return

    logger.info(f"Loading embeddings from: {embeddings_file}")
    data = torch.load(embeddings_file)
    embeddings = data["embeddings"]
    labels = data["labels"]

    logger.info(f"Loaded embeddings: {embeddings.shape}, labels: {labels.shape}")

    # Compute similarity matrix
    logger.info("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(embeddings)

    # Create genuine and imposter pairs
    n_samples = len(labels)
    genuine_scores = []
    imposter_scores = []

    logger.info("Computing genuine and imposter scores...")
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            score = similarity_matrix[i, j].item()
            if labels[i] == labels[j]:
                genuine_scores.append(score)
            else:
                imposter_scores.append(score)

    genuine_scores = np.array(genuine_scores)
    imposter_scores = np.array(imposter_scores)

    logger.info(f"Genuine pairs: {len(genuine_scores)}")
    logger.info(f"Imposter pairs: {len(imposter_scores)}")

    # Compute statistics
    genuine_mean = genuine_scores.mean()
    genuine_std = genuine_scores.std()
    imposter_mean = imposter_scores.mean()
    imposter_std = imposter_scores.std()

    logger.info(f"Genuine scores - Mean: {genuine_mean:.4f}, Std: {genuine_std:.4f}")
    logger.info(f"Imposter scores - Mean: {imposter_mean:.4f}, Std: {imposter_std:.4f}")

    # Compute ROC and find EER
    y_true = np.concatenate(
        [np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))]
    )
    y_scores = np.concatenate([genuine_scores, imposter_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Find EER (Equal Error Rate)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]

    logger.info(f"EER: {eer:.4f} at threshold: {eer_threshold:.4f}")

    # Compute accuracy at EER threshold
    predictions = (y_scores >= eer_threshold).astype(int)
    accuracy = accuracy_score(y_true, predictions)
    logger.info(f"Accuracy at EER threshold: {accuracy:.4f}")

    # Compute FAR and FRR at different thresholds
    far_dict = {}
    frr_dict = {}
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        predictions = (y_scores >= thresh).astype(int)
        false_accept = np.sum((predictions == 1) & (y_true == 0))
        false_reject = np.sum((predictions == 0) & (y_true == 1))
        far = false_accept / len(imposter_scores)
        frr = false_reject / len(genuine_scores)
        far_dict[thresh] = far
        frr_dict[thresh] = frr
        logger.info(f"Threshold {thresh:.1f} - FAR: {far:.4f}, FRR: {frr:.4f}")

    # Save metrics
    os.makedirs(metrics_output_dir, exist_ok=True)
    metrics = {
        "dataset": dataset_name,
        "genuine_mean": genuine_mean,
        "genuine_std": genuine_std,
        "imposter_mean": imposter_mean,
        "imposter_std": imposter_std,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "accuracy_at_eer": accuracy,
        "far_dict": far_dict,
        "frr_dict": frr_dict,
    }

    metrics_file = os.path.join(metrics_output_dir, f"{dataset_name}_metrics.pth")
    torch.save(metrics, metrics_file)
    logger.info(f"Saved metrics to: {metrics_file}")

    return metrics


def visualize_similarity(
    embeddings_file=None, dataset_name=None, output_dir="artifacts/plots"
):
    """Visualize similarity distributions.

    Args:
        embeddings_file: Path to embeddings file
        dataset_name: Name of dataset (used if embeddings_file is None)
        output_dir: Directory to save plots
    """
    if embeddings_file is None and dataset_name is not None:
        config = get_dataset_config(dataset_name)
        embeddings_file = os.path.join(
            "artifacts/embeddings", config["embeddings_file"]
        )

    if not os.path.exists(embeddings_file):
        print(f"Embeddings file not found: {embeddings_file}")
        return

    data = torch.load(embeddings_file)
    embeddings = data["embeddings"]
    labels = data["labels"]
    dataset = data.get("dataset", "unknown")

    # Compute similarities
    similarity_matrix = compute_similarity_matrix(embeddings)

    n_samples = len(labels)
    genuine_scores = []
    imposter_scores = []

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            score = similarity_matrix[i, j].item()
            if labels[i] == labels[j]:
                genuine_scores.append(score)
            else:
                imposter_scores.append(score)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.hist(
        genuine_scores,
        bins=50,
        alpha=0.7,
        label=f"Genuine (n={len(genuine_scores)})",
        color="green",
    )
    plt.hist(
        imposter_scores,
        bins=50,
        alpha=0.7,
        label=f"Imposter (n={len(imposter_scores)})",
        color="red",
    )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title(f"Similarity Distribution - {dataset.upper()}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"{dataset}_similarity_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {plot_path}")
    plt.close()


# --------------------------
# Main Function
# --------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Test fingerprint embedding model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings
  python src/fingerprint/test.py --dataset casia --mode generate
  python src/fingerprint/test.py --dataset fvc2000 --mode generate
  
  # Evaluate model
  python src/fingerprint/test.py --dataset casia --mode evaluate
  python src/fingerprint/test.py --dataset fvc2000 --mode evaluate
  
  # Visualize distributions
  python src/fingerprint/test.py --dataset casia --mode visualize
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["casia", "fvc2000"],
        help="Dataset to test on (casia or fvc2000)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["generate", "evaluate", "visualize", "all"],
        help="Test mode: generate (create embeddings), evaluate (compute metrics), visualize (plot distributions), all (do everything)",
    )

    parser.add_argument(
        "--embeddings-file",
        type=str,
        default=None,
        help="Path to embeddings file (optional, auto-detected if not provided)",
    )

    args = parser.parse_args()

    if args.mode == "generate":
        generate_embeddings(args.dataset)

    elif args.mode == "evaluate":
        evaluate_model(args.dataset, embeddings_file=args.embeddings_file)

    elif args.mode == "visualize":
        visualize_similarity(
            embeddings_file=args.embeddings_file, dataset_name=args.dataset
        )

    elif args.mode == "all":
        # Generate, evaluate, and visualize
        embeddings_file = generate_embeddings(args.dataset)
        if embeddings_file:
            evaluate_model(args.dataset, embeddings_file=embeddings_file)
            visualize_similarity(
                embeddings_file=embeddings_file, dataset_name=args.dataset
            )


if __name__ == "__main__":
    main()
