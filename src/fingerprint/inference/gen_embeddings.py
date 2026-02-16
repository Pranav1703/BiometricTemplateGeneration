"""
Generate embeddings from trained fingerprint model.
Supports both FVC2000 and CASIA datasets.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.Dataset_Loader import FingerprintDataset
from src.config import FVC2000_VAL_CSV, CASIA_VAL_CSV, SAVED_MODELS_DIR
from src.fingerprint.train import FingerprintEmbeddingNet, EMBEDDING_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_CONFIG = {
    "fvc2000": {
        "val_csv": FVC2000_VAL_CSV,
        "model_name": "fvc2000_arcface_model.pth",
        "output_name": "fvc2000_embeddings.pt",
    },
    "casia": {
        "val_csv": CASIA_VAL_CSV,
        "model_name": "casia_arcface_model.pth",
        "output_name": "casia_embeddings.pt",
    },
}


def generate_embeddings(dataset_name, output_dir="artifacts/embeddings"):
    """Generate embeddings for a dataset."""
    config = DATASET_CONFIG[dataset_name]

    print(f"Generating embeddings for {dataset_name} dataset...")

    # Load model
    model_path = os.path.join(SAVED_MODELS_DIR, config["model_name"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please train the model first: python -m src.fingerprint.train --dataset {dataset_name}"
        )

    backbone = FingerprintEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    backbone.load_state_dict(checkpoint["backbone"])
    backbone.eval()

    print(f"Loaded model from: {model_path}")

    # Load dataset
    test_dataset = FingerprintDataset(config["val_csv"], train=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Processing {len(test_dataset)} images...")

    # Extract embeddings
    all_embeddings = []
    all_labels = []

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
    output_path = os.path.join(output_dir, config["output_name"])

    torch.save(
        {
            "embeddings": all_embeddings,
            "labels": all_labels,
            "dataset": dataset_name,
        },
        output_path,
    )

    print(f"Saved embeddings to: {output_path}")
    print(f"Embeddings shape: {all_embeddings.shape}")
    print(f"Labels shape: {all_labels.shape}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings from trained fingerprint model"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["fvc2000", "casia"],
        help="Dataset to generate embeddings for",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/embeddings",
        help="Output directory for embeddings",
    )

    args = parser.parse_args()

    generate_embeddings(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
