"""
Biometric Cryptosystem Integration Module
Combines embedding generation with cancelable biometrics and fuzzy commitment

Usage:
    # Enroll a user
    python -m src.fingerprint.inference.enroll --dataset fvc2000 --user-key "user_001"

    # Verify a user
    python -m src.fingerprint.inference.verify --dataset fvc2000 --user-key "user_001"
"""

import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from src.config import (
    FVC2000_VAL_CSV,
    CASIA_VAL_CSV,
    SAVED_MODELS_DIR,
    EMBEDDING_DIM,
)
from src.utils.Dataset_Loader import FingerprintDataset
from src.fingerprint.train import FingerprintEmbeddingNet
from src.fingerprint.core.biometric_crypto_system import BiometricCryptoSystem

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_CONFIG = {
    "fvc2000": {
        "val_csv": FVC2000_VAL_CSV,
        "model_name": "fvc2000_arcface_model.pth",
    },
    "casia": {
        "val_csv": CASIA_VAL_CSV,
        "model_name": "casia_arcface_model.pth",
    },
}


def load_backbone(model_name):
    """Load trained fingerprint embedding model."""
    model_path = os.path.join(SAVED_MODELS_DIR, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please train the model first: python -m src.fingerprint.train --dataset <dataset>"
        )

    backbone = FingerprintEmbeddingNet(embedding_dim=EMBEDDING_DIM).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    backbone.load_state_dict(checkpoint["backbone"])
    backbone.eval()

    return backbone


def extract_embedding(backbone, image_tensor):
    """Extract embedding from image tensor."""
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        embedding = backbone(image_tensor)
        return embedding.cpu().numpy().flatten()


def extract_embeddings_from_dataset(backbone, dataset, batch_size=32):
    """Extract embeddings from dataset."""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    labels = []

    for images, targets in tqdm(dataloader, desc="Extracting embeddings"):
        batch_embeddings = extract_embedding(backbone, images)
        embeddings.append(batch_embeddings)
        labels.extend(targets.numpy())

    return np.array(embeddings), np.array(labels)


def enroll_user(dataset_name, user_key, template_dir="artifacts/templates"):
    """
    Enroll all users in the dataset with the biometric cryptosystem.

    Args:
        dataset_name: Name of dataset ('fvc2000' or 'casia')
        user_key: User/application-specific key
        template_dir: Directory to save templates
    """
    config = DATASET_CONFIG[dataset_name]

    print(f"=" * 60)
    print(f"Enrolling users from {dataset_name} dataset")
    print(f"=" * 60)

    # Load model
    print("\nLoading model...")
    backbone = load_backbone(config["model_name"])

    # Initialize cryptosystem
    crypto_system = BiometricCryptoSystem(
        embedding_dim=EMBEDDING_DIM, key_size=32, ecc_capacity=0.2, cancelable_alpha=0.6
    )

    # Load dataset
    print("Loading dataset...")
    dataset = FingerprintDataset(config["val_csv"], train=False)

    # Get embeddings
    print("Extracting embeddings...")
    embeddings, labels = extract_embeddings_from_dataset(backbone, dataset)

    # Enroll unique users
    unique_labels = np.unique(labels)
    print(f"Found {len(unique_labels)} unique users")

    os.makedirs(template_dir, exist_ok=True)

    templates = {}

    for label in tqdm(unique_labels, desc="Enrolling users"):
        # Get first embedding for this user
        user_indices = np.where(labels == label)[0]
        embedding = embeddings[user_indices[0]]

        # Enroll
        template = crypto_system.enroll(embedding, f"{user_key}_{label}")

        templates[str(label)] = {
            "hash_key": template["hash_key"],
            "helper_data": template["helper_data"].hex()
            if isinstance(template["helper_data"], bytes)
            else template["helper_data"],
            "cancelable_params": {
                "user_key": template["cancelable_params"]["user_key"],
                "alpha": float(template["cancelable_params"]["alpha"]),
                "embedding_dim": int(template["cancelable_params"]["embedding_dim"]),
            },
        }

    # Save templates
    template_file = os.path.join(template_dir, f"{dataset_name}_templates.json")
    with open(template_file, "w") as f:
        json.dump(templates, f, indent=2)

    print(f"\nEnrolled {len(templates)} users")
    print(f"Templates saved to: {template_file}")

    return templates


def verify_user(dataset_name, user_key, sample_index=0):
    """
    Verify a user from the dataset.

    Args:
        dataset_name: Name of dataset
        user_key: User key used during enrollment
        sample_index: Index of sample to verify
    """
    config = DATASET_CONFIG[dataset_name]

    print(f"=" * 60)
    print(f"Verifying user from {dataset_name} dataset")
    print(f"=" * 60)

    # Load model
    print("\nLoading model...")
    backbone = load_backbone(config["model_name"])

    # Initialize cryptosystem
    crypto_system = BiometricCryptoSystem(
        embedding_dim=EMBEDDING_DIM, key_size=32, ecc_capacity=0.2, cancelable_alpha=0.6
    )

    # Load dataset
    print("Loading dataset...")
    dataset = FingerprintDataset(config["val_csv"], train=False)

    # Get embedding for sample
    print(f"Extracting embedding for sample {sample_index}...")
    image, label = dataset[sample_index]
    embedding = extract_embedding(backbone, image.unsqueeze(0))

    # Create template for this user
    template = crypto_system.enroll(embedding, f"{user_key}_{label}")

    # Verify with same embedding
    print("\nVerifying with same embedding...")
    success, recovered_key = crypto_system.verify(
        embedding, template, f"{user_key}_{label}"
    )
    print(f"  Same embedding: {'SUCCESS' if success else 'FAILED'}")

    # Verify with different sample of same user (if available)
    user_indices = [
        i
        for i, l in enumerate(
            dataset.labels if hasattr(dataset, "labels") else range(len(dataset))
        )
        if i != sample_index
        and (dataset.samples[i][1] if hasattr(dataset, "samples") else i) == label
    ]

    if user_indices:
        image2, _ = dataset[user_indices[0]]
        embedding2 = extract_embedding(backbone, image2.unsqueeze(0))

        print("Verifying with different sample of same user...")
        success2, _ = crypto_system.verify(embedding2, template, f"{user_key}_{label}")
        print(f"  Different sample: {'SUCCESS' if success2 else 'FAILED'}")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Biometric Cryptosystem Enrollment and Verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll all users from dataset
  python -m src.fingerprint.inference.enroll --dataset fvc2000 --user-key "app_001"
  
  # Verify a user
  python -m src.fingerprint.inference.verify --dataset fvc2000 --user-key "app_001" --sample 0
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Enroll command
    enroll_parser = subparsers.add_parser("enroll", help="Enroll users")
    enroll_parser.add_argument(
        "--dataset", type=str, required=True, choices=["fvc2000", "casia"]
    )
    enroll_parser.add_argument(
        "--user-key", type=str, required=True, help="User/application key"
    )
    enroll_parser.add_argument(
        "--template-dir",
        type=str,
        default="artifacts/templates",
        help="Template output directory",
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a user")
    verify_parser.add_argument(
        "--dataset", type=str, required=True, choices=["fvc2000", "casia"]
    )
    verify_parser.add_argument("--user-key", type=str, required=True, help="User key")
    verify_parser.add_argument(
        "--sample", type=int, default=0, help="Sample index to verify"
    )

    args = parser.parse_args()

    if args.command == "enroll":
        enroll_user(args.dataset, args.user_key, args.template_dir)
    elif args.command == "verify":
        verify_user(args.dataset, args.user_key, args.sample)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
