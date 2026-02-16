import csv
from pathlib import Path
from torch.utils.data import Dataset
import torch
import os
from typing import Optional, Tuple

from .preprocess_fingerprint import preprocess_fingerprint
from .logger import get_logger
from ..config import CASIA_DIR, FVC2000_DIR, CASIA_LABELS_DIR, FVC2000_LABELS_DIR


class FingerprintDataset(Dataset):
    """
    Unified dataset loader for fingerprint images.

    Supports multiple datasets:
    - CASIA: CASIA fingerprint dataset
    - FVC2000: FVC2000 DB1a fingerprint dataset

    The CSV file should contain columns:
    - filepath: Path to the image file
    - person_id: ID of the person (will be converted to 0-indexed labels)

    Usage:
        # For FVC2000
        dataset = FingerprintDataset('datasets/FVC2000/labels/fvc2000_train.csv', train=True)

        # For CASIA
        dataset = FingerprintDataset('datasets/CASIA-dataset/labels/casia_train.csv', train=True)
    """

    def __init__(
        self, csv_file: str, train: bool = True, dataset_type: Optional[str] = None
    ):
        """
        Initialize the dataset.

        Args:
            csv_file: Path to CSV file containing image paths and labels
            train: Whether this is for training (affects preprocessing augmentations)
            dataset_type: Type of dataset ('casia', 'fvc2000', or None for auto-detect)
        """
        self.samples = []
        self.train = train
        self.csv_file = csv_file
        self.dataset_type = dataset_type or self._detect_dataset_type(csv_file)

        self._load_samples()

    def _detect_dataset_type(self, csv_file: str) -> str:
        """Auto-detect dataset type from file path."""
        csv_path_lower = csv_file.lower()
        if "casia" in csv_path_lower:
            return "casia"
        elif "fvc" in csv_path_lower or "fvc2000" in csv_path_lower:
            return "fvc2000"
        else:
            # Try to infer from directory structure
            csv_dir = os.path.dirname(csv_file)
            if os.path.exists(CASIA_DIR) and CASIA_DIR in csv_file:
                return "casia"
            elif os.path.exists(FVC2000_DIR) and FVC2000_DIR in csv_file:
                return "fvc2000"
            else:
                return "unknown"

    def _load_samples(self):
        """Load samples from CSV file."""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        with open(self.csv_file, newline="") as f:
            reader = csv.DictReader(f)

            # Detect column names (handle variations)
            fieldnames = reader.fieldnames or []

            # Map possible column names
            filepath_col = self._get_column_name(
                fieldnames, ["filepath", "file_path", "path", "image_path"]
            )
            person_id_col = self._get_column_name(
                fieldnames, ["person_id", "label", "id", "class", "subject_id"]
            )

            if not filepath_col:
                raise ValueError(
                    f"Could not find filepath column in {self.csv_file}. Available columns: {fieldnames}"
                )
            if not person_id_col:
                raise ValueError(
                    f"Could not find person_id column in {self.csv_file}. Available columns: {fieldnames}"
                )

            for row in reader:
                filepath = row[filepath_col]

                # Handle person_id based on dataset type
                if self.dataset_type == "fvc2000":
                    # FVC IDs are 1..100, map to 0..99
                    pid = int(row[person_id_col])
                    label_idx = pid - 1
                else:
                    # CASIA and others: assume IDs are already 0-indexed or sequential
                    pid = int(row[person_id_col])
                    label_idx = pid

                self.samples.append((filepath, label_idx))

        if len(self.samples) == 0:
            raise ValueError(f"No samples loaded from {self.csv_file}")

    def _get_column_name(self, fieldnames: list, possible_names: list) -> Optional[str]:
        """Find matching column name from possible options."""
        for name in possible_names:
            if name in fieldnames:
                return name
        return fieldnames[0] if fieldnames else None

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        img_path, label = self.samples[idx]

        # Load and preprocess
        try:
            img_tensor = preprocess_fingerprint(
                img_path, train=self.train, dataset_type=self.dataset_type
            )
        except Exception as e:
            # Log error and return zero tensor as fallback
            logger = get_logger("dataset")
            logger.warning(f"Error loading {img_path}: {e}. Returning zero tensor.")
            img_tensor = torch.zeros((3, 224, 224))

        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

    def get_num_classes(self) -> int:
        """Return number of unique classes in the dataset."""
        return len(set(label for _, label in self.samples))

    def get_class_distribution(self) -> dict:
        """Return distribution of samples per class."""
        from collections import Counter

        labels = [label for _, label in self.samples]
        return dict(Counter(labels))

    def __repr__(self) -> str:
        """String representation of dataset."""
        return (
            f"FingerprintDataset("
            f"csv_file='{self.csv_file}', "
            f"dataset_type='{self.dataset_type}', "
            f"n_samples={len(self)}, "
            f"n_classes={self.get_num_classes()}, "
            f"train={self.train})"
        )


def create_fingerprint_dataloaders(
    train_csv: str,
    val_csv: str,
    batch_size: int = 32,
    num_workers: int = 2,
    dataset_type: Optional[str] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create training and validation dataloaders.

    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        dataset_type: Type of dataset ('casia', 'fvc2000', or None for auto-detect)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = FingerprintDataset(train_csv, train=True, dataset_type=dataset_type)
    val_dataset = FingerprintDataset(val_csv, train=False, dataset_type=dataset_type)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# Legacy function for backward compatibility
def download_dataset():
    """Legacy function - datasets should be manually downloaded."""
    logger = get_logger("dataset")
    logger.info("Dataset download is handled manually.")
    logger.info("Please ensure datasets are available at:")
    logger.info(f"  - CASIA: {CASIA_DIR}")
    logger.info(f"  - FVC2000: {FVC2000_DIR}")
