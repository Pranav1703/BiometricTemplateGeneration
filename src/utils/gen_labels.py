"""
Unified Label Generation Script for Fingerprint Datasets
Supports: CASIA and FVC2000 datasets

Usage:
  # Generate labels for FVC2000 dataset
  python -m src.utils.gen_labels --dataset fvc2000
  
  # Generate labels for CASIA dataset  
  python -m src.utils.gen_labels --dataset casia
"""

import argparse
import csv
import random
import re
from pathlib import Path
from typing import List, Tuple, Optional
from src.config import FVC2000_LABELS_DIR, FVC2000_DB1A_DIR, CASIA_LABELS_DIR, CASIA_DIR
from src.utils.logger import get_logger

ALLOWED_EXT = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
VAL_RATIO = 0.2  # Default validation split ratio
SEED = 42


# --------------------------
# FVC2000 Dataset Functions
# --------------------------
def extract_fvc2000_person_id(filename: str) -> Optional[str]:
    """
    Parses FVC filenames like '1_1.tif' or '100_8.tif'.
    Returns the number before the underscore (the Person ID).
    """
    m = re.match(r"^(\d+)_\d+", filename)
    return m.group(1) if m else None


def scan_fvc2000_root(root: Path) -> List[Tuple[str, str]]:
    """
    Scans a flat directory for FVC2000 images.
    Expected pattern: ID_Impression.tif (e.g., 1_1.tif, 100_8.tif)
    """
    fingerprint_rows = []

    # Sort for consistent ordering
    all_files = sorted(root.iterdir())

    for p in all_files:
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            person_id = extract_fvc2000_person_id(p.name)
            if person_id:
                fingerprint_rows.append((str(p), person_id))

    return fingerprint_rows


# --------------------------
# CASIA Dataset Functions
# --------------------------
def extract_casia_person_id(folder_name: str) -> str:
    """
    Extracts person ID from CASIA folder name.
    Folder format: 000, 001, 002, etc.
    """
    m = re.match(r"(\d+)", folder_name)
    return m.group(1) if m else folder_name


def gather_images_from_dir(dirpath: Path) -> List[Path]:
    """Recursively gather allowed image files from directory."""
    rows = []
    if not dirpath or not dirpath.exists():
        return rows
    for p in dirpath.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            rows.append(p.resolve())
    return rows


def scan_casia_root(root: Path) -> List[Tuple[str, str]]:
    """
    Scans nested directory structure for CASIA images.
    Expected structure: root/001/Fingerprint/*.bmp, root/001/*/*.bmp
    """
    fingerprint_rows = []

    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue

        person_id = extract_casia_person_id(person_dir.name)

        # Scan subdirectories (L, R, or direct images)
        for sub in person_dir.iterdir():
            if sub.is_dir():
                images = gather_images_from_dir(sub)
                fingerprint_rows += [(str(p), person_id) for p in images]
            elif sub.is_file() and sub.suffix.lower() in ALLOWED_EXT:
                fingerprint_rows.append((str(sub), person_id))

    return fingerprint_rows


# --------------------------
# Common Functions
# --------------------------
def save_csv(rows: List[Tuple[str, str]], outpath: Path):
    """Save rows to CSV file with filepath and person_id columns."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "person_id"])
        writer.writerows(rows)


def split_and_save(
    rows: List[Tuple[str, str]],
    outdir: Path,
    prefix: str,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
):
    """Split data into train/val and save to CSV files."""
    random.seed(seed)
    random.shuffle(rows)

    split_idx = int(len(rows) * (1 - val_ratio))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    save_csv(train_rows, outdir / f"{prefix}_train.csv")
    save_csv(val_rows, outdir / f"{prefix}_val.csv")

    print(f"{prefix} dataset split: {len(train_rows)} train, {len(val_rows)} val")

    return len(train_rows), len(val_rows)


# --------------------------
# Dataset Scanner Mapping
# --------------------------
DATASET_SCANNERS = {
    "fvc2000": {
        "scanner": scan_fvc2000_root,
        "default_root": FVC2000_DB1A_DIR,
        "default_outdir": FVC2000_LABELS_DIR,
        "csv_prefix": "fvc2000",
    },
    "casia": {
        "scanner": scan_casia_root,
        "default_root": CASIA_DIR,
        "default_outdir": CASIA_LABELS_DIR,
        "csv_prefix": "casia",
    },
}


# --------------------------
# Main Function
# --------------------------
def generate_labels(
    dataset: str,
    root: Optional[str] = None,
    outdir: Optional[str] = None,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
):
    """
    Generate train/validation CSV labels for fingerprint datasets.

    Args:
        dataset: Dataset name ('casia' or 'fvc2000')
        root: Root directory of the dataset (auto-detected if not provided)
        outdir: Output directory for CSV files (auto-detected if not provided)
        val_ratio: Validation split ratio (default: 0.2)
        seed: Random seed for reproducibility
    """
    logger = get_logger(f"gen_labels_{dataset}")

    if dataset not in DATASET_SCANNERS:
        raise ValueError(
            f"Unknown dataset: {dataset}. Choose from: {list(DATASET_SCANNERS.keys())}"
        )

    config = DATASET_SCANNERS[dataset]

    # Use provided paths or defaults
    root_path = Path(root) if root else Path(config["default_root"])
    output_path = Path(outdir) if outdir else Path(config["default_outdir"])
    csv_prefix = config["csv_prefix"]

    # Validate root directory
    if not root_path.exists() or not root_path.is_dir():
        logger.error(f"Root path does not exist or is not a directory: {root_path}")
        raise SystemExit(f"Error: Root path does not exist: {root_path}")

    print(f"Scanning {dataset} dataset at: {root_path}")

    # Scan for images
    scanner = config["scanner"]
    fingerprint_rows = scanner(root_path)

    if not fingerprint_rows:
        logger.error("No valid images found! Check your directory path and filenames.")
        return

    logger.info(f"Found {len(fingerprint_rows)} valid fingerprint images.")
    print(f"Found {len(fingerprint_rows)} images")

    # Split and save
    train_count, val_count = split_and_save(
        fingerprint_rows, output_path, csv_prefix, val_ratio=val_ratio, seed=seed
    )

    print(f"Train/validation CSV files written to: {output_path.resolve()}")
    logger.info(f"Generated {train_count} training and {val_count} validation samples")


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/validation labels for fingerprint datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate labels for FVC2000 dataset
  python -m src.utils.gen_labels --dataset fvc2000
  
  # Generate labels for CASIA dataset  
  python -m src.utils.gen_labels --dataset casia
  
  # With custom paths
  python -m src.utils.gen_labels --dataset fvc2000 --root ./data/FVC2000/DB1_a --outdir ./data/FVC2000/labels
  
  # With custom validation ratio
  python -m src.utils.gen_labels --dataset casia --val-ratio 0.3
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["casia", "fvc2000"],
        help="Dataset to generate labels for",
    )

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory of the dataset (auto-detected if not provided)",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for CSV files (auto-detected if not provided)",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=f"Validation split ratio (default: {VAL_RATIO})",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})",
    )

    args = parser.parse_args()

    generate_labels(
        dataset=args.dataset,
        root=args.root,
        outdir=args.outdir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
