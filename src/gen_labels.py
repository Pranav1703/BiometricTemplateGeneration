"""
Unified Label Generation Script for Fingerprint Datasets
Supports: CASIA, FVC2000, FVC2004, and CMBD datasets

Usage:
  # Generate labels for FVC2000 dataset
  python -m src.utils.gen_labels --dataset fvc2000

  # Generate labels for CMBD dataset (Stratified 4:1 Split)
  python -m src.utils.gen_labels --dataset cmbd
"""

import argparse
import csv
import random
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Optional

# Make sure to add CMBD_LABELS_DIR and CMBD_DIR to your src.config!
from src.config import (
    FVC2000_LABELS_DIR,
    FVC2000_DB1A_DIR,
    FVC2004_LABELS_DIR,
    FVC2004_DB1A_DIR,
    CASIA_LABELS_DIR,
    CASIA_DIR,
    CMBD_LABELS_DIR,
    CMBD_DIR,
)
from src.utils.logger import get_logger

ALLOWED_EXT = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
VAL_RATIO = 0.2  # Default validation split ratio (1/5 images)
SEED = 42

# --------------------------
# Dataset Scanners
# --------------------------
def extract_fvc2000_person_id(filename: str) -> Optional[str]:
    m = re.match(r"^(\d+)_\d+", filename)
    return m.group(1) if m else None

def scan_fvc2000_root(root: Path) -> List[Tuple[str, str]]:
    fingerprint_rows = []
    all_files = sorted(root.iterdir())
    for p in all_files:
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            person_id = extract_fvc2000_person_id(p.name)
            if person_id:
                fingerprint_rows.append((str(p), person_id))
    return fingerprint_rows

def scan_fvc2004_root(root: Path) -> List[Tuple[str, str]]:
    fingerprint_rows = []
    subdirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("Db")]
    
    if subdirs:
        db1_dir = next((d for d in subdirs if d.name == "Db1_a"), None)
        if db1_dir:
            all_files = sorted(db1_dir.iterdir())
            for p in all_files:
                if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
                    person_id = extract_fvc2000_person_id(p.name)
                    if person_id:
                        fingerprint_rows.append((str(p), person_id))
    else:
        all_files = sorted(root.iterdir())
        for p in all_files:
            if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
                person_id = extract_fvc2000_person_id(p.name)
                if person_id:
                    fingerprint_rows.append((str(p), person_id))
    return fingerprint_rows

def extract_casia_finger_id(filepath: Path) -> str:
    m = re.search(r'(\d+_[a-zA-Z]\d+)', filepath.name)
    if m:
        return m.group(1).upper()
    return f"{filepath.parent.parent.name}_{filepath.parent.name}"

def scan_casia_root(root: Path) -> List[Tuple[str, str]]:
    fingerprint_rows = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            finger_id = extract_casia_finger_id(p)
            fingerprint_rows.append((str(p.resolve()), finger_id))
    return fingerprint_rows

# --- NEW: CMBD Scanner ---
def scan_cmbd_root(root: Path) -> List[Tuple[str, str]]:
    """Scans CMBD dataset. Expects folders representing classes."""
    fingerprint_rows = []
    # Grab all subdirectories (the 104 classes)
    folders = sorted([d for d in root.iterdir() if d.is_dir()])
    
    for folder in folders:
        person_id = folder.name
        for p in sorted(folder.glob("*.*")):
            if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
                fingerprint_rows.append((str(p.resolve()), person_id))
    return fingerprint_rows


# --------------------------
# Common Functions
# --------------------------
def save_csv(rows: List[Tuple[str, str]], outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "person_id"])
        writer.writerows(rows)

def stratified_split_and_save(
    rows: List[Tuple[str, str]],
    outdir: Path,
    prefix: str,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
):
    """
    Groups images by class first, THEN splits them. 
    Ensures every class has representation in both Train and Val sets.
    """
    random.seed(seed)
    
    # Group by identity
    grouped_data = defaultdict(list)
    for filepath, person_id in rows:
        grouped_data[person_id].append((filepath, person_id))
        
    train_rows = []
    val_rows = []
    
    # Split each class individually
    for person_id, images in grouped_data.items():
        random.shuffle(images)
        split_idx = int(len(images) * (1 - val_ratio))
        
        # Force at least 1 val image if the folder has more than 1 image
        if split_idx == len(images) and len(images) > 1:
            split_idx -= 1
            
        train_rows.extend(images[:split_idx])
        val_rows.extend(images[split_idx:])
        
    # Shuffle the final lists so batches are mixed
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    save_csv(train_rows, outdir / f"{prefix}_train.csv")
    save_csv(val_rows, outdir / f"{prefix}_val.csv")

    print(f"{prefix} dataset stratified split: {len(train_rows)} train, {len(val_rows)} val")
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
    "fvc2004": {
        "scanner": scan_fvc2004_root,
        "default_root": FVC2004_DB1A_DIR,
        "default_outdir": FVC2004_LABELS_DIR,
        "csv_prefix": "fvc2004",
    },
    "casia": {
        "scanner": scan_casia_root,
        "default_root": CASIA_DIR,
        "default_outdir": CASIA_LABELS_DIR,
        "csv_prefix": "casia",
    },
    "cmbd": {
        "scanner": scan_cmbd_root,
        "default_root": CMBD_DIR,
        "default_outdir": CMBD_LABELS_DIR,
        "csv_prefix": "cmbd",
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
    logger = get_logger(f"gen_labels_{dataset}")

    if dataset not in DATASET_SCANNERS:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: {list(DATASET_SCANNERS.keys())}")

    config = DATASET_SCANNERS[dataset]
    root_path = Path(root) if root else Path(config["default_root"])
    output_path = Path(outdir) if outdir else Path(config["default_outdir"])
    csv_prefix = config["csv_prefix"]

    if not root_path.exists() or not root_path.is_dir():
        logger.error(f"Root path does not exist or is not a directory: {root_path}")
        raise SystemExit(f"Error: Root path does not exist: {root_path}")

    print(f"Scanning {dataset} dataset at: {root_path}")

    scanner = config["scanner"]
    fingerprint_rows = scanner(root_path)

    if not fingerprint_rows:
        logger.error("No valid images found! Check your directory path and filenames.")
        return

    unique_classes = len(set([row[1] for row in fingerprint_rows]))
    print(f"Found {len(fingerprint_rows)} images across {unique_classes} unique classes.")

    train_count, val_count = stratified_split_and_save(
        fingerprint_rows, output_path, csv_prefix, val_ratio=val_ratio, seed=seed
    )

    print(f"Train/validation CSV files written to: {output_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Generate train/val labels for fingerprint datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["casia", "fvc2000", "fvc2004", "cmbd"],
        help="Dataset to generate labels for",
    )
    parser.add_argument("--root", type=str, default=None, help="Custom root directory")
    parser.add_argument("--outdir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")

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