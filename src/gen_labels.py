import csv
import random
import re
from pathlib import Path
from src.utils.logger import get_logger
from .config import FVC2000_LABELS_DIR, FVC2000_DB1A_DIR

ALLOWED_EXT = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Validation split ratio (e.g., 0.2 means 20% for validation)
VAL_RATIO = 0.2

def extract_person_id_from_filename(filename: str) -> str:
    """
    Parses FVC filenames like '1_1.tif' or '100_8.tif'.
    Returns the number before the underscore (the Person ID).
    """
    # Regex explanation:
    # ^(\d+)  -> Start of string, capture one or more digits (Group 1)
    # _       -> Match a literal underscore
    # \d+     -> Match the impression number (we ignore this)
    m = re.match(r"^(\d+)_\d+", filename)
    return m.group(1) if m else ""

def scan_root(root: Path):
    """
    Scans a flat directory for images matching the pattern ID_Impression.tif.
    """
    fingerprint_rows = []
    
    # Sort the file list to ensure consistent ordering before shuffling
    # (Fixes reproducibility issues across different OS file systems)
    all_files = sorted(root.iterdir())

    for p in all_files:
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            # Parse the filename to get the ID
            person_id = extract_person_id_from_filename(p.name)
            
            if person_id:
                fingerprint_rows.append((str(p), person_id))
            else:
                # Optional: Warn if a file doesn't match the pattern
                # print(f"Skipping non-compliant file: {p.name}")
                pass

    return fingerprint_rows

def save_csv(rows, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "person_id"])
        writer.writerows(rows)

def split_and_save(rows, outdir: Path, prefix: str, val_ratio=0.2, seed=42):
    random.seed(seed)
    random.shuffle(rows)
    
    split_idx = int(len(rows) * (1 - val_ratio))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    save_csv(train_rows, outdir / f"{prefix}_train.csv")
    save_csv(val_rows, outdir / f"{prefix}_val.csv")

    print(f"{prefix} dataset split: {len(train_rows)} train, {len(val_rows)} val")

def main():
    logger = get_logger("train")

    root = Path(FVC2000_DB1A_DIR)
    
    if not root.exists() or not root.is_dir():
        logger.error(f"Root path does not exist or is not a directory: {root}")
        raise SystemExit(f"Root path does not exist or is not a directory: {root}")

    print(f"Scanning root directory: {root}")
    fingerprint_rows = scan_root(root)
    
    if not fingerprint_rows:
        logger.error("No valid images found! Check your directory path and filenames.")
        return

    logger.info(f"Found {len(fingerprint_rows)} valid fingerprint images.")

    outdir = Path(FVC2000_LABELS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    split_and_save(fingerprint_rows, outdir, "fvc2000", val_ratio=VAL_RATIO)

    print(f"Train/validation CSV files written to {outdir.resolve()}")

if __name__ == "__main__":
    main()