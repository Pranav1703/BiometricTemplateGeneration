import csv
import random
import re
from pathlib import Path
from src.utils.logger import get_logger  # still using your logger utility

ALLOWED_EXT = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# --- Hard-coded paths instead of src.config ---
DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "CASIA-dataset"   # root of your fingerprint data
LABELS_DIR = DATA_DIR / "labels"           # where CSVs will be saved
VAL_RATIO = 0.2                            # validation split ratio

def extract_person_id(folder_name: str) -> str:
    """Return numeric prefix if present, else folder name."""
    m = re.match(r"(\d+)", folder_name)
    return m.group(1) if m else folder_name

def gather_images_from_dir(dirpath: Path):
    """Recursively gather allowed image files from dirpath."""
    rows = []
    if not dirpath or not dirpath.exists():
        return rows
    for p in dirpath.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOWED_EXT:
            rows.append(p.resolve())
    return rows

def scan_root(root: Path):
    fingerprint_rows = []
    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue

        person_id = extract_person_id(person_dir.name)

        # L and R (or L0, L1, R0…) subfolders
        for sub in person_dir.iterdir():
            if sub.is_dir():
                images = gather_images_from_dir(sub)
                fingerprint_rows += [(str(p), person_id) for p in images]

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

    root = Path(DATASET_DIR)
    if not root.exists() or not root.is_dir():
        logger.error(f"Root path does not exist or is not a directory: {root}")
        raise SystemExit(f"Root path does not exist or is not a directory: {root}")

    print(f"Scanning root: {root}")
    fingerprint_rows = scan_root(root)
    logger.info(f"Found {len(fingerprint_rows)} fingerprint images")

    outdir = Path(LABELS_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    split_and_save(fingerprint_rows, outdir, "fingerprint", val_ratio=VAL_RATIO)

    print(f"Train/validation CSV files written to {outdir.resolve()}")

if __name__ == "__main__":
    main()
