import argparse
import csv
import random
import re
from pathlib import Path

ALLOWED_EXT = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}

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
    iris_rows = []

    for person_dir in sorted(root.iterdir()):
        if not person_dir.is_dir():
            continue

        person_id = extract_person_id(person_dir.name)

        # Fingerprint folder
        finger_dir = person_dir / "Fingerprint"
        if finger_dir.exists():
            images = gather_images_from_dir(finger_dir)
            fingerprint_rows += [(str(p), person_id) for p in images]

        # Iris folders: left & right
        left_dir = person_dir / "left"
        right_dir = person_dir / "right"
        for iris_dir in (left_dir, right_dir):
            if iris_dir.exists():
                images = gather_images_from_dir(iris_dir)
                iris_rows += [(str(p), person_id) for p in images]

    return fingerprint_rows, iris_rows

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to root directory containing person folders")
    parser.add_argument("--outdir", default=".", help="Directory to save CSV files")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio")
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Root path does not exist or is not a directory: {root}")

    print(f"Scanning root: {root}")
    fingerprint_rows, iris_rows = scan_root(root)
    print(f"Found {len(fingerprint_rows)} fingerprint images and {len(iris_rows)} iris images")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    split_and_save(fingerprint_rows, outdir, "fingerprint", val_ratio=args.val_ratio)
    split_and_save(iris_rows, outdir, "iris", val_ratio=args.val_ratio)

    print(f"Train/validation CSV files written to {outdir.resolve()}")

if __name__ == "__main__":
    main()

#python src/gen-labels.py --root ./data/IRIS_and_FINGERPRINT_DATASET --outdir ./labels --val_ratio 0.2
