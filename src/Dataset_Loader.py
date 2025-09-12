import csv
from pathlib import Path
from torch.utils.data import Dataset
import torch
import os
import kagglehub
from src.fingerprint.preprocess_fingerprint import preprocess_fingerprint
from .gen_labels import main as gen_labels
from src.utils.logger import get_logger

DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "CASIA-dataset"
LABELS_DIR = DATA_DIR / "labels"

class FingerprintDataset(Dataset):
    def __init__(self, csv_file, train=True):
        download_dataset()
        self.samples = []
        self.train = train
        self.label_to_index = {}

        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = int(row['person_id'])
                if pid not in self.label_to_index:
                    self.label_to_index[pid] = len(self.label_to_index)
                idx = self.label_to_index[pid]
                self.samples.append((row['filepath'], idx))

        # expose num_classes for ArcFace head
        self.num_classes = len(self.label_to_index)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_tensor = preprocess_fingerprint(img_path, train=self.train)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

def download_dataset():
    logger = get_logger("train")
    if not DATASET_DIR.exists():
        logger.warning("Dataset not found, Downloading...")
        os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)
        path = kagglehub.dataset_download("your/new-dataset-name")
        logger.info("Downloaded dataset to: %s", path)
    else:
        logger.info("Dataset is already Downloaded")
        print("Dataset is already Downloaded")

    if not LABELS_DIR.exists():
        logger.info("Creating Labels...")
        gen_labels()
        logger.info("Labels Created...")
        print("\n--- Process Finished ---")
    else:
        logger.info("Labels are already Created")
        print("Labels is already Created")
