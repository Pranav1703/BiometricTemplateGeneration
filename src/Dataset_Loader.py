import csv
from pathlib import Path
from torch.utils.data import Dataset
import torch
import os
import kagglehub
from src.fingerprint.preprocess_fingerprint import preprocess_fingerprint
from .gen_labels import main as gen_labels
from src.utils.logger import get_logger
from .config import FVC2000_DB1A_DIR, FVC2000_LABELS_DIR


class FingerprintDataset(Dataset):
    def __init__(self, csv_file, train=True):
        self.samples = []
        self.train = train
        
        # FIX: Do not build dynamic labels. 
        # FVC IDs are 1..100. We map them to 0..99 directly.
        
        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filepath = row['filepath']
                # FVC IDs are strings "1", "2", ... "100"
                # We convert "1" -> 0, "100" -> 99
                pid = int(row['person_id'])
                label_idx = pid - 1  
                
                self.samples.append((filepath, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load and Preprocess
        try:
            img_tensor = preprocess_fingerprint(img_path, train=self.train)
        except Exception as e:
            # Fallback for corrupted images (rare but possible)
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, 224, 224)), torch.tensor(label, dtype=torch.long)

        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

def download_dataset():
    logger = get_logger("train")
    if not os.path.exists(FVC2000_DB1A_DIR):
        logger.warning("Dataset not found, Need to Download...")
        # os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)
        # path = kagglehub.dataset_download("your/new-dataset-name")
        # logger.info("Downloaded dataset to: %s", path)
    else:
        logger.info("Dataset is already Downloaded")
        print("Dataset is already Downloaded")

    if not os.path.exists(FVC2000_LABELS_DIR):
        logger.info("Creating Labels...")
        gen_labels()
        logger.info("Labels Created...")
        print("\n--- Process Finished ---")
    else:
        logger.info("Labels are already Created")
        print("Labels is already Created")
