import csv
from pathlib import Path
from torch.utils.data import Dataset
import torch
import os
import shutil
from src.preprocess.fingerprint import preprocess_fingerprint  # your preprocessing function
from .gen_labels import main as gen_labels
from src.utils.logger import get_logger
import kagglehub

# --- Hard-coded paths instead of src.config ---
# adjust these to your project’s actual folders
DATA_DIR = Path("data")
DATASET_DIR = DATA_DIR / "CASIA-dataset"      # where the CASIA fingerprint data lives
LABELS_DIR = DATA_DIR / "labels"              # where CSVs will be written

class FingerprintDataset(Dataset):
    def __init__(self, csv_file, train=True):
        """
        Args:
            csv_file (str or Path): Path to CSV file with 'filepath,person_id' columns
        """
        download_dataset()
        self.samples = []
        self.train = train

        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row['filepath'], int(row['person_id'])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_tensor = preprocess_fingerprint(img_path, train=self.train)  # returns tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

def download_dataset():
    logger = get_logger("train")
    if not DATASET_DIR.exists():
        logger.warning("Dataset not found, Downloading...")
        os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)
        # replace below string with the actual Kaggle dataset name if you’re still downloading it
        path = kagglehub.dataset_download("your/new-dataset-name")
        logger.info("Downloaded dataset to: %s", path)
        # move/cleanup code if needed – depends on KaggleHub folder structure
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

if __name__ == "__main__":
    download_dataset()
