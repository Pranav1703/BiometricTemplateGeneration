import csv
from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np
import kagglehub
import os
from src.config import DATA_DIR, DATASET_DIR, LABELS_DIR
import shutil
from src.preprocess.fingerprint import preprocess_fingerprint  # Import your preprocessing function
from src.preprocess.iris import preprocess_image
from .gen_labels import main as gen_labels
from src.utils.logger import get_logger

class FingerprintDataset(Dataset):
    def __init__(self, csv_file,train=True):
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
        img_tensor = preprocess_fingerprint(img_path, train=self.train)  # already a Tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

class IrisDataset(Dataset):
    def __init__(self, csv_file, train=True):
        
        download_dataset()
        self.samples = []
        self.train = train

        with open(csv_file, newline='') as f:
            import csv
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row['filepath'], int(row['person_id'])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # preprocess_image now returns tensor [1,H,W]
        img_tensor = preprocess_image(img_path, size=(224, 224))
        label_tensor = torch.tensor(label, dtype=torch.long)

        return img_tensor, label_tensor

def download_dataset():

    logger = get_logger("train")
    if not (os.path.exists(DATASET_DIR)):
        
        # Initializing Loger

        logger.warning("Dataset not found, Downloading...")
        # Setting it download in right folder
        os.environ["KAGGLEHUB_CACHE"] = DATA_DIR
        source_path = os.path.join(DATA_DIR, "datasets", "ninadmehendale", "multimodal-iris-fingerprint-biometric-data", "versions", "1", "IRIS and FINGERPRINT DATASET")
        destination_path = os.path.join(DATA_DIR)
        cleanup_path = os.path.join(DATA_DIR, "datasets")

        # Download latest version
        path = kagglehub.dataset_download("ninadmehendale/multimodal-iris-fingerprint-biometric-data/versions/1")

        logger.info("Path to dataset files:", path)
        try:
            # Check if the source folder exists before trying to move it
            if not os.path.exists(source_path):
                print(f"Error: Source path '{source_path}' does not exist.")
                return

            # Step 1: Move the folder
            print(f"Moving '{source_path}' to '{destination_path}'...")
            shutil.move(source_path, destination_path)
            print("Move complete. ✅")

            # Step 2: Remove the old, now-empty parent directory structure
            print(f"Cleaning up '{cleanup_path}'...")
            shutil.rmtree(cleanup_path)
            print("Cleanup complete. ✅")

        except FileNotFoundError:

            logger.error("Error: A file or directory was not found. Check your paths.")
            print(f"Error: A file or directory was not found. Check your paths.")
        except Exception as e:

            logger.error(f"An unexpected error occurred: {e}")
            print(f"An unexpected error occurred: {e}")

    else:
        logger.info("Dataset is already Downloaded")
        print("Dataset is already Downloaded")
        
        
    if not (os.path.exists(LABELS_DIR)):
        logger.info("Creating Labels...")
        gen_labels()
        
        logger.info("Labels Created...")
        print("\n--- Process Finished ---")

    else:
        logger.info("Labels is already Created")
        print("Labels is already Created")

if __name__ == "__main__":
    download_dataset()