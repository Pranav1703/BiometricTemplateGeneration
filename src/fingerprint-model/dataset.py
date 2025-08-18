import csv
from pathlib import Path
from torch.utils.data import Dataset
import torch
import numpy as np
from preprocess import preprocess_fingerprint  # Import your preprocessing function

class FingerprintDataset(Dataset):
    def __init__(self, csv_file,train=True):
        """
        Args:
            csv_file (str or Path): Path to CSV file with 'filepath,person_id' columns
        """
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
