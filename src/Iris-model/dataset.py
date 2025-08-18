import csv
from torch.utils.data import Dataset
import torch
from preprocess import preprocess_image  # fixed version

class IrisDataset(Dataset):
    def __init__(self, csv_file, train=True):
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
