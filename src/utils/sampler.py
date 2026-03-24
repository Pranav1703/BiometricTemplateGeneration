import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict

class PKSampler(Sampler):
    """
    Randomly samples P classes, and then K images per class.
    Guarantees positive pairs in every batch for BER calculation.
    """
    def __init__(self, dataset, p_classes=16, k_samples=4):
        self.dataset = dataset
        self.p_classes = p_classes
        self.k_samples = k_samples
        self.batch_size = p_classes * k_samples

        # Group all dataset indices by their class label
        self.label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.dataset.samples):
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())

        # Estimate how many batches make up a full "epoch"
        self.num_batches = len(self.dataset) // self.batch_size

    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            # 1. Pick P random fingers (classes)
            sampled_classes = np.random.choice(self.labels, self.p_classes, replace=False)

            # 2. Pick K images for each finger
            for label in sampled_classes:
                indices = self.label_to_indices[label]
                # If a class has fewer than K samples, replace=True allows duplicates to pad the batch
                sampled_indices = np.random.choice(
                    indices, size=self.k_samples, replace=(len(indices) < self.k_samples)
                )
                batch_indices.extend(sampled_indices)

            yield batch_indices

    def __len__(self):
        return self.num_batches