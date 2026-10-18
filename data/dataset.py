"""
PyTorch Dataset and Batch Sampler for Multi-Session Contrastive Learning.
Implements:
  - EEGMultiSessionDataset: Torch Dataset wrapping EEG epochs and metadata.
  - BalancedSessionBatchSampler: Constructs mini-batches with P classes and K views
    (across different temporal sessions) to maximize Supervised Contrastive Loss effectiveness.
"""

import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
from typing import List, Dict, Iterator

class EEGMultiSessionDataset(Dataset):
    """
    PyTorch Dataset wrapping multi-session EEG epochs with subject and session labels.
    """
    def __init__(self, epochs: np.ndarray, metadata: List[Dict]):
        """
        Args:
            epochs: np.ndarray of shape (N, Channels, TimeSamples)
            metadata: List of metadata dictionaries
        """
        self.epochs = torch.tensor(epochs, dtype=torch.float32)
        self.metadata = metadata
        self.subject_ids = torch.tensor([m["subject_id"] for m in metadata], dtype=torch.long)
        self.session_ids = torch.tensor([m["session_id"] for m in metadata], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.epochs)

    def __getitem__(self, idx: int):
        return {
            "epoch": self.epochs[idx],
            "subject_id": self.subject_ids[idx],
            "session_id": self.session_ids[idx],
            "index": idx
        }

class BalancedSessionBatchSampler(Sampler):
    """
    Samples batches specifically tailored for Supervised Contrastive Learning:
    Selects 'num_classes_per_batch' subjects, and for each subject, samples
    'num_samples_per_class' epochs (preferably from distinct sessions).
    """
    def __init__(
        self,
        dataset: EEGMultiSessionDataset,
        num_classes_per_batch: int = 8,
        num_samples_per_class: int = 4,
        total_batches: int = 100
    ):
        self.dataset = dataset
        self.p = num_classes_per_batch
        self.k = num_samples_per_class
        self.batch_size = self.p * self.k
        self.total_batches = total_batches

        # Build index mapping: subject_id -> list of indices
        self.subject_to_indices = {}
        for idx, sub in enumerate(dataset.subject_ids.numpy()):
            if sub not in self.subject_to_indices:
                self.subject_to_indices[sub] = []
            self.subject_to_indices[sub].append(idx)

        self.available_subjects = list(self.subject_to_indices.keys())
        if len(self.available_subjects) < self.p:
            self.p = len(self.available_subjects)
            self.batch_size = self.p * self.k

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self) -> Iterator[List[int]]:
        for _ in range(self.total_batches):
            batch = []
            selected_classes = np.random.choice(self.available_subjects, size=self.p, replace=False)
            for cls in selected_classes:
                indices = self.subject_to_indices[cls]
                if len(indices) >= self.k:
                    chosen = np.random.choice(indices, size=self.k, replace=False)
                else:
                    chosen = np.random.choice(indices, size=self.k, replace=True)
                batch.extend(chosen.tolist())
            np.random.shuffle(batch)
            yield batch
