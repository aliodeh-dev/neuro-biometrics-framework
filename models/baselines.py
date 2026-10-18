"""
Baseline Models for Empirical Benchmarking.
Implements:
  1. Classical Handcrafted Features (Welch PSD in 5 frequency bands + SVM / Random Forest)
  2. Standard 1D-CNN trained solely with Cross-Entropy Loss (No contrastive learning)
  3. Standard EEGNet Baseline (Lawhern et al., 2018)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from config.settings import ModelConfig

class ClassicalPSDBaseline:
    """
    Classical Neuro-Biometric Pipeline:
    Extracts Power Spectral Density (PSD) using Welch's method across 5 bands:
    Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-45 Hz).
    Classifies using Support Vector Machine with RBF kernel or Random Forest.
    """
    def __init__(self, fs: int = 250, classifier_type: str = "svm"):
        self.fs = fs
        self.bands = {
            "delta": (0.5, 4.0),
            "theta": (4.0, 8.0),
            "alpha": (8.0, 13.0),
            "beta": (13.0, 30.0),
            "gamma": (30.0, 45.0)
        }
        self.classifier_type = classifier_type
        if classifier_type == "svm":
            self.clf = SVC(probability=True, kernel="rbf", C=1.0)
        else:
            self.clf = RandomForestClassifier(n_estimators=100, random_state=42)

    def extract_features_single(self, epoch: np.ndarray) -> np.ndarray:
        """
        Args:
            epoch: np.ndarray of shape (channels, samples)
        Returns:
            features: 1D array of shape (channels * 5,)
        """
        feats = []
        for ch in range(epoch.shape[0]):
            freqs, psd = welch(epoch[ch], fs=self.fs, nperseg=min(epoch.shape[1], 128))
            for _, (fmin, fmax) in self.bands.items():
                idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                band_power = np.sum(psd[idx_band])
                feats.append(band_power)
        # Log-transform power for variance stabilization
        return np.log(np.array(feats) + 1e-8)

    def extract_dataset(self, epochs: np.ndarray) -> np.ndarray:
        """
        Extracts features for an array of epochs: (N, channels, samples) -> (N, feats)
        """
        all_feats = [self.extract_features_single(ep) for ep in epochs]
        return np.array(all_feats)

    def fit(self, X: np.ndarray, y: np.ndarray):
        features = self.extract_dataset(X)
        self.clf.fit(features, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        features = self.extract_dataset(X)
        return self.clf.predict_proba(features)

    def get_embeddings(self, X: np.ndarray) -> np.ndarray:
        return self.extract_dataset(X)

class Standard1DCNNBaseline(nn.Module):
    """
    Standard 1D-CNN network trained purely with Cross-Entropy Loss.
    Serves as an ablation baseline to demonstrate the impact of:
      1. Lack of temporal attention
      2. Lack of Supervised Contrastive Loss
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.conv1 = nn.Conv1d(config.num_channels, 32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, config.num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool2(out)
        out = F.relu(self.bn3(self.conv3(out)))
        h = self.gap(out).squeeze(-1)
        return h

    def forward(self, x: torch.Tensor, return_projection: bool = False, return_logits: bool = True):
        h = self.extract_features(x)
        logits = self.fc(h)
        # For fair evaluation in metric space, provide normalized embedding
        z_norm = F.normalize(h, p=2, dim=1)
        return {
            "embedding": h,
            "projection": z_norm,
            "logits": logits
        }

class EEGNetBaseline(nn.Module):
    """
    Faithful PyTorch implementation of the standard EEGNet-8,2 architecture (Lawhern et al., 2018).
    Trained conventionally with Cross-Entropy.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        f1 = 8
        d = 2
        f2 = f1 * d

        self.conv1 = nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise = nn.Conv2d(f1, f2, (config.num_channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f2)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(0.25)

        self.separable = nn.Sequential(
            nn.Conv2d(f2, f2, (1, 16), padding=(0, 8), groups=f2, bias=False),
            nn.Conv2d(f2, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(f2, config.num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = self.bn1(self.conv1(x))
        out = self.elu(self.bn2(self.depthwise(out)))
        out = self.pool1(out)
        out = self.dropout1(out)
        out = self.separable(out)
        h = self.gap(out).flatten(start_dim=1)
        return h

    def forward(self, x: torch.Tensor, return_projection: bool = False, return_logits: bool = True):
        h = self.extract_features(x)
        logits = self.classifier(h)
        z_norm = F.normalize(h, p=2, dim=1)
        return {
            "embedding": h,
            "projection": z_norm,
            "logits": logits
        }
