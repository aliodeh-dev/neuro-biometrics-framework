"""
Training and Optimization Script.
Trains WearableBrainprintNet with Supervised Contrastive Loss + Cross-Entropy,
as well as comparison baselines.
"""

import os
import argparse
import json
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.settings import FrameworkConfig
from data.synthetic_generator import SyntheticEEGGenerator
from data.dataset import EEGMultiSessionDataset, BalancedSessionBatchSampler
from models.brainprint_net import WearableBrainprintNet
from models.baselines import Standard1DCNNBaseline, EEGNetBaseline
from losses.supervised_contrastive import JointBiometricLoss
from evaluation.metrics import BiometricMetricsEvaluator

def train_proposed_model(
    config: FrameworkConfig,
    dataset: EEGMultiSessionDataset,
    epochs: int = 30
) -> Tuple[WearableBrainprintNet, Dict[str, Any]]:
    device = config.training.device
    model = WearableBrainprintNet(config.model).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = JointBiometricLoss(
        temperature=config.training.temperature,
        contrastive_weight=config.training.contrastive_weight
    )

    batch_sampler = BalancedSessionBatchSampler(
        dataset=dataset,
        num_classes_per_batch=min(8, config.num_subjects),
        num_samples_per_class=4,
        total_batches=40
    )
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

    history = {"loss": [], "supcon": [], "ce": []}

    print(f"\n=======================================================")
    print(f" Training Proposed WearableBrainprintNet (SupCon + CE)")
    print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_sampler.batch_size}")
    print(f"=======================================================")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_total = 0.0
        epoch_supcon = 0.0
        epoch_ce = 0.0

        for batch in dataloader:
            x = batch["epoch"].to(device)
            y = batch["subject_id"].to(device)

            optimizer.zero_grad()
            out = model(x, return_projection=True, return_logits=True)
            loss, l_supcon, l_ce = criterion(out["projection"], out["logits"], y)

            loss.backward()
            optimizer.step()

            epoch_total += loss.item()
            epoch_supcon += l_supcon.item()
            epoch_ce += l_ce.item()

        scheduler.step()

        avg_loss = epoch_total / len(dataloader)
        avg_supcon = epoch_supcon / len(dataloader)
        avg_ce = epoch_ce / len(dataloader)

        history["loss"].append(avg_loss)
        history["supcon"].append(avg_supcon)
        history["ce"].append(avg_ce)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f" Epoch [{epoch:2d}/{epochs:2d}] -> Total Loss: {avg_loss:.4f} | SupCon: {avg_supcon:.4f} | CE: {avg_ce:.4f}")

    return model, history

def train_baseline_model(
    config: FrameworkConfig,
    dataset: EEGMultiSessionDataset,
    model_name: str = "Standard1DCNN",
    epochs: int = 30
) -> Tuple[nn.Module, Dict[str, Any]]:
    device = config.training.device
    if model_name == "Standard1DCNN":
        model = Standard1DCNNBaseline(config.model).to(device)
    elif model_name == "EEGNet":
        model = EEGNetBaseline(config.model).to(device)
    else:
        raise ValueError(f"Unknown baseline: {model_name}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)
    history = {"loss": []}

    print(f"\n=======================================================")
    print(f" Training Baseline: {model_name} (Cross-Entropy Only)")
    print(f" Device: {device} | Epochs: {epochs}")
    print(f"=======================================================")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            x = batch["epoch"].to(device)
            y = batch["subject_id"].to(device)

            optimizer.zero_grad()
            out = model(x, return_logits=True)
            loss = criterion(out["logits"], y)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        history["loss"].append(avg_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f" Epoch [{epoch:2d}/{epochs:2d}] -> CE Loss: {avg_loss:.4f}")

    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Neuro-Biometrics Models")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--subjects", type=int, default=20, help="Number of subjects")
    parser.add_argument("--sessions", type=int, default=4, help="Number of sessions")
    args = parser.parse_args()

    cfg = FrameworkConfig(num_subjects=args.subjects, num_sessions=args.sessions)
    cfg.model.num_classes = args.subjects

    print("Generating Multi-Session Dataset...")
    generator = SyntheticEEGGenerator(cfg)
    epochs_data, metadata = generator.generate_full_dataset()
    dataset = EEGMultiSessionDataset(epochs_data, metadata)
    print(f"Dataset generated: {len(dataset)} total epochs across {args.subjects} subjects and {args.sessions} sessions.")

    # Train proposed
    proposed_model, prop_hist = train_proposed_model(cfg, dataset, epochs=args.epochs)

    # Save checkpoint
    os.makedirs("results/checkpoints", exist_ok=True)
    torch.save(proposed_model.state_dict(), "results/checkpoints/wearable_brainprint_net.pt")
    print("Saved checkpoint to results/checkpoints/wearable_brainprint_net.pt")
