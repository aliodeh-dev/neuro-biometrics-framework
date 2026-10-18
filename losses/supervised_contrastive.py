"""
Supervised Contrastive Learning Loss Module.
Implements:
  - Supervised Contrastive Loss (SupConLoss) as formulated by Khosla et al. (NeurIPS 2020).
  - Joint SupCon + Cross-Entropy Loss for multi-task metric and semantic regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    Pulls embeddings of the same subject (even across different temporal sessions) together,
    while pushing embeddings of different subjects apart on the unit hypersphere.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            projections: Tensor of shape (Batch, Projection_Dim), L2-normalized.
            labels: Tensor of shape (Batch,) containing subject IDs.
        Returns:
            loss: Scalar tensor
        """
        device = projections.device
        batch_size = projections.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        # Binary mask of same-subject pairs: mask[i, j] = 1 if y_i == y_j
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute cosine similarity matrix: (Batch, Batch)
        # Note: Since projections are L2-normalized, dot product equals cosine similarity
        anchor_dot_contrast = torch.div(
            torch.matmul(projections, projections.T),
            self.temperature
        )

        # For numerical stability, subtract max per row
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast (diagonal elements)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute denominator: sum over all non-self elements
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        # Positive pairs count per anchor
        pos_counts = mask.sum(1)
        
        # Avoid division by zero for anchors with no other positives in batch
        valid_anchors = pos_counts > 0
        if valid_anchors.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(1)[valid_anchors] / pos_counts[valid_anchors]
        loss = -mean_log_prob_pos.mean()

        return loss

class JointBiometricLoss(nn.Module):
    """
    Joint multi-task objective:
      L_total = lambda * L_SupCon + (1 - lambda) * L_CE
    """
    def __init__(self, temperature: float = 0.07, contrastive_weight: float = 0.7):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.supcon_loss = SupervisedContrastiveLoss(temperature=temperature)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        projections: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        loss_supcon = self.supcon_loss(projections, labels)
        loss_ce = self.ce_loss(logits, labels)
        total_loss = self.contrastive_weight * loss_supcon + (1.0 - self.contrastive_weight) * loss_ce
        return total_loss, loss_supcon, loss_ce
