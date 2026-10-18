"""
End-to-End Wearable Brainprint Neural Network (WearableBrainprintNet).
Combines:
  1. Spatial-Temporal 1D-CNN (Local oscillatory & scalp montage filtering)
  2. Multi-Head Temporal Self-Attention (Global temporal phase modeling)
  3. Non-linear Metric Projection Head (Hyperspherical embedding for SupCon loss)
  4. Linear Classification Head (Auxiliary classification and closed-set verification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.settings import ModelConfig
from .spatial_temporal_cnn import SpatialTemporalCNNBlock
from .temporal_attention import TemporalSelfAttention

class WearableBrainprintNet(nn.Module):
    """
    Unified neuro-biometric deep neural network.
    Designed for high-accuracy, spoof-resistant authentication on low-cost wearable EEG.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Front-end: Spatial-Temporal Convolutional feature extractor
        self.cnn_frontend = SpatialTemporalCNNBlock(config)
        cnn_out_dim = config.temporal_filters * config.spatial_depth_multiplier # e.g. 16 * 2 = 32

        # Linear projection to attention embedding dim
        self.adapter = nn.Linear(cnn_out_dim, config.embedding_dim)

        # Backbone: Multi-Head Temporal Self-Attention
        self.temporal_attention = TemporalSelfAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout
        )

        # Global temporal pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Metric Projection Head (Maps to unit hypersphere for Supervised Contrastive Loss)
        self.projection_head = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.BatchNorm1d(config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.projection_dim)
        )

        # Supervised Classification Head (For closed-set or auxiliary cross-entropy)
        self.classifier_head = nn.Linear(config.embedding_dim, config.num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts deep latent temporal representation h prior to projection.
        Args:
            x: Tensor of shape (Batch, Channels, TimeSamples)
        Returns:
            h: Tensor of shape (Batch, embedding_dim)
        """
        # 1. Spatial-temporal convs -> (Batch, Seq_Len, Cnn_Out_Dim)
        cnn_feats = self.cnn_frontend(x)

        # 2. Linear projection to attention dim -> (Batch, Seq_Len, Embedding_Dim)
        attn_in = self.adapter(cnn_feats)

        # 3. Temporal Self-Attention -> (Batch, Seq_Len, Embedding_Dim)
        attn_out = self.temporal_attention(attn_in)

        # 4. Global Average Pooling over time -> (Batch, Embedding_Dim)
        # Permute to (Batch, Embedding_Dim, Seq_Len) for 1D pooling
        pooled = self.pool(attn_out.permute(0, 2, 1)).squeeze(-1)
        return pooled

    def forward(
        self,
        x: torch.Tensor,
        return_projection: bool = True,
        return_logits: bool = True
    ):
        """
        Forward pass yielding normalized projection vectors and/or classification logits.
        """
        h = self.extract_features(x)

        res = {"embedding": h}

        if return_projection:
            # Map through projection head and normalize onto unit hypersphere: ||z||_2 = 1
            z = self.projection_head(h)
            z_normalized = F.normalize(z, p=2, dim=1)
            res["projection"] = z_normalized

        if return_logits:
            logits = self.classifier_head(h)
            res["logits"] = logits

        return res
