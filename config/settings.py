"""
Configuration settings for Neuro-Biometrics Brainprint Authentication Framework.
Defines parameters for signal preprocessing, dataset generation, model architecture,
contrastive learning, and evaluation protocols.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class PreprocessingConfig:
    sampling_rate: int = 250               # Wearable EEG sampling rate in Hz (e.g., Emotiv, OpenBCI)
    lowcut: float = 0.5                    # High-pass cutoff frequency in Hz (removes DC drift)
    highcut: float = 45.0                  # Low-pass cutoff frequency in Hz (removes high-frequency EMG)
    notch_freq: float = 50.0               # Powerline interference notch frequency in Hz (50Hz / 60Hz)
    notch_q: float = 30.0                  # Quality factor for notch filter
    filter_order: int = 4                  # Butterworth bandpass filter order
    window_duration_sec: float = 2.0       # Epoch temporal window length in seconds
    window_stride_sec: float = 1.0         # Sliding window stride in seconds (50% overlap)
    artifact_std_threshold: float = 5.0    # Z-score clipping threshold for eye/movement burst suppression
    channels: List[str] = field(default_factory=lambda: [
        "AF3", "AF4", "F3", "F4", "T7", "T8", "P7", "P8"
    ])  # 8-channel wearable layout covering frontal, temporal, and parietal lobes

@dataclass
class ModelConfig:
    num_channels: int = 8
    num_time_samples: int = 500            # sampling_rate * window_duration_sec (250 * 2.0 = 500)
    temporal_filters: int = 16             # Number of temporal 1D convolution kernels
    temporal_kernel_size: int = 65         # Temporal kernel size (approx. sampling_rate / 4)
    spatial_depth_multiplier: int = 2      # Depthwise spatial filter multiplier
    attention_heads: int = 4               # Multi-head temporal self-attention heads
    attention_dropout: float = 0.1
    embedding_dim: int = 128               # Base latent feature representation dimension
    projection_dim: int = 64               # Contrastive projection head dimension (unit hypersphere)
    num_classes: int = 20                  # Number of enrolled subjects in the cohort

@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 35
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 0.07              # Supervised Contrastive Loss temperature parameter tau
    contrastive_weight: float = 0.7        # Weight lambda for SupCon loss in joint objective
    cross_entropy_weight: float = 0.3      # Weight (1 - lambda) for Cross-Entropy loss
    device: str = "cpu"                    # "cuda" if GPU is available else "cpu"
    seed: int = 42

@dataclass
class SecurityConfig:
    cosine_similarity_threshold: float = 0.75  # Default verification threshold on unit hypersphere
    liveness_phase_coherence_min: float = 0.25 # Minimum inter-frequency phase coherence for liveness
    liveness_entropy_bounds: Tuple[float, float] = (1.2, 4.5)  # Permissible spectral entropy bounds
    replay_jitter_std: float = 0.05            # Synthetic temporal jitter std in replay simulation
    adversarial_noise_snr_db: float = 15.0     # Signal-to-noise ratio for noisy replay attacks

@dataclass
class FrameworkConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    num_subjects: int = 20
    num_sessions: int = 4                 # Temporal sessions per subject (for cross-session LOSO-CV)
    records_per_session_sec: float = 60.0 # Length of continuous recording per session in seconds
