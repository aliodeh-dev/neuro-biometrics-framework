"""
Presentation & Replay Attack Simulation Module.
Simulates realistic biometric cyber threats:
  1. Exact Digital Replay (Injection of captured authentic session epoch)
  2. Temporal Jittered & Resampled Replay (Warped replay attempting to evade temporal deduplication)
  3. Noisy Synthetic Impostor (Simulated GAN / generative synthetic brainwave injection)
  4. Cross-Subject Masquerade (Zero-effort impostor presenting their own EEG as victim's)
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple
from config.settings import SecurityConfig

class ReplayAttackSimulator:
    """
    Simulates malicious presentation and replay attacks against the neuro-biometric gate.
    """
    def __init__(self, config: SecurityConfig):
        self.config = config

    def exact_replay(self, legitimate_epoch: np.ndarray) -> np.ndarray:
        """
        Exact copy of previously recorded authentic EEG epoch.
        Shape: (channels, samples)
        """
        return legitimate_epoch.copy()

    def jittered_replay(self, legitimate_epoch: np.ndarray) -> np.ndarray:
        """
        Applies random temporal phase shift and micro-jitter to simulate an adversary
        manipulating recorded packets to bypass static caching.
        """
        channels, samples = legitimate_epoch.shape
        jitter_shift = int(np.random.normal(0, samples * self.config.replay_jitter_std))
        
        # Roll signal cyclically with small edge boundary smoothing
        warped = np.roll(legitimate_epoch, shift=jitter_shift, axis=-1)
        # Add micro-distortion
        noise = np.random.normal(0, 0.05, size=warped.shape)
        return (warped + noise).astype(np.float32)

    def synthetic_noise_spoof(self, legitimate_epoch: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
        """
        Simulates synthetic generative brainwave spoof with additive perturbation.
        """
        signal_power = np.mean(legitimate_epoch ** 2)
        noise_power = signal_power / (10.0 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), size=legitimate_epoch.shape)
        spoof = legitimate_epoch + noise
        return spoof.astype(np.float32)

    def cross_subject_masquerade(
        self,
        impostor_epoch: np.ndarray
    ) -> np.ndarray:
        """
        Zero-effort impostor masquerade: An unauthorized subject presents their real EEG.
        """
        return impostor_epoch.copy()

    def generate_attack_test_suite(
        self,
        target_subject_epoch: np.ndarray,
        impostor_subject_epoch: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary containing all attack variants for evaluation.
        """
        return {
            "legitimate_probe": target_subject_epoch,
            "exact_replay": self.exact_replay(target_subject_epoch),
            "jittered_replay": self.jittered_replay(target_subject_epoch),
            "synthetic_spoof": self.synthetic_noise_spoof(target_subject_epoch),
            "cross_subject_impostor": self.cross_subject_masquerade(impostor_subject_epoch)
        }
