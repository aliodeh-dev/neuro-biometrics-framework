"""
Signal Preprocessing Module for Wearable EEG Neuro-Biometrics.
Implements:
  - Zero-phase Butterworth Bandpass Filtering (0.5 - 45.0 Hz)
  - IIR Notch Filtering (50.0 Hz / 60.0 Hz) for Powerline Interference
  - Baseline Drift Correction & Linear Detrending
  - Robust Outlier / Artifact Attenuation (Z-score Clipping)
  - Temporal Per-Channel Standardization
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, detrend
from typing import Optional, Tuple
from config.settings import PreprocessingConfig

class EEGPreprocessor:
    """
    Automated EEG signal cleansing and noise attenuation pipeline.
    Ensures zero-phase distortion so that phase-based biometrics are preserved.
    """
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.fs = self.config.sampling_rate
        self.lowcut = self.config.lowcut
        self.highcut = self.config.highcut
        self.notch_freq = self.config.notch_freq
        self.notch_q = self.config.notch_q
        self.order = self.config.filter_order
        self.artifact_thresh = self.config.artifact_std_threshold

        # Precompute Second-Order Sections (SOS) for numerical stability
        self._sos_bandpass = butter(
            N=self.order,
            Wn=[self.lowcut, self.highcut],
            btype='bandpass',
            fs=self.fs,
            output='sos'
        )

        # Precompute Notch filter coefficients
        self._b_notch, self._a_notch = iirnotch(
            w0=self.notch_freq,
            Q=self.notch_q,
            fs=self.fs
        )

    def bandpass_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Applies zero-phase Butterworth bandpass filter.
        Args:
            signal: np.ndarray of shape (channels, samples) or (samples,)
        Returns:
            Filtered signal of same shape
        """
        axis = -1
        # sosfiltfilt applies forward-backward filtering to avoid phase distortion
        return sosfiltfilt(self._sos_bandpass, signal, axis=axis)

    def notch_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        Applies zero-phase IIR notch filter to remove line noise (50/60 Hz).
        """
        axis = -1
        return filtfilt(self._b_notch, self._a_notch, signal, axis=axis)

    def remove_baseline_drift(self, signal: np.ndarray) -> np.ndarray:
        """
        Removes low-frequency baseline drift and DC offset via linear detrending.
        """
        return detrend(signal, axis=-1, type='linear')

    def attenuate_artifacts(self, signal: np.ndarray) -> np.ndarray:
        """
        Soft-clips extreme physiological artifacts (eye-blinks, muscle clenching)
        beyond artifact_thresh standard deviations while preserving wave dynamics.
        """
        mean = np.mean(signal, axis=-1, keepdims=True)
        std = np.std(signal, axis=-1, keepdims=True) + 1e-8
        z_score = (signal - mean) / std

        clipped_z = np.clip(z_score, -self.artifact_thresh, self.artifact_thresh)
        return clipped_z * std + mean

    def normalize(self, signal: np.ndarray) -> np.ndarray:
        """
        Channel-wise z-score normalization across temporal axis.
        """
        mean = np.mean(signal, axis=-1, keepdims=True)
        std = np.std(signal, axis=-1, keepdims=True) + 1e-8
        return (signal - mean) / std

    def process_raw_stream(self, raw_signal: np.ndarray) -> np.ndarray:
        """
        Complete end-to-end preprocessing pipeline:
          1. Baseline Detrending
          2. Bandpass Filtering (0.5 - 45 Hz)
          3. Notch Filtering (50/60 Hz)
          4. Artifact Attenuation
          5. Channel-wise Standardization
        """
        s = self.remove_baseline_drift(raw_signal)
        s = self.bandpass_filter(s)
        s = self.notch_filter(s)
        s = self.attenuate_artifacts(s)
        s = self.normalize(s)
        return s
