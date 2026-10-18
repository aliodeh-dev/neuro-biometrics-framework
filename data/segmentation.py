"""
Sliding-Window Segmentation & Epoching Module.
Segments continuous multi-channel EEG signals into temporal windows suitable
for deep representation learning and continuous authentication.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from config.settings import PreprocessingConfig

class EEGSegmenter:
    """
    Slices continuous multi-channel EEG streams into overlapping temporal epochs.
    """
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.fs = config.sampling_rate
        self.window_samples = int(config.window_duration_sec * self.fs)
        self.stride_samples = int(config.window_stride_sec * self.fs)

    def segment_signal(
        self,
        signal: np.ndarray,
        subject_id: int,
        session_id: int
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Segments a continuous multi-channel array into epochs.
        Args:
            signal: np.ndarray of shape (channels, total_samples)
            subject_id: Unique identifier of subject
            session_id: Session identifier (0, 1, 2, ...)
        Returns:
            epochs: np.ndarray of shape (num_epochs, channels, window_samples)
            metadata: List of dicts storing metadata for each epoch
        """
        channels, total_samples = signal.shape
        if total_samples < self.window_samples:
            raise ValueError(
                f"Signal length ({total_samples}) is shorter than window length ({self.window_samples})"
            )

        epochs_list = []
        meta_list = []

        start = 0
        window_idx = 0
        while start + self.window_samples <= total_samples:
            end = start + self.window_samples
            epoch = signal[:, start:end]
            epochs_list.append(epoch)

            meta_list.append({
                "subject_id": subject_id,
                "session_id": session_id,
                "window_idx": window_idx,
                "start_sample": start,
                "end_sample": end,
                "start_time_sec": start / self.fs,
                "end_time_sec": end / self.fs
            })

            start += self.stride_samples
            window_idx += 1

        return np.array(epochs_list, dtype=np.float32), meta_list
