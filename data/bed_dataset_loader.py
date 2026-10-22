"""
BED (Biometric EEG Dataset) Loader Module.
Reference:
  Arnau-González, P., et al. (2021). "BED: A Dataset for EEG-Based Biometrics."
  IEEE Internet of Things Journal, 8(15), 12232-12242.
  Available on Zenodo: https://zenodo.org/records/4451416

Dataset Profile:
  - 21 healthy human subjects.
  - 3 independent recording sessions separated by ~1 week (ideal for LOSO-CV).
  - 14 channels recorded via Emotiv EPOC+ headset at 256 Hz.
  - Our wearable 8-channel montage (AF3, AF4, F3, F4, T7, T8, P7, P8) is an exact subset
    of the 14 EPOC+ channels!
"""

import os
import glob
import numpy as np
import scipy.signal as signal
from typing import Dict, List, Tuple, Optional
from config.settings import FrameworkConfig
from data.preprocessing import EEGPreprocessor
from data.segmentation import EEGSegmenter

class BEDDatasetLoader:
    """
    Loads and standardizes the real-world BED dataset for longitudinal consumer EEG biometrics.
    """
    # 14-Channel Emotiv EPOC+ montage in BED
    EMOTIV_14_CHANNELS = [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
        "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
    ]

    # Target 8-channel subset used across our framework
    TARGET_8_CHANNELS = [
        "AF3", "AF4", "F3", "F4", "T7", "T8", "P7", "P8"
    ]

    def __init__(self, data_dir: str, config: Optional[FrameworkConfig] = None):
        self.data_dir = data_dir
        self.config = config or FrameworkConfig()
        self.fs_bed = 256 # Native BED sampling frequency
        self.fs_target = self.config.preprocessing.sampling_rate # 250 Hz

        # Compute index mapping from 14 channels to our 8 channels
        self.channel_indices = [
            self.EMOTIV_14_CHANNELS.index(ch) for ch in self.TARGET_8_CHANNELS
        ]

        self.preprocessor = EEGPreprocessor(self.config.preprocessing)
        self.segmenter = EEGSegmenter(self.config.preprocessing)

    def extract_target_channels(self, raw_14ch_data: np.ndarray) -> np.ndarray:
        """
        Extracts the 8 wearable channels from 14-channel recording:
        Input shape: (14, Samples) -> Output shape: (8, Samples)
        """
        return raw_14ch_data[self.channel_indices, :]

    def resample_to_target(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Resamples from 256 Hz to 250 Hz if needed.
        """
        if self.fs_bed == self.fs_target:
            return signal_data
        num_target_samples = int(signal_data.shape[1] * (self.fs_target / self.fs_bed))
        return signal.resample(signal_data, num_target_samples, axis=1)

    def load_subject_session(
        self,
        subject_id: int,
        session_id: int
    ) -> Optional[np.ndarray]:
        """
        Loads raw recording for a given subject and session from data_dir.
        Supports CSV and NumPy format exported from Zenodo BED archives.
        """
        patterns = [
            os.path.join(self.data_dir, f"sub_{subject_id:02d}_ses_{session_id:02d}*.csv"),
            os.path.join(self.data_dir, f"s{subject_id}_s{session_id}*.npy"),
            os.path.join(self.data_dir, f"sub-{subject_id:02d}", f"ses-{session_id:02d}", "*.csv")
        ]

        matches = []
        for p in patterns:
            matches = glob.glob(p)
            if matches:
                break

        if not matches:
            return None

        file_path = matches[0]
        if file_path.endswith(".npy"):
            raw_data = np.load(file_path)
        else:
            raw_data = np.loadtxt(file_path, delimiter=",", skiprows=1).T

        # Extract 8 channels
        data_8ch = self.extract_target_channels(raw_data)
        # Resample to 250 Hz
        data_resampled = self.resample_to_target(data_8ch)
        return data_resampled.astype(np.float32)

    def load_full_cohort(
        self,
        num_subjects: int = 21,
        num_sessions: int = 3
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Loads, cleans, and segments all subjects and sessions in BED.
        Returns:
            epochs_array: (Total_Epochs, 8, 500)
            metadata: List of epoch metadata dicts
        """
        all_epochs = []
        all_meta = []

        for sub in range(num_subjects):
            for ses in range(num_sessions):
                raw = self.load_subject_session(sub, ses)
                if raw is None:
                    continue
                clean = self.preprocessor.process_raw_stream(raw)
                epochs, meta = self.segmenter.segment_signal(clean, sub, ses)
                all_epochs.append(epochs)
                all_meta.extend(meta)

        if not all_epochs:
            raise FileNotFoundError(
                f"No BED dataset recordings found in '{self.data_dir}'. "
                f"Download the dataset from Zenodo: https://zenodo.org/records/4451416"
            )

        return np.concatenate(all_epochs, axis=0), all_meta
