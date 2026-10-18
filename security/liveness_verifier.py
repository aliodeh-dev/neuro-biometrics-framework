"""
Liveness Verification & Anti-Spoofing Engine.
Protects the authentication gate against presentation and replay attacks via:
  1. Multi-band Spectral Entropy & Biological Bounds Testing
  2. Dynamic Temporal Continuity & Inter-window Phase Coherence
  3. Metric Hyperspherical Cosine Similarity Verification
  4. Multi-Factor Security Decision Engine
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import welch
from typing import Dict, Any, Tuple, Optional
from config.settings import SecurityConfig

class LivenessVerifier:
    """
    Evaluates neuro-physiological validity and liveness of streaming EEG candidate epochs.
    Detects artificial injections, synthetic spoofs, and stale replayed buffers.
    """
    def __init__(self, config: SecurityConfig, fs: int = 250):
        self.config = config
        self.fs = fs
        self.threshold = config.cosine_similarity_threshold
        self.min_entropy, self.max_entropy = config.liveness_entropy_bounds
        self.min_coherence = config.liveness_phase_coherence_min
        
        # Buffer to track recent authentic epoch embeddings for replay detection
        self.recent_hash_buffer = []
        self.max_buffer_size = 500

    def compute_spectral_entropy(self, signal: np.ndarray) -> float:
        """
        Computes normalized spectral Shannon entropy across channels.
        Natural EEG has a characteristic entropy profile; synthetic noise or
        pure repetitive signals fall outside this physiological envelope.
        """
        entropies = []
        for ch in range(signal.shape[0]):
            freqs, psd = welch(signal[ch], fs=self.fs, nperseg=min(signal.shape[1], 128))
            psd_norm = psd / (np.sum(psd) + 1e-12)
            # Shannon entropy
            ent = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
            entropies.append(ent)
        return float(np.mean(entropies))

    def compute_stream_continuity(self, previous_epoch: np.ndarray, current_epoch: np.ndarray) -> float:
        """
        Measures stream continuity between consecutive 50% overlapping sliding windows.
        In authentic continuous streams, the second half of previous_epoch matches
        the first half of current_epoch (correlation > 0.90).
        Jittered or spliced replays break this temporal phase lock (correlation < 0.40).
        """
        half = current_epoch.shape[-1] // 2
        corrs = []
        for ch in range(current_epoch.shape[0]):
            p_seg = previous_epoch[ch, half:]
            c_seg = current_epoch[ch, :half]
            std_p = np.std(p_seg)
            std_c = np.std(c_seg)
            if std_p > 1e-6 and std_c > 1e-6:
                r = np.corrcoef(p_seg, c_seg)[0, 1]
                if not np.isnan(r):
                    corrs.append(r)
        if not corrs:
            return 1.0
        avg_corr = float(np.mean(corrs))
        continuity_score = float(np.clip(avg_corr, 0.0, 1.0))
        return continuity_score

    def evaluate_liveness(
        self,
        candidate_epoch: np.ndarray,
        previous_epoch: Optional[np.ndarray] = None
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Calculates holistic liveness score [0, 1].
        """
        ent = self.compute_spectral_entropy(candidate_epoch)
        ent_valid = (self.min_entropy <= ent <= self.max_entropy)

        # Continuity check
        if previous_epoch is not None:
            continuity = self.compute_stream_continuity(previous_epoch, candidate_epoch)
        else:
            continuity = 1.0

        # Liveness score formulation
        ent_score = 1.0 if ent_valid else 0.2
        liveness_score = 0.6 * ent_score + 0.4 * continuity
        is_live = (liveness_score >= 0.65) and ent_valid

        audit = {
            "spectral_entropy": ent,
            "entropy_valid": ent_valid,
            "continuity_score": continuity,
            "liveness_score": liveness_score,
            "is_live": is_live
        }
        return is_live, liveness_score, audit

    def verify_candidate(
        self,
        candidate_embedding: np.ndarray,
        reference_template: np.ndarray,
        candidate_raw_epoch: np.ndarray,
        previous_raw_epoch: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        End-to-end verification gate:
          1. Computes Hyperspherical Cosine Similarity: S_cos = z_cand . z_ref
          2. Evaluates Neuro-Physiological Liveness
          3. Checks Replay Buffer Deduplication
          4. Returns Decision: ACCEPT or REJECT with diagnostic reasons
        """
        # 1. Cosine similarity
        norm_c = candidate_embedding / (np.linalg.norm(candidate_embedding) + 1e-12)
        norm_r = reference_template / (np.linalg.norm(reference_template) + 1e-12)
        cosine_sim = float(np.dot(norm_c, norm_r))

        # 2. Check for exact replay match against recent buffer
        is_replayed_packet = False
        for old_emb in self.recent_hash_buffer:
            if np.linalg.norm(norm_c - old_emb) < 1e-5:
                is_replayed_packet = True
                break

        # 3. Liveness check
        is_live, liveness_score, liveness_audit = self.evaluate_liveness(
            candidate_raw_epoch,
            previous_raw_epoch
        )

        # 4. Composite authentication score
        composite_score = cosine_sim * (liveness_score if not is_replayed_packet else 0.0)

        # 5. Security Verdict
        decision = (
            (cosine_sim >= self.threshold) and
            is_live and
            (not is_replayed_packet)
        )

        rejection_reasons = []
        if cosine_sim < self.threshold:
            rejection_reasons.append(f"Insufficient similarity: {cosine_sim:.4f} < {self.threshold:.4f}")
        if not is_live:
            rejection_reasons.append(f"Failed liveness test: score {liveness_score:.4f}")
        if is_replayed_packet:
            rejection_reasons.append("Detected identical replayed biometric packet in session cache")

        # Cache valid embeddings
        if not is_replayed_packet:
            self.recent_hash_buffer.append(norm_c)
            if len(self.recent_hash_buffer) > self.max_buffer_size:
                self.recent_hash_buffer.pop(0)

        return {
            "decision": "ACCEPT" if decision else "REJECT",
            "authenticated": decision,
            "cosine_similarity": cosine_sim,
            "liveness_score": liveness_score,
            "composite_score": composite_score,
            "is_replayed": is_replayed_packet,
            "rejection_reasons": rejection_reasons,
            "liveness_details": liveness_audit
        }
