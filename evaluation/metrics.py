"""
Biometric Performance Metrics & Benchmarking Module.
Computes:
  - Genuine & Impostor Similarity Distributions
  - False Acceptance Rate (FAR) & False Rejection Rate (FRR)
  - Equal Error Rate (EER) with fine interpolation
  - Area Under the ROC Curve (AUC-ROC)
  - Decidability Index (d-prime)
  - Inference Latency and Model Parameter Footprint
"""

import time
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc
from typing import Dict, Any, Tuple, List

class BiometricMetricsEvaluator:
    """
    Standardized ISO/IEC 19795 biometric performance evaluator.
    """
    def __init__(self, thresholds: np.ndarray = None):
        if thresholds is None:
            self.thresholds = np.linspace(-1.0, 1.0, 2001)
        else:
            self.thresholds = thresholds

    def compute_similarity_scores(
        self,
        embeddings: np.ndarray,
        subject_ids: np.ndarray,
        session_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes all pairwise cosine similarities, partitioning into:
          - Genuine Scores: Same subject, strictly different temporal sessions
          - Impostor Scores: Different subjects
        Args:
            embeddings: (N, D) normalized vectors
            subject_ids: (N,) subject labels
            session_ids: (N,) session labels
        Returns:
            genuine_scores, impostor_scores
        """
        # Ensure L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        norm_embs = embeddings / norms

        sim_matrix = np.dot(norm_embs, norm_embs.T)
        n = len(subject_ids)

        genuine_scores = []
        impostor_scores = []

        has_multiple_sessions = len(np.unique(session_ids)) > 1

        # Pairwise iteration
        for i in range(n):
            for j in range(i + 1, n):
                score = sim_matrix[i, j]
                if subject_ids[i] == subject_ids[j]:
                    if has_multiple_sessions:
                        if session_ids[i] != session_ids[j]:
                            genuine_scores.append(score)
                    else:
                        genuine_scores.append(score)
                else:
                    impostor_scores.append(score)

        return np.array(genuine_scores, dtype=np.float32), np.array(impostor_scores, dtype=np.float32)

    def compute_template_probe_scores(
        self,
        enroll_embs: np.ndarray,
        enroll_subs: np.ndarray,
        probe_embs: np.ndarray,
        probe_subs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes template-to-probe verification scores (standard ISO biometric protocol):
          - Enrolled templates computed from training sessions
          - Test session probes matched against all enrolled templates
        """
        # Normalize
        enroll_norm = enroll_embs / (np.linalg.norm(enroll_embs, axis=1, keepdims=True) + 1e-12)
        probe_norm = probe_embs / (np.linalg.norm(probe_embs, axis=1, keepdims=True) + 1e-12)

        # Compute enrolled templates
        unique_subs = np.unique(enroll_subs)
        templates = {}
        for s in unique_subs:
            s_vecs = enroll_norm[enroll_subs == s]
            mean_vec = np.mean(s_vecs, axis=0)
            templates[s] = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)

        genuine_scores = []
        impostor_scores = []

        for p_idx, p_sub in enumerate(probe_subs):
            p_vec = probe_norm[p_idx]
            for t_sub, t_vec in templates.items():
                score = float(np.dot(p_vec, t_vec))
                if p_sub == t_sub:
                    genuine_scores.append(score)
                else:
                    impostor_scores.append(score)

        return np.array(genuine_scores, dtype=np.float32), np.array(impostor_scores, dtype=np.float32)

    def compute_eer(self, genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Computes the Equal Error Rate (EER) and the optimal operating threshold.
        Returns:
            eer: Equal Error Rate [0, 1]
            eer_threshold: Operating similarity threshold at EER
            far_curve: FAR values across thresholds
            frr_curve: FRR values across thresholds
        """
        num_gen = len(genuine_scores)
        num_imp = len(impostor_scores)

        if num_gen == 0 or num_imp == 0:
            return 0.5, 0.0, np.array([]), np.array([])

        far_curve = []
        frr_curve = []

        for th in self.thresholds:
            # Impostor accepted if score >= threshold
            far = np.sum(impostor_scores >= th) / num_imp
            # Genuine rejected if score < threshold
            frr = np.sum(genuine_scores < th) / num_gen

            far_curve.append(far)
            frr_curve.append(frr)

        far_arr = np.array(far_curve)
        frr_arr = np.array(frr_curve)

        # Find crossover where FAR == FRR
        diffs = np.abs(far_arr - frr_arr)
        min_idx = np.argmin(diffs)
        eer = float((far_arr[min_idx] + frr_arr[min_idx]) / 2.0)
        eer_th = float(self.thresholds[min_idx])

        return eer, eer_th, far_arr, frr_arr

    def compute_roc_auc(self, genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Computes Area Under ROC Curve (AUC) using standard sklearn metric.
        """
        y_true = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        y_scores = np.concatenate([genuine_scores, impostor_scores])

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = float(auc(fpr, tpr))
        return roc_auc, fpr, tpr

    def compute_d_prime(self, genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> float:
        """
        Computes biometric Decidability Index d':
          d' = |mu_gen - mu_imp| / sqrt(0.5 * (var_gen + var_imp))
        """
        mu_gen = np.mean(genuine_scores)
        mu_imp = np.mean(impostor_scores)
        var_gen = np.var(genuine_scores)
        var_imp = np.var(impostor_scores)

        denom = np.sqrt(0.5 * (var_gen + var_imp)) + 1e-12
        return float(np.abs(mu_gen - mu_imp) / denom)

    def benchmark_inference_latency(
        self,
        model: torch.nn.Module,
        input_shape: Tuple[int, int, int] = (1, 8, 500),
        num_runs: int = 100,
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Benchmarks inference speed (latency per epoch in ms) and model size.
        """
        model.eval()
        model.to(device)
        dummy_input = torch.randn(*input_shape).to(device)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

        # Benchmark
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input)
        end_time = time.perf_counter()

        total_time_sec = end_time - start_time
        avg_latency_ms = (total_time_sec / num_runs) * 1000.0

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "avg_latency_ms": float(avg_latency_ms),
            "throughput_epochs_per_sec": float(num_runs / total_time_sec),
            "trainable_parameters": int(num_params)
        }

    def _package_metrics(self, gen_scores: np.ndarray, imp_scores: np.ndarray) -> Dict[str, Any]:
        eer, eer_th, far_curve, frr_curve = self.compute_eer(gen_scores, imp_scores)
        roc_auc, fpr, tpr = self.compute_roc_auc(gen_scores, imp_scores)
        d_prime = self.compute_d_prime(gen_scores, imp_scores)

        if len(far_curve) > 0 and len(frr_curve) > 0:
            far_at_frr_1pct = float(far_curve[np.argmin(np.abs(frr_curve - 0.01))])
            frr_at_far_1pct = float(frr_curve[np.argmin(np.abs(far_curve - 0.01))])
        else:
            far_at_frr_1pct = 0.5
            frr_at_far_1pct = 0.5

        return {
            "EER": eer,
            "EER_threshold": eer_th,
            "AUC": roc_auc,
            "d_prime": d_prime,
            "FAR@FRR=1%": far_at_frr_1pct,
            "FRR@FAR=1%": frr_at_far_1pct,
            "genuine_mean": float(np.mean(gen_scores)) if len(gen_scores) > 0 else 0.0,
            "impostor_mean": float(np.mean(imp_scores)) if len(imp_scores) > 0 else 0.0,
            "genuine_std": float(np.std(gen_scores)) if len(gen_scores) > 0 else 0.0,
            "impostor_std": float(np.std(imp_scores)) if len(imp_scores) > 0 else 0.0,
            "num_genuine_pairs": len(gen_scores),
            "num_impostor_pairs": len(imp_scores),
            "_curves": {
                "thresholds": self.thresholds,
                "far": far_curve,
                "frr": frr_curve,
                "fpr": fpr,
                "tpr": tpr
            }
        }

    def evaluate_all(
        self,
        embeddings: np.ndarray,
        subject_ids: np.ndarray,
        session_ids: np.ndarray
    ) -> Dict[str, Any]:
        """
        Runs comprehensive biometric evaluation suite across pairwise scores.
        """
        gen_scores, imp_scores = self.compute_similarity_scores(embeddings, subject_ids, session_ids)
        return self._package_metrics(gen_scores, imp_scores)

    def evaluate_template_probe(
        self,
        enroll_embs: np.ndarray,
        enroll_subs: np.ndarray,
        probe_embs: np.ndarray,
        probe_subs: np.ndarray
    ) -> Dict[str, Any]:
        """
        Runs comprehensive biometric evaluation by matching test probes against enrolled templates.
        """
        gen_scores, imp_scores = self.compute_template_probe_scores(enroll_embs, enroll_subs, probe_embs, probe_subs)
        return self._package_metrics(gen_scores, imp_scores)
