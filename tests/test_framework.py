"""
Automated Test Suite for Neuro-Biometrics Framework.
Tests:
  1. Preprocessing pipeline & zero-phase filtering
  2. Sliding window segmentation consistency
  3. Synthetic multi-session EEG generation
  4. WearableBrainprintNet forward pass and hyperspherical normalization
  5. Supervised Contrastive Loss stability & gradient backpropagation
  6. Biometric Metrics Evaluator (EER, AUC-ROC, Decidability)
  7. Replay Attack Simulation & Liveness Verification
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import numpy as np
import torch

from config.settings import FrameworkConfig, PreprocessingConfig, ModelConfig, SecurityConfig
from data.preprocessing import EEGPreprocessor
from data.segmentation import EEGSegmenter
from data.synthetic_generator import SyntheticEEGGenerator
from data.dataset import EEGMultiSessionDataset, BalancedSessionBatchSampler
from models.brainprint_net import WearableBrainprintNet
from models.baselines import Standard1DCNNBaseline, EEGNetBaseline, ClassicalPSDBaseline
from losses.supervised_contrastive import SupervisedContrastiveLoss, JointBiometricLoss
from security.replay_attack_simulator import ReplayAttackSimulator
from security.liveness_verifier import LivenessVerifier
from evaluation.metrics import BiometricMetricsEvaluator

class TestNeuroBiometricsFramework(unittest.TestCase):
    def setUp(self):
        self.config = FrameworkConfig(num_subjects=4, num_sessions=2, records_per_session_sec=10.0)
        self.fs = self.config.preprocessing.sampling_rate

    def test_01_preprocessing_pipeline(self):
        preprocessor = EEGPreprocessor(self.config.preprocessing)
        raw_eeg = np.random.randn(8, 1000).astype(np.float32)
        clean_eeg = preprocessor.process_raw_stream(raw_eeg)

        self.assertEqual(clean_eeg.shape, raw_eeg.shape)
        # Check channel normalization: mean ~ 0, std ~ 1
        for ch in range(8):
            self.assertAlmostEqual(np.mean(clean_eeg[ch]), 0.0, places=1)
            self.assertAlmostEqual(np.std(clean_eeg[ch]), 1.0, places=1)

    def test_02_segmentation(self):
        segmenter = EEGSegmenter(self.config.preprocessing)
        # 10 seconds of signal = 2500 samples
        signal = np.random.randn(8, 2500).astype(np.float32)
        epochs, meta = segmenter.segment_signal(signal, subject_id=0, session_id=0)

        window_samples = int(self.config.preprocessing.window_duration_sec * self.fs) # 500
        self.assertEqual(epochs.shape[1], 8)
        self.assertEqual(epochs.shape[2], window_samples)
        self.assertEqual(len(epochs), len(meta))
        self.assertEqual(meta[0]["subject_id"], 0)

    def test_03_synthetic_generator(self):
        generator = SyntheticEEGGenerator(self.config)
        session_signal = generator.generate_session(subject_id=0, session_id=0)
        
        expected_samples = int(self.config.records_per_session_sec * self.fs)
        self.assertEqual(session_signal.shape, (8, expected_samples))
        self.assertFalse(np.isnan(session_signal).any())

    def test_04_model_forward_and_hypersphere(self):
        model_cfg = ModelConfig(num_channels=8, num_time_samples=500, num_classes=4)
        model = WearableBrainprintNet(model_cfg)
        model.eval()

        dummy_x = torch.randn(4, 8, 500)
        out = model(dummy_x, return_projection=True, return_logits=True)

        self.assertIn("embedding", out)
        self.assertIn("projection", out)
        self.assertIn("logits", out)

        proj = out["projection"]
        self.assertEqual(proj.shape, (4, model_cfg.projection_dim))

        # Check L2-normalization on unit hypersphere: ||z||_2 == 1.0
        norms = torch.norm(proj, p=2, dim=1)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=4)

    def test_05_supervised_contrastive_loss(self):
        loss_fn = SupervisedContrastiveLoss(temperature=0.07)
        
        # Batch of 4 samples: 2 from class 0, 2 from class 1
        projections = torch.randn(4, 64)
        projections = torch.nn.functional.normalize(projections, p=2, dim=1)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        projections.requires_grad = True
        loss = loss_fn(projections, labels)

        self.assertGreater(loss.item(), 0.0)
        # Test backprop
        loss.backward()
        self.assertIsNotNone(projections.grad)

    def test_06_biometric_metrics(self):
        evaluator = BiometricMetricsEvaluator()
        
        # Mock high separation: genuine scores near 0.9, impostor scores near 0.1
        gen_scores = np.random.normal(0.85, 0.05, 500)
        imp_scores = np.random.normal(0.15, 0.05, 1000)

        eer, eer_th, _, _ = evaluator.compute_eer(gen_scores, imp_scores)
        auc_val, _, _ = evaluator.compute_roc_auc(gen_scores, imp_scores)
        d_prime = evaluator.compute_d_prime(gen_scores, imp_scores)

        self.assertLess(eer, 0.05) # EER should be very low
        self.assertGreater(auc_val, 0.95)
        self.assertGreater(d_prime, 4.0)

    def test_07_replay_attacks_and_liveness(self):
        sec_cfg = SecurityConfig()
        attack_sim = ReplayAttackSimulator(sec_cfg)
        verifier = LivenessVerifier(sec_cfg, fs=self.fs)

        legit_epoch = np.random.randn(8, 500).astype(np.float32)

        # 1. Exact replay attack
        replayed = attack_sim.exact_replay(legit_epoch)
        self.assertTrue(np.array_equal(legit_epoch, replayed))

        # 2. Liveness evaluation
        is_live, score, audit = verifier.evaluate_liveness(legit_epoch)
        self.assertIn("spectral_entropy", audit)
        self.assertIn("liveness_score", audit)

if __name__ == "__main__":
    unittest.main()
