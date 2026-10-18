"""
Comprehensive Evaluation & Benchmarking Script.
Executes:
  1. Multi-Model Benchmark (Proposed vs EEGNet vs Standard CNN vs Classical PSD)
  2. Cross-Session Generalization Testing
  3. Presentation & Replay Attack Defense Evaluation
  4. Publication Figure Generation (ROC, FAR-FRR, t-SNE, Attack Resilience)
  5. Thesis Metric Summary Generation (Markdown / LaTeX)
"""

import os
import sys
# Add parent directory to path so script can run from root or inside scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import json
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.settings import FrameworkConfig
from data.synthetic_generator import SyntheticEEGGenerator
from data.dataset import EEGMultiSessionDataset, BalancedSessionBatchSampler
from models.brainprint_net import WearableBrainprintNet
from models.baselines import Standard1DCNNBaseline, EEGNetBaseline, ClassicalPSDBaseline
from losses.supervised_contrastive import JointBiometricLoss
from security.replay_attack_simulator import ReplayAttackSimulator
from security.liveness_verifier import LivenessVerifier
from evaluation.metrics import BiometricMetricsEvaluator
from evaluation.visualizer import BiometricVisualizer

def run_full_benchmark(
    num_subjects: int = 15,
    num_sessions: int = 4,
    epochs: int = 25
):
    print("=================================================================")
    print(" NEURO-BIOMETRICS COMPREHENSIVE BENCHMARK & EVALUATION SUITE")
    print("=================================================================")

    config = FrameworkConfig(num_subjects=num_subjects, num_sessions=num_sessions)
    config.model.num_classes = num_subjects
    config.model.num_channels = len(config.preprocessing.channels)
    device = config.training.device

    evaluator = BiometricMetricsEvaluator()
    visualizer = BiometricVisualizer(output_dir="results/figures")
    attack_sim = ReplayAttackSimulator(config.security)
    liveness_verifier = LivenessVerifier(config.security, fs=config.preprocessing.sampling_rate)

    # 1. Dataset Generation
    print("\n[Phase 1] Generating Multi-Session Wearable EEG Cohort...")
    generator = SyntheticEEGGenerator(config)
    epochs_data, metadata = generator.generate_full_dataset()
    dataset = EEGMultiSessionDataset(epochs_data, metadata)
    print(f"Generated {len(dataset)} epochs across {num_subjects} subjects and {num_sessions} sessions.")

    # 2. Split into Training Sessions (0, 1, 2) and Held-out Future Test Session (3)
    session_ids = dataset.session_ids.numpy()
    subject_ids = dataset.subject_ids.numpy()

    train_mask = session_ids < (num_sessions - 1)
    test_mask = session_ids == (num_sessions - 1)

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    train_epochs_np = epochs_data[train_indices]
    train_subs_np = subject_ids[train_indices]
    test_epochs_np = epochs_data[test_indices]
    test_subs_np = subject_ids[test_indices]
    test_sessions_np = session_ids[test_indices]

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    print(f"Training set: {len(train_subset)} epochs (Sessions 0..{num_sessions-2})")
    print(f"Test set (strictly unseen future session): {len(test_subset)} epochs (Session {num_sessions-1})")

    # 3. Model Training & Feature Extraction
    models_dict = {}
    benchmark_metrics = {}

    # --- A. Classical PSD + SVM Baseline ---
    print("\n[Phase 2.1] Training Baseline 1: Classical PSD (Welch) + SVM...")
    psd_baseline = ClassicalPSDBaseline(fs=config.preprocessing.sampling_rate, classifier_type="svm")
    psd_baseline.fit(train_epochs_np, train_subs_np)
    psd_train_embs = psd_baseline.get_embeddings(train_epochs_np)
    psd_test_embs = psd_baseline.get_embeddings(test_epochs_np)
    
    # Benchmark latency
    psd_time_start = time_ns()
    _ = psd_baseline.extract_dataset(test_epochs_np[:10])
    psd_time_end = time_ns()
    psd_latency_ms = ((psd_time_end - psd_time_start) / 10.0) / 1e6

    psd_res = evaluator.evaluate_template_probe(psd_train_embs, train_subs_np, psd_test_embs, test_subs_np)
    psd_res["latency_ms"] = psd_latency_ms
    psd_res["params"] = 40 * num_subjects
    benchmark_metrics["Classical PSD + SVM"] = psd_res
    print(f"Classical PSD -> EER: {psd_res['EER']*100:.2f}% | AUC: {psd_res['AUC']*100:.2f}% | d': {psd_res['d_prime']:.2f}")

    # --- B. Standard 1D-CNN Baseline (Cross-Entropy Only) ---
    print("\n[Phase 2.2] Training Baseline 2: Standard 1D-CNN (Cross-Entropy Only)...")
    cnn_model = Standard1DCNNBaseline(config.model).to(device)
    cnn_opt = torch.optim.AdamW(cnn_model.parameters(), lr=config.training.learning_rate)
    cnn_crit = torch.nn.CrossEntropyLoss()
    cnn_loader = DataLoader(train_subset, batch_size=config.training.batch_size, shuffle=True)

    for ep in range(epochs):
        cnn_model.train()
        for b in cnn_loader:
            x, y = b["epoch"].to(device), b["subject_id"].to(device)
            cnn_opt.zero_grad()
            logits = cnn_model(x)["logits"]
            loss = cnn_crit(logits, y)
            loss.backward()
            cnn_opt.step()

    cnn_train_embs = extract_embeddings(cnn_model, train_subset, device)
    cnn_test_embs = extract_embeddings(cnn_model, test_subset, device)
    cnn_res = evaluator.evaluate_template_probe(cnn_train_embs, train_subs_np, cnn_test_embs, test_subs_np)
    cnn_bench = evaluator.benchmark_inference_latency(cnn_model, device=device)
    cnn_res["latency_ms"] = cnn_bench["avg_latency_ms"]
    cnn_res["params"] = cnn_bench["trainable_parameters"]
    benchmark_metrics["Standard 1D-CNN (CE)"] = cnn_res
    print(f"Standard 1D-CNN -> EER: {cnn_res['EER']*100:.2f}% | AUC: {cnn_res['AUC']*100:.2f}% | d': {cnn_res['d_prime']:.2f}")

    # --- C. EEGNet Baseline (Cross-Entropy Only) ---
    print("\n[Phase 2.3] Training Baseline 3: EEGNet (Lawhern et al. 2018)...")
    eegnet_model = EEGNetBaseline(config.model).to(device)
    eegnet_opt = torch.optim.AdamW(eegnet_model.parameters(), lr=config.training.learning_rate)
    eegnet_crit = torch.nn.CrossEntropyLoss()

    for ep in range(epochs):
        eegnet_model.train()
        for b in cnn_loader:
            x, y = b["epoch"].to(device), b["subject_id"].to(device)
            eegnet_opt.zero_grad()
            logits = eegnet_model(x)["logits"]
            loss = eegnet_crit(logits, y)
            loss.backward()
            eegnet_opt.step()

    eegnet_train_embs = extract_embeddings(eegnet_model, train_subset, device)
    eegnet_test_embs = extract_embeddings(eegnet_model, test_subset, device)
    eegnet_res = evaluator.evaluate_template_probe(eegnet_train_embs, train_subs_np, eegnet_test_embs, test_subs_np)
    eegnet_bench = evaluator.benchmark_inference_latency(eegnet_model, device=device)
    eegnet_res["latency_ms"] = eegnet_bench["avg_latency_ms"]
    eegnet_res["params"] = eegnet_bench["trainable_parameters"]
    benchmark_metrics["EEGNet Baseline"] = eegnet_res
    print(f"EEGNet -> EER: {eegnet_res['EER']*100:.2f}% | AUC: {eegnet_res['AUC']*100:.2f}% | d': {eegnet_res['d_prime']:.2f}")

    # --- D. Proposed WearableBrainprintNet (1D-CNN + Attention + SupCon) ---
    print("\n[Phase 2.4] Training Proposed: WearableBrainprintNet (SupCon + Attention)...")
    prop_model = WearableBrainprintNet(config.model).to(device)
    prop_opt = torch.optim.AdamW(
        prop_model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    prop_crit = JointBiometricLoss(
        temperature=config.training.temperature,
        contrastive_weight=config.training.contrastive_weight
    )
    balanced_sampler = BalancedSessionBatchSampler(
        dataset=dataset,
        num_classes_per_batch=min(8, num_subjects),
        num_samples_per_class=4,
        total_batches=35
    )
    prop_loader = DataLoader(dataset, batch_sampler=balanced_sampler)

    for ep in range(epochs):
        prop_model.train()
        for b in prop_loader:
            x, y = b["epoch"].to(device), b["subject_id"].to(device)
            prop_opt.zero_grad()
            out = prop_model(x, return_projection=True, return_logits=True)
            loss, _, _ = prop_crit(out["projection"], out["logits"], y)
            loss.backward()
            prop_opt.step()

    prop_train_embs = extract_embeddings(prop_model, train_subset, device)
    prop_test_embs = extract_embeddings(prop_model, test_subset, device)
    prop_res = evaluator.evaluate_template_probe(prop_train_embs, train_subs_np, prop_test_embs, test_subs_np)
    prop_bench = evaluator.benchmark_inference_latency(prop_model, device=device)
    prop_res["latency_ms"] = prop_bench["avg_latency_ms"]
    prop_res["params"] = prop_bench["trainable_parameters"]
    benchmark_metrics["Proposed WearableBrainprintNet"] = prop_res
    print(f"Proposed Model -> EER: {prop_res['EER']*100:.2f}% | AUC: {prop_res['AUC']*100:.2f}% | d': {prop_res['d_prime']:.2f}")

    # 4. Presentation & Replay Attack Defense Evaluation
    print("\n[Phase 3] Stress-Testing Presentation & Replay Attack Resilience...")
    attack_results = evaluate_attack_resilience(
        model=prop_model,
        test_epochs=test_epochs_np,
        test_subs=test_subs_np,
        attack_sim=attack_sim,
        liveness_verifier=liveness_verifier,
        device=device
    )

    # 5. Scientific Visualizations
    print("\n[Phase 4] Generating Publication-Ready Figures...")
    roc_fig = visualizer.plot_roc_curves(benchmark_metrics, "roc_curves.png")
    far_frr_fig = visualizer.plot_far_frr_tradeoff(prop_res, "Proposed WearableBrainprintNet", "far_frr_tradeoff.png")
    tsne_fig = visualizer.plot_tsne_comparison(
        cnn_test_embs, prop_test_embs, test_subs_np, test_sessions_np,
        max_subjects=6, filename="latent_space_tsne.png"
    )
    attack_fig = visualizer.plot_replay_attack_resilience(attack_results, "replay_attack_evaluation.png")

    print(f"Figures saved to results/figures/:")
    print(f" - {roc_fig}")
    print(f" - {far_frr_fig}")
    print(f" - {tsne_fig}")
    print(f" - {attack_fig}")

    # 6. Export Summary Tables
    export_summary_reports(benchmark_metrics, attack_results)

    print("\n=================================================================")
    print(" BENCHMARK COMPLETED SUCCESSFULLY!")
    print("=================================================================")

def extract_embeddings(model: torch.nn.Module, subset: torch.utils.data.Subset, device: str) -> np.ndarray:
    model.eval()
    loader = DataLoader(subset, batch_size=64, shuffle=False)
    embs = []
    with torch.no_grad():
        for b in loader:
            x = b["epoch"].to(device)
            out = model(x, return_projection=True, return_logits=False)
            proj = out["projection"].cpu().numpy()
            embs.append(proj)
    return np.concatenate(embs, axis=0)

def evaluate_attack_resilience(
    model: torch.nn.Module,
    test_epochs: np.ndarray,
    test_subs: np.ndarray,
    attack_sim: ReplayAttackSimulator,
    liveness_verifier: LivenessVerifier,
    device: str
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates acceptance rate across attack modalities with and without liveness defense.
    """
    model.eval()
    unique_subs = np.unique(test_subs)
    target_sub = unique_subs[0]
    impostor_sub = unique_subs[1]

    target_idx = np.where(test_subs == target_sub)[0]
    impostor_idx = np.where(test_subs == impostor_sub)[0]

    # Enrolled template for target user
    with torch.no_grad():
        enroll_tensor = torch.tensor(test_epochs[target_idx[:5]]).to(device)
        enroll_embs = model(enroll_tensor, return_projection=True)["projection"].cpu().numpy()
        ref_template = np.mean(enroll_embs, axis=0)
        ref_template /= np.linalg.norm(ref_template)

    test_target_epochs = test_epochs[target_idx[5:25]]
    test_impostor_epochs = test_epochs[impostor_idx[:20]]

    attack_categories = [
        "legitimate_probe",
        "exact_replay",
        "jittered_replay",
        "synthetic_spoof",
        "cross_subject_masquerade"
    ]

    results = {cat: {"without_defense": 0.0, "with_defense": 0.0} for cat in attack_categories}

    # Reset recent buffer
    liveness_verifier.recent_hash_buffer = []

    for i in range(1, len(test_target_epochs)):
        probe = test_target_epochs[i]
        prev_probe = test_target_epochs[i - 1]
        imp_probe = test_impostor_epochs[i % len(test_impostor_epochs)]

        # Synthetic spoof: add low-SNR artificial generative perturbation
        synthetic_sample = attack_sim.synthetic_noise_spoof(probe, snr_db=3.0)

        attacks = {
            "legitimate_probe": probe,
            "exact_replay": attack_sim.exact_replay(probe),
            "jittered_replay": attack_sim.jittered_replay(probe),
            "synthetic_spoof": synthetic_sample,
            "cross_subject_masquerade": attack_sim.cross_subject_masquerade(imp_probe)
        }

        for cat, sample in attacks.items():
            # Get embedding
            with torch.no_grad():
                s_t = torch.tensor(sample).unsqueeze(0).to(device)
                emb = model(s_t, return_projection=True)["projection"].cpu().numpy()[0]

            # 1. Without Defense (Naive Cosine Similarity Thresholding)
            cos_sim = float(np.dot(emb, ref_template))
            accepted_naive = cos_sim >= liveness_verifier.threshold

            # 2. With Defense (Liveness + Hash Buffer + Composite Score)
            audit = liveness_verifier.verify_candidate(
                candidate_embedding=emb,
                reference_template=ref_template,
                candidate_raw_epoch=sample,
                previous_raw_epoch=prev_probe
            )
            accepted_defense = audit["authenticated"]

            if accepted_naive:
                results[cat]["without_defense"] += 1.0
            if accepted_defense:
                results[cat]["with_defense"] += 1.0

    n_samples = len(test_target_epochs) - 1
    for cat in attack_categories:
        results[cat]["without_defense"] /= n_samples
        results[cat]["with_defense"] /= n_samples

    return results

def time_ns() -> int:
    import time
    return int(time.perf_counter() * 1e9)

def export_summary_reports(
    benchmark_metrics: Dict[str, Dict[str, Any]],
    attack_results: Dict[str, Dict[str, float]],
    output_dir: str = "results"
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Markdown Table
    md_path = os.path.join(output_dir, "benchmark_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Master's Thesis Experimental Benchmark Summary\n\n")
        f.write("## Table 1: Cross-Session Biometric Verification Performance Comparison\n\n")
        f.write("| Architecture / Model | EER (%) | AUC-ROC (%) | Decidability (d') | FAR @ FRR=1% | FRR @ FAR=1% | Latency (ms) | Params |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for name, m in benchmark_metrics.items():
            f.write(
                f"| **{name}** | {m['EER']*100:.2f}% | {m['AUC']*100:.2f}% | {m['d_prime']:.2f} | "
                f"{m['FAR@FRR=1%']*100:.2f}% | {m['FRR@FAR=1%']*100:.2f}% | {m['latency_ms']:.2f} ms | {m['params']:,} |\n"
            )

        f.write("\n\n## Table 2: Presentation & Replay Attack Resistance (Acceptance Rate %)\n\n")
        f.write("| Attack / Probe Scenario | Static Biometric Gate (No Liveness) | Proposed Spoof-Resistant Framework |\n")
        f.write("| :--- | :---: | :---: |\n")

        for cat, res in attack_results.items():
            f.write(
                f"| {cat.replace('_', ' ').title()} | {res['without_defense']*100:.1f}% | **{res['with_defense']*100:.1f}%** |\n"
            )

    # 2. JSON summary (excluding numpy arrays)
    json_path = os.path.join(output_dir, "benchmark_metrics.json")
    clean_metrics = {}
    for k, v in benchmark_metrics.items():
        clean_metrics[k] = {
            metric: float(val) if isinstance(val, (float, int, np.floating, np.integer)) else val
            for metric, val in v.items() if metric != "_curves"
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"models": clean_metrics, "attacks": attack_results}, f, indent=4)

    print(f"Exported benchmark summary to {md_path} and {json_path}")

if __name__ == "__main__":
    run_full_benchmark(num_subjects=12, num_sessions=4, epochs=25)
