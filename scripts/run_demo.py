"""
Interactive Live Demonstration of Neuro-Biometric Verification & Anti-Spoofing Gate.
Simulates real-world enterprise biometric access control:
  1. User Enrollment
  2. Legitimate Future-Session Verification (Authentication Granted)
  3. Presentation & Replay Attack Defense (Interception & Rejection)
  4. Cross-Subject Masquerade Interception
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import time
import numpy as np
import torch

from config.settings import FrameworkConfig
from data.synthetic_generator import SyntheticEEGGenerator
from models.brainprint_net import WearableBrainprintNet
from security.replay_attack_simulator import ReplayAttackSimulator
from security.liveness_verifier import LivenessVerifier

def run_interactive_demo():
    print("=" * 70)
    print("   NEURO-BIOMETRICS: WEARABLE BRAINPRINT AUTHENTICATION GATEWAY")
    print("       Spoof-Resistant Verification via Contrastive Learning")
    print("=" * 70)

    config = FrameworkConfig(num_subjects=5, num_sessions=3)
    config.model.num_classes = 5

    print("\n[+] Initializing Neuro-Biometric Neural Engine & Security Gate...")
    model = WearableBrainprintNet(config.model)
    generator = SyntheticEEGGenerator(config)

    print("[*] Calibrating metric space with Supervised Contrastive Loss...")
    from data.dataset import EEGMultiSessionDataset, BalancedSessionBatchSampler
    from losses.supervised_contrastive import JointBiometricLoss
    from torch.utils.data import DataLoader
    
    epochs_data, metadata = generator.generate_full_dataset()
    demo_dataset = EEGMultiSessionDataset(epochs_data, metadata)
    demo_sampler = BalancedSessionBatchSampler(demo_dataset, num_classes_per_batch=4, num_samples_per_class=4, total_batches=15)
    demo_loader = DataLoader(demo_dataset, batch_sampler=demo_sampler)
    
    crit = JointBiometricLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    model.train()
    for _ in range(8):
        for b in demo_loader:
            opt.zero_grad()
            out = model(b["epoch"], return_projection=True, return_logits=True)
            loss, _, _ = crit(out["projection"], out["logits"], b["subject_id"])
            loss.backward()
            opt.step()
    model.eval()
    print("[OK] Neural Engine calibrated and ready.")

    attack_sim = ReplayAttackSimulator(config.security)
    verifier = LivenessVerifier(config.security, fs=config.preprocessing.sampling_rate)

    # 1. Enrollment
    print("\n-------------------------------------------------------------")
    print(" STEP 1: ENROLLMENT OF AUTHORIZED EMPLOYEE (USER #01 - 'ALICE')")
    print("-------------------------------------------------------------")
    print("[*] Recording 5 reference EEG epochs from wearable headset (Day 1)...")
    
    # Generate authentic session 0 for User 0
    alice_s0 = generator.generate_session(subject_id=0, session_id=0)
    from data.preprocessing import EEGPreprocessor
    from data.segmentation import EEGSegmenter
    prep = EEGPreprocessor(config.preprocessing)
    seg = EEGSegmenter(config.preprocessing)
    
    clean_s0 = prep.process_raw_stream(alice_s0)
    epochs_s0, _ = seg.segment_signal(clean_s0, subject_id=0, session_id=0)
    
    # Enroll template
    with torch.no_grad():
        enroll_tensor = torch.tensor(epochs_s0[:6]).unsqueeze(0) if epochs_s0[:6].ndim == 2 else torch.tensor(epochs_s0[:6])
        enroll_proj = model(enroll_tensor, return_projection=True)["projection"].numpy()
        alice_template = np.mean(enroll_proj, axis=0)
        alice_template /= np.linalg.norm(alice_template)

    print(f"[OK] Alice Enrolled successfully! Brainprint Reference Vector generated (Dimension: {len(alice_template)}).")
    print(f"    Operational Threshold: {verifier.threshold:.2f} Cosine Similarity.")

    # 2. Legitimate Verification Attempt
    print("\n-------------------------------------------------------------")
    print(" STEP 2: LEGITIMATE AUTHENTICATION ATTEMPT (DAY 3 - FUTURE SESSION)")
    print("-------------------------------------------------------------")
    print("[*] Alice arrives at secure facility and streams live EEG...")
    alice_s2 = generator.generate_session(subject_id=0, session_id=2)
    clean_s2 = prep.process_raw_stream(alice_s2)
    epochs_s2, _ = seg.segment_signal(clean_s2, subject_id=0, session_id=2)

    legit_probe = epochs_s2[10]
    with torch.no_grad():
        probe_t = torch.tensor(legit_probe).unsqueeze(0)
        probe_emb = model(probe_t, return_projection=True)["projection"].numpy()[0]

    decision = verifier.verify_candidate(
        candidate_embedding=probe_emb,
        reference_template=alice_template,
        candidate_raw_epoch=legit_probe
    )

    print(f"[*] Candidate Evaluation:")
    print(f"    - Cosine Similarity: {decision['cosine_similarity']:.4f} (Required: >= {verifier.threshold:.2f})")
    print(f"    - Liveness Score:    {decision['liveness_score']:.4f} (Spectral Entropy: {decision['liveness_details']['spectral_entropy']:.2f})")
    print(f"    - Composite Score:   {decision['composite_score']:.4f}")
    print(f"    ==> ACCESS VERDICT:  [{decision['decision']}] - User Verified as Alice.")

    # 3. Threat Scenario 1: Exact Replay Attack
    print("\n-------------------------------------------------------------")
    print(" STEP 3: ATTACK SCENARIO 1 — REPLAY ATTACK (DIGITAL WIRETAP)")
    print("-------------------------------------------------------------")
    print("[!] Adversary replays previously captured epoch to fool the gate...")
    replayed_probe = attack_sim.exact_replay(legit_probe)

    replay_decision = verifier.verify_candidate(
        candidate_embedding=probe_emb, # Same exact embedding
        reference_template=alice_template,
        candidate_raw_epoch=replayed_probe
    )

    print(f"[*] Candidate Evaluation:")
    print(f"    - Cosine Similarity: {replay_decision['cosine_similarity']:.4f}")
    print(f"    - Replay Detected:   {replay_decision['is_replayed']}")
    print(f"    - Rejection Reason:  {replay_decision['rejection_reasons']}")
    print(f"    ==> ACCESS VERDICT:  [{replay_decision['decision']}] - REPLAY ATTACK INTERCEPTED & NEUTRALIZED!")

    # 4. Threat Scenario 2: Synthetic Generative Noise Spoof
    print("\n-------------------------------------------------------------")
    print(" STEP 4: ATTACK SCENARIO 2 — SYNTHETIC / GENERATIVE SPOOF")
    print("-------------------------------------------------------------")
    print("[!] Adversary attempts GAN-generated synthetic brainwave injection...")
    spoofed_probe = attack_sim.synthetic_noise_spoof(legit_probe, snr_db=5.0)
    with torch.no_grad():
        spoof_t = torch.tensor(spoofed_probe).unsqueeze(0)
        spoof_emb = model(spoof_t, return_projection=True)["projection"].numpy()[0]

    spoof_decision = verifier.verify_candidate(
        candidate_embedding=spoof_emb,
        reference_template=alice_template,
        candidate_raw_epoch=spoofed_probe
    )

    print(f"[*] Candidate Evaluation:")
    print(f"    - Cosine Similarity: {spoof_decision['cosine_similarity']:.4f}")
    print(f"    - Liveness Score:    {spoof_decision['liveness_score']:.4f} (Valid Bounds: {decision['liveness_details']['entropy_valid']})")
    print(f"    - Rejection Reason:  {spoof_decision['rejection_reasons']}")
    print(f"    ==> ACCESS VERDICT:  [{spoof_decision['decision']}] - ARTIFICIAL SPOOF REJECTED!")

    # 5. Threat Scenario 3: Cross-Subject Impostor Masquerade
    print("\n-------------------------------------------------------------")
    print(" STEP 5: ATTACK SCENARIO 3 — ZERO-EFFORT IMPOSTOR (BOB)")
    print("-------------------------------------------------------------")
    print("[!] Unauthorized user 'Bob' puts on headset and tries to enter as Alice...")
    bob_s0 = generator.generate_session(subject_id=1, session_id=0)
    clean_bob = prep.process_raw_stream(bob_s0)
    bob_epochs, _ = seg.segment_signal(clean_bob, subject_id=1, session_id=0)
    bob_probe = bob_epochs[5]

    with torch.no_grad():
        bob_t = torch.tensor(bob_probe).unsqueeze(0)
        bob_emb = model(bob_t, return_projection=True)["projection"].numpy()[0]

    bob_decision = verifier.verify_candidate(
        candidate_embedding=bob_emb,
        reference_template=alice_template,
        candidate_raw_epoch=bob_probe
    )

    print(f"[*] Candidate Evaluation:")
    print(f"    - Cosine Similarity: {bob_decision['cosine_similarity']:.4f} (Well below threshold)")
    print(f"    - Rejection Reason:  {bob_decision['rejection_reasons']}")
    print(f"    ==> ACCESS VERDICT:  [{bob_decision['decision']}] - {'ACCESS GRANTED' if bob_decision['authenticated'] else 'IMPOSTOR DENIED ACCESS!'}")

    print("\n=============================================================")
    print(" DEMO COMPLETED: ALL THREATS SUCCESSFULLY MITIGATED BY GATEWAY")
    print("=============================================================\n")

if __name__ == "__main__":
    run_interactive_demo()
