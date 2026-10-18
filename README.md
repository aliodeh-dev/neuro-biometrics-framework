# Neuro-Biometrics: A Spoof-Resistant Brainprint Authentication Framework Using Wearable EEG Signals and Contrastive Deep Learning

**إطار عمل القياسات الحيوية العصبية: مصادقة بصمة الدماغ المقاومة للهجمات التزييفية باستخدام إشارات التخطيط الدماغي المحمولة والتعلم التبايني العميق**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: 7 Passed](https://img.shields.io/badge/Tests-7%20Passed-brightgreen.svg)]()
[![Code Style: Modular](https://img.shields.io/badge/Code%20Style-Clean%20%26%20Modular-blueviolet.svg)]()

---

## 📌 Overview / نظرة عامة

This repository contains the complete open-source practical implementation of **Neuro-Biometrics**: an end-to-end, spoof-resistant brainprint authentication framework engineered for low-cost, 8-channel wearable Electroencephalography (EEG) headsets ($AF3, AF4, F3, F4, T7, T8, P7, P8$ at $250\text{ Hz}$).

### Core Challenges Addressed:
1. **Intra-Subject & Longitudinal Session Drift**: Biological and electrode impedance shifts across days are overcome via **Supervised Contrastive Representation Learning (SupCon)**, clustering same-user multi-session representations into invariant hyperspherical centroids.
2. **Presentation & Replay Attacks**: Adversaries attempting digital wiretapping, phase-jittered packet injection, or synthetic GAN brainwave synthesis are intercepted via an integrated multi-tier **Liveness Verification Gateway**.
3. **Edge Computational Feasibility**: Lightweight depthwise separable convolutions coupled with temporal self-attention achieve ultra-low inference latency ($2.09\text{ ms}$ on standard CPUs), enabling continuous background verification with $<0.3\%$ processor duty cycle.

---

## 🏛️ System Architecture / الهيكلية التقنية

```
                                      [ Wearable 8-Ch EEG Headset ]
                               (AF3, AF4, F3, F4, T7, T8, P7, P8 @ 250 Hz)
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Stage 1: Zero-Phase Preprocessing  |
                                  |   - Linear Baseline Detrending     |
                                  |   - Bandpass (0.5 - 45 Hz SOS)     |
                                  |   - IIR Notch (50/60 Hz, Q = 30)   |
                                  |   - Outlier Z-score Winsorization  |
                                  +------------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Stage 2: Sliding-Window Epoching   |
                                  |   - Window: 2.0s (500 samples)     |
                                  |   - Stride: 1.0s (50% Overlap)     |
                                  +------------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Stage 3: WearableBrainprintNet     |
                                  |   1. Temporal 1D Convolution       |
                                  |   2. Spatial Depthwise Conv (x8)   |
                                  |   3. Separable Convolutions        |
                                  |   4. Multi-Head Temporal Attention |
                                  |   5. Hyperspherical Projection     |
                                  +------------------------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
          +-------------------------------+                     +-------------------------------+
          | Supervised Contrastive Loss   |                     | Auxiliary Cross-Entropy       |
          | Pulls same-subject views      |                     | Enforces class discrimination |
          | onto unit hypersphere S^(63)  |                     | over base representation h    |
          +-------------------------------+                     +-------------------------------+
                         \                                                     /
                          +-------------------------+-------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Stage 4: Multi-Tiered Liveness Gate|
                                  |   Tier 1: Dynamic Stream Continuity|
                                  |   Tier 2: Spectral Shannon Entropy |
                                  |   Tier 3: Circular Replay Cache    |
                                  |   Tier 4: Cosine Metric Gate       |
                                  +------------------------------------+
```

---

## 📂 Project Directory Structure / هيكل المشروع

```
neuro_biometrics_framework/
├── config/
│   ├── settings.py           # Preprocessing, Model, Training & Security Dataclasses
│   └── __init__.py
├── data/
│   ├── preprocessing.py      # Zero-phase Butterworth, Notch, Detrending, Winsorizing
│   ├── segmentation.py       # Overlapping sliding-window epoching
│   ├── synthetic_generator.py# Physiologically realistic multi-session EEG simulator
│   ├── dataset.py            # PyTorch Dataset & BalancedSessionBatchSampler
│   └── __init__.py
├── models/
│   ├── spatial_temporal_cnn.py # 1D Temporal & Spatial Depthwise Convolution block
│   ├── temporal_attention.py # Multi-Head Temporal Self-Attention with Positional Encoding
│   ├── brainprint_net.py     # End-to-end WearableBrainprintNet architecture
│   ├── baselines.py          # Classical Welch PSD + SVM, Standard 1D-CNN, EEGNet
│   └── __init__.py
├── losses/
│   ├── supervised_contrastive.py # Supervised Contrastive Loss (SupCon) & Joint Loss
│   └── __init__.py
├── security/
│   ├── replay_attack_simulator.py # Exact replay, Jittered replay, Synthetic spoofing
│   ├── liveness_verifier.py  # Stream continuity, Spectral entropy, Hash deduplication
│   └── __init__.py
├── evaluation/
│   ├── metrics.py            # FAR, FRR, EER, AUC-ROC, d-prime, latency benchmarks
│   ├── loso_cv.py            # Leave-One-Session-Out Cross-Validation protocol manager
│   ├── visualizer.py         # Publication-grade plotting (ROC, FAR-FRR, t-SNE, Attacks)
│   └── __init__.py
├── scripts/
│   ├── train.py              # CLI training runner for deep networks and baselines
│   ├── evaluate.py           # Full benchmark suite & figure generator
│   └── run_demo.py           # Interactive biometric gateway demo
├── tests/
│   └── test_framework.py     # 7 comprehensive unit tests (100% pass rate)
├── results/
│   ├── benchmark_summary.md  # Generated evaluation tables (Markdown)
│   ├── benchmark_metrics.json# Generated metrics JSON
│   └── figures/              # Generated high-resolution figures:
│       ├── roc_curves.png
│       ├── far_frr_tradeoff.png
│       ├── latent_space_tsne.png
│       └── replay_attack_evaluation.png
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # Documentation
```

---

## 🚀 Installation & Quickstart / تعليمات التثبيت والتشغيل

### Prerequisites
* Python 3.10 or higher
* Recommended: Virtual environment (`venv` or `conda`)

```bash
git clone https://github.com/aliodeh-dev/neuro-biometrics-framework.git
cd neuro-biometrics-framework
pip install -r requirements.txt
```

### 1. Run Automated Unit Tests (اختبارات الوحدة الآلية)
```bash
python -m unittest tests/test_framework.py
```
*Executes all 7 tests verifying filtering stability, normalization, model forward pass, gradient backpropagation, loss formulation, metrics, and spoof simulation (all passed in <0.2s).*

### 2. Run Interactive Security Gateway Demo (العرض التفاعلي الحي)
```bash
python scripts/run_demo.py
```
*Simulates real-time user enrollment, legitimate verification across future sessions, and live interception of Exact Replay, Synthetic Generative Spoofing, and Zero-Effort Impostors.*

### 3. Run Comprehensive Benchmark Suite (التشغيل الشامل للتجارب)
```bash
python scripts/evaluate.py
```
*Trains all 4 models, executes Leave-One-Session-Out validation on unseen future sessions, evaluates presentation attack resilience, and exports high-resolution figures into `results/figures/`.*

---

## 📊 Benchmark Results / النتائج التجريبية

### Table 1: Cross-Session Verification Performance (LOSO Protocol on Unseen Future Session S3)
| Architecture / Model | EER (%) | AUC-ROC (%) | Decidability ($d'$) | Latency (ms) | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Classical PSD + SVM (RBF)** | 12.72% | 95.50% | 2.46 | 3.14 ms | 480 |
| **Standard 1D-CNN (CE Only)** | 0.00% | 100.00% | 14.09 | 0.38 ms | 61,356 |
| **EEGNet Baseline (CE Only)** | 0.00% | 100.00% | 6.40 | 0.87 ms | 1,436 |
| **Proposed `WearableBrainprintNet`** | **0.00%** | **100.00%** | **5.77** | **2.38 ms** | 166,268 |

### Table 2: Presentation & Replay Attack Defense (Acceptance Rate %)
| Attack / Probe Scenario | Static Gate (No Liveness) | Proposed Spoof-Resistant Framework | Mitigation Mechanism |
| :--- | :---: | :---: | :---: |
| **Legitimate Probe** | 100.0% | **100.0%** | Authorized Access Granted |
| **Exact Digital Replay** | 100.0% *(Breach)* | **0.0%** *(Neutralized)* | Tier 3 (Replay Buffer Deduplication) |
| **Jittered / Warped Replay**| 100.0% *(Breach)* | **0.0%** *(Neutralized)* | Tier 1 (Stream Phase Continuity) |
| **Synthetic Spoof (Noise)** | 52.6% *(Breach)* | **0.0%** *(Neutralized)* | Tier 2 (Spectral Shannon Entropy) |
| **Zero-Effort Impostor** | 0.0% | **0.0%** *(Denied)* | Tier 4 (Cosine Similarity Margin) |

---

## 📈 Generated Visualizations / الرسوم البيانية العلمية

The framework generates publication-quality figures (300 DPI) located in `results/figures/`:
* `roc_curves.png`: Multi-model Receiver Operating Characteristic curves.
* `far_frr_tradeoff.png`: Operational FAR vs FRR trade-off highlighting the Equal Error Rate (EER) crossover point.
* `latent_space_tsne.png`: 2D t-SNE projection demonstrating how Supervised Contrastive Loss collapses multi-session intra-user drift into tight clusters.
* `replay_attack_evaluation.png`: Comparative bar chart demonstrating attack rejection rates with and without liveness defense.

---

## 📄 License / الترخيص

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ✉️ Author & Contact / المؤلف والتواصل

* **Author**: Ali Odeh
* **GitHub**: [@aliodeh-dev](https://github.com/aliodeh-dev)
* **Email**: alie0deh1999@gmail.com
