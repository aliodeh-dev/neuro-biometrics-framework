# Master's Thesis Experimental Benchmark Summary

## Table 1: Cross-Session Biometric Verification Performance Comparison

| Architecture / Model | EER (%) | AUC-ROC (%) | Decidability (d') | FAR @ FRR=1% | FRR @ FAR=1% | Latency (ms) | Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical PSD + SVM** | 18.61% | 90.03% | 1.74 | 78.72% | 38.56% | 3.70 ms | 480 |
| **Standard 1D-CNN (CE)** | 0.28% | 100.00% | 7.82 | 0.06% | 0.00% | 0.45 ms | 61,356 |
| **EEGNet Baseline** | 1.28% | 99.83% | 4.69 | 1.34% | 3.39% | 0.68 ms | 1,436 |
| **Proposed WearableBrainprintNet** | 0.00% | 100.00% | 6.17 | 0.00% | 0.00% | 1.51 ms | 166,268 |


## Table 2: Presentation & Replay Attack Resistance (Acceptance Rate %)

| Attack / Probe Scenario | Static Biometric Gate (No Liveness) | Proposed Spoof-Resistant Framework |
| :--- | :---: | :---: |
| Legitimate Probe | 100.0% | **100.0%** |
| Exact Replay | 100.0% | **0.0%** |
| Jittered Replay | 100.0% | **68.4%** |
| Synthetic Spoof | 73.7% | **10.5%** |
| Cross Subject Masquerade | 0.0% | **0.0%** |
