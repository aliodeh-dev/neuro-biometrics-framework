# Master's Thesis Experimental Benchmark Summary

## Table 1: Cross-Session Biometric Verification Performance Comparison

| Architecture / Model | EER (%) | AUC-ROC (%) | Decidability (d') | FAR @ FRR=1% | FRR @ FAR=1% | Latency (ms) | Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical PSD + SVM** | 12.72% | 95.50% | 2.46 | 45.65% | 27.12% | 3.14 ms | 480 |
| **Standard 1D-CNN (CE)** | 0.00% | 100.00% | 14.09 | 0.00% | 0.00% | 0.38 ms | 61,356 |
| **EEGNet Baseline** | 0.00% | 100.00% | 6.40 | 0.00% | 0.00% | 0.87 ms | 1,436 |
| **Proposed WearableBrainprintNet** | 0.00% | 100.00% | 5.77 | 0.00% | 0.00% | 2.38 ms | 166,268 |


## Table 2: Presentation & Replay Attack Resistance (Acceptance Rate %)

| Attack / Probe Scenario | Static Biometric Gate (No Liveness) | Proposed Spoof-Resistant Framework |
| :--- | :---: | :---: |
| Legitimate Probe | 100.0% | **100.0%** |
| Exact Replay | 100.0% | **0.0%** |
| Jittered Replay | 100.0% | **52.6%** |
| Synthetic Spoof | 52.6% | **0.0%** |
| Cross Subject Masquerade | 0.0% | **0.0%** |
