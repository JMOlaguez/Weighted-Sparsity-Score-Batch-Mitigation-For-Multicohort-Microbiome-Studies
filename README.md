# 🧬 Microbiome Batch Effect Neutralization & ASD Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-State--of--the--Art-EE4C2C)
![LightGBM](https://img.shields.io/badge/LightGBM-Robust-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📌 Overview
This repository contains a cutting-edge computational pipeline designed to predict Autism Spectrum Disorder (ASD) from microbiome sequencing data. 

In multi-center clinical datasets, Machine Learning models often suffer from **"Shortcut Learning."** Instead of learning the complex biological signature of a disease, they memorize the technical "Geographic Barcode" (Batch Effect) caused by different extraction kits, sequencing depth, or lab protocols across different hospitals. 

This project introduces a **"Causal Sterilization"** methodology. We deploy adversarial networks, mathematical footprinting, and causal zero imputation to systematically annihilate geographic biases while preserving true biological rarity, proving that our models learn real biology, not just zip codes.

## 🔬 The Core Methodology: Weighted Sparsity Score (WSS)
At the heart of our surgical imputation is the **Weighted Sparsity Score (WSS)**. WSS identifies if a microbe's absence (zeros) is due to true biological rarity or technical dropout.

$$WSS = S_{global} \cdot \text{Var}(S_{batch})$$

Where $S_{global}$ is the global sparsity, and $\text{Var}(S_{batch})$ is the variance of sparsity across different hospitals. Microbes with high WSS are deemed "technically toxic" and undergo surgical imputation, while low-WSS microbes are protected.

## 🛠️ Prerequisites & Installation

To execute the pipelines, ensure you have a CUDA-enabled GPU (highly recommended for CDAN and Next-Gen models) and the following dependencies:

pip install pandas numpy scikit-learn lightgbm xgboost catboost torch torchvision seaborn matplotlib scipy joblib

Note: Next-Gen architectures (TabICL, MambaTab, TabM) may require specific environment setups as per their respective 2024-2026 literature.

## 📂 Pipeline Architecture (Script Index)

The pipeline is designed to be executed sequentially.

Phase 1: Detection & Stress Testing
9_test_invarianza_dominio.py: Domain Invariance Stress Test. Introduces the "Inquisitor" (an adversarial Random Forest) to prove the neural network is hiding the hospital label in its latent space.

10_dann_mixup_invariance.py: CDAN (Cascade) + Latent Mixup. A PyTorch-based adversarial network designed to blind the Inquisitor while protecting the clinical signal.

Phase 2: Tabular Impurity & Label Collision
11_synthetic_tabular_generator.py: Injects synthetic clones to destroy batch identification.

12_label_anonymizer.py: Clones real patients altering only their hospital label (Adversarial Collision).

13_absolute_forensic_anonymizer.py: Omnipresent Poisoning. Clones patients across all hospitals simultaneously and extracts the exact features the Inquisitor uses to cheat.

Phase 3: The Surgical Antidote
14_sparsity_mitigation.py: X-Ray & Gaussian Ray. Drops hyper-sparse columns and overwrites moderately sparse ones.

15_wss_precision_surgery.py: Introduces the WSS metric to punish technical variance and protect biological rarity.

16_negative_binomial_mitigation.py: Replaces the Gaussian Ray with a Negative Binomial distribution to respect the natural overdispersion of sequencing counts.

17_empirical_mds_mitigation.py: Marginal Distribution Sampling (MDS). Uses non-parametric empirical sampling to heal toxic bacteria without mathematical assumptions.

18_batch_variance_stabilization.py: Z-Score matching to flatten technical biases while preserving intra-patient covariance.

19_causal_zero_alignment.py: Causal Zero Alignment. Sews up structural zeros (dropouts) using only the positive empirical distribution.

19b_causal_metrics_export.py: Lightweight exporter of the Phase 3 causal metrics for rapid forensic analysis.

Phase 4: Massive Orchestration & Benchmarking (Pooled Data)
Note: This phase utilizes a globally concatenated (pooled) dataset to establish a clean performance baseline before strict spatial testing.

20_orchestrator_frontier_protocol.py: The Traditional ML Benchmark. Trains 23 algorithms on the causally sterilized dataset, strictly avoiding Data Leakage during cross-validation.

21_nextgen_orchestrator.py: The 2024-2026 SOTA Benchmark. Tests Foundation Models, State-Space Models (Mamba), and Deep Ensembles against the sterilized data to prove deep learning capabilities beyond batch effect memorization.

Phase 5: Ultimate Clinical Validation22_causal_sterilized_lodo.py: Leave-One-Domain-Out (LODO). Strict spatial evaluation where sterilization parameters are fitted on $N-1$ hospitals and projected onto the test hospital.23_dirty_lodo_baseline.py: The Control Group. LODO on raw data to empirically demonstrate the catastrophic collapse caused by Shortcut Learning.24_supplementary_montecarlo_validation.py: Generates high-resolution sensitivity sweeps, Wasserstein distance divergence analysis, and a 250-run Monte Carlo simulation to prove mathematical stability ($\sigma^2 \to 0$).

## 🚀 Usage
Ensure your input data is located in the 1_INPUT_DATA directory.

DATA_TRAIN_READY.csv

DATA_TEST_READY.csv

CONSENSO_BIOMARCADORES.csv

Execute the scripts in order to replicate the full sterilization and validation lifecycle. All outputs, plots, and deployable .pkl models will be automatically routed to their respective _RESULTS folders.

## 📝 License & Academic Citation
This project is licensed under the MIT License. If you use this methodology or the WSS algorithm in your research, please cite this article:


And cite this repository.
