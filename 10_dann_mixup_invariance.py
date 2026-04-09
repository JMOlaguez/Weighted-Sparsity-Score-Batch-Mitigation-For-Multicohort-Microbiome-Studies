# -*- coding: utf-8 -*-
"""
10_dann_mixup_invariance.py

THE ANTIDOTE V3.0: CDAN (CASCADE) + LATENT MIXUP + AUTOENCODER
Objective: Destroy the geographic signature (Batch Effect) bringing the accuracy
closer to 8.33%, while protecting the ASD biological signal (avoiding Semantic Collapse).

PREREQUISITES:
- Python 3.8+
- Libraries: torch, pandas, numpy, matplotlib, seaborn, scikit-learn
- Hardware: CUDA-enabled GPU (Highly recommended for PyTorch), fallback to CPU
- Required Input Files (in DIR_INPUT):
  * DATA_TRAIN_READY.csv
  * DATA_TEST_READY.csv
  * CONSENSO_BIOMARCADORES.csv
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Function

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import warnings

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
warnings.filterwarnings("ignore")

# --- PATHS ---
DIR_INPUT = "1_INPUT_DATA"
DIR_RESULTS = "2_OUTPUT_RESULTS"
DIR_PLOTS = "3_PLOTS_DANN"

# Automatic error-proof directory creation
os.makedirs(DIR_RESULTS, exist_ok=True)
os.makedirs(DIR_PLOTS, exist_ok=True)

# ==========================================
# 1. GRADIENT REVERSAL LAYER (GRL)
# ==========================================
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Multiply the gradient by -alpha
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalFn.apply(x, alpha)

# ==========================================
# 2. NETWORK ARCHITECTURE
# ==========================================
class DANN_Mixup_Model(nn.Module):
    def __init__(self, input_dim, num_hospitals):
        super(DANN_Mixup_Model, self).__init__()

        # Bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 16),
            nn.ReLU()
        )

        # Biological Regularizer
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

        # Clinical Classifier
        self.clinical_classifier = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        # The Conditioned Inquisitor (CDAN) - Accepts 17 dimensions
        self.domain_classifier = nn.Sequential(
            nn.Linear(17, 16), # 16 from latent + 1 from clinical prediction
            nn.ReLU(),
            nn.Linear(16, num_hospitals)
        )

    def forward(self, x, alpha=1.0, apply_mixup=False, mixup_lambda=None, index=None):
        latent = self.encoder(x)

        if apply_mixup and mixup_lambda is not None and index is not None:
            latent_mixed = mixup_lambda * latent + (1 - mixup_lambda) * latent[index, :]
        else:
            latent_mixed = latent

        reconstruction = self.decoder(latent)
        clinical_pred = self.clinical_classifier(latent)

        # --- THE CDAN TRICK (THE CASCADE) ---
        # Concatenate the latent space with the clinical prediction
        conditioned_latent = torch.cat((latent_mixed, clinical_pred), dim=1)

        # Reverse the gradient of the conditioned latent space
        reversed_latent = grad_reverse(conditioned_latent, alpha)
        domain_pred = self.domain_classifier(reversed_latent)

        return clinical_pred, domain_pred, reconstruction, latent

# ==========================================
# 3. MAIN ENGINE
# ==========================================
def ejecutar_dann():
    print("\n" + "🔥"*30)
    print(" STARTING CDAN V3 PROTOCOL (CASCADE + MIXUP) ")
    print(" Objective: Shield Autism biology and blind the Hospital")
    print("🔥"*30 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_train = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TRAIN_READY.csv"))
    df_test = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TEST_READY.csv"))
    df_consenso = pd.read_csv(os.path.join(DIR_INPUT, "CONSENSO_BIOMARCADORES.csv"))

    top_features = df_consenso[df_consenso.columns[0]].head(20).tolist()
    features_validas = [f for f in top_features if f in df_train.columns]

    X_train_raw = df_train[features_validas].values
    y_train_asd = df_train['diagnosis'].values
    le_hosp = LabelEncoder()
    y_train_hosp = le_hosp.fit_transform(df_train['source'].values)

    X_test_raw = df_test[features_validas].values
    y_test_asd = df_test['diagnosis'].values
    y_test_hosp = le_hosp.transform(df_test['source'].values)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    X_tr_t = torch.FloatTensor(X_train).to(device)
    y_asd_tr_t = torch.FloatTensor(y_train_asd).unsqueeze(1).to(device)
    y_hosp_tr_t = torch.LongTensor(y_train_hosp).to(device)

    train_data = TensorDataset(X_tr_t, y_asd_tr_t, y_hosp_tr_t)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    model = DANN_Mixup_Model(input_dim=len(features_validas), num_hospitals=len(le_hosp.classes_)).to(device)

    criterion_clinical = nn.BCELoss()
    criterion_domain = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    epochs = 150
    print("\n⚔️ Executing Cascade Conditioning...")

    model.train()
    for epoch in range(epochs):
        p = float(epoch) / epochs
        # Return alpha to a maximum of 1.0 for stability with CDAN
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        for batch_x, batch_y_asd, batch_y_hosp in train_loader:
            optimizer.zero_grad()

            mixup_lambda = np.random.beta(2.0, 2.0)
            index = torch.randperm(batch_x.size(0)).to(device)

            pred_asd, pred_hosp, recon, _ = model(
                batch_x, alpha=alpha, apply_mixup=True, mixup_lambda=mixup_lambda, index=index
            )

            loss_domain = mixup_lambda * criterion_domain(pred_hosp, batch_y_hosp) + \
                          (1 - mixup_lambda) * criterion_domain(pred_hosp, batch_y_hosp[index])

            loss_clinical = criterion_clinical(pred_asd, batch_y_asd)

            loss_recon = criterion_recon(recon, batch_x) * 2.0

            loss_total = loss_clinical + loss_recon + loss_domain

            loss_total.backward()
            optimizer.step()

    # --- FORENSIC EVALUATION ---
    model.eval()
    with torch.no_grad():
        X_te_t = torch.FloatTensor(X_test).to(device)
        pred_asd_test, pred_hosp_test, _, embeddings_test = model(X_te_t, alpha=1.0)

        pred_asd_labels = (pred_asd_test.cpu().numpy() >= 0.5).astype(int)
        acc_asd = accuracy_score(y_test_asd, pred_asd_labels)

        pred_hosp_labels = torch.argmax(pred_hosp_test, dim=1).cpu().numpy()
        acc_hosp = accuracy_score(y_test_hosp, pred_hosp_labels)
        azar = 1.0 / len(le_hosp.classes_)

    print("\n" + "="*50)
    print(f"📊 FINAL GENERALIZATION RESULTS (CDAN V3):")
    print(f"   |-- Clinical Accuracy (ASD): {acc_asd*100:.2f}%")
    print(f"   |-- Inquisitor Accuracy (Hospital): {acc_hosp*100:.2f}% (Expected Chance: {azar*100:.2f}%)")
    print("="*50)

    print("\n🎨 Generating visual proofs (t-SNE)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings_test.cpu().numpy())

    df_tsne = pd.DataFrame({
        'TSNE_1': emb_2d[:, 0],
        'TSNE_2': emb_2d[:, 1],
        'Hospital': df_test['source'],
        'Diagnosis': df_test['diagnosis'].map({0: 'Control', 1: 'Autism'})
    })

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='TSNE_1', y='TSNE_2', hue='Hospital', palette='tab20', data=df_tsne, alpha=0.8)
    plt.title('CDAN V3: Latent Space Colored by Hospital\n(Geographic barriers should be destroyed)', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "7_TSNE_V3_Hospital.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='TSNE_1', y='TSNE_2', hue='Diagnosis', palette=['#3498db', '#e74c3c'], data=df_tsne, alpha=0.7)
    plt.title('CDAN V3: Latent Space Colored by Clinical Status\n(Recovering the ASD biological signal)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "8_TSNE_V3_Diagnosis.png"), dpi=300)
    plt.close()

    print(f"✅ Completed! Check the new folder '{DIR_PLOTS}'.")

if __name__ == "__main__":
    ejecutar_dann()
