# -*- coding: utf-8 -*-
"""
9_test_invarianza_dominio.py

DOMAIN INVARIANCE STRESS TEST (GOLD STANDARD)
Objective: Demonstrate if the neural network "hides" the zip code
(batch effects/source hospital) in its latent space (embeddings)
when trying to predict Autism.

PREREQUISITES:
- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
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
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import warnings

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
warnings.filterwarnings("ignore")

# --- NEW ARCHITECTURE PATHS ---
DIR_INPUT = "1_INPUT_DATA"
DIR_RESULTS = "2_OUTPUT_RESULTS"
DIR_PLOTS = "3_PLOTS"

# Forensic function to extract the neural network's "mind"
def extraer_espacio_latente(mlp, X):
    """
    Passes data through hidden layers by multiplying matrices and applying ReLU.
    Returns embeddings from the penultimate layer.
    """
    activaciones = X.values
    # Iterate through all layers except the last one (output)
    for i in range(mlp.n_layers_ - 2):
        activaciones = np.dot(activaciones, mlp.coefs_[i]) + mlp.intercepts_[i]
        # ReLU activation function
        activaciones = np.maximum(0, activaciones)
    return activaciones

def ejecutar_test_invarianza():
    print("\n" + "👁️"*30)
    print(" STARTING DOMAIN INVARIANCE STRESS TEST ")
    print(" 1. Training Neural Network (MLP) to predict ASD")
    print(" 2. Extracting the Latent Space (Embeddings)")
    print(" 3. Launching the Inquisitor (RF) to predict Origin")
    print("👁️"*30 + "\n")

    # 1. Load Data
    df_train = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TRAIN_READY.csv"))
    df_test = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TEST_READY.csv"))
    df_consenso = pd.read_csv(os.path.join(DIR_INPUT, "CONSENSO_BIOMARCADORES.csv"))

    # Use ONLY the Top 20 Autism biomarkers (Strict Mode)
    col_nombre = df_consenso.columns[0]
    top_features = df_consenso[col_nombre].head(20).tolist()
    features_validas = [f for f in top_features if f in df_train.columns]

    print(f"🧬 Filtering Megadataset to use ONLY the top {len(features_validas)} ASD features...")

    X_train_raw = df_train[features_validas]
    y_train_asd = df_train['diagnosis']
    y_train_hosp = df_train['source']

    X_test_raw = df_test[features_validas]
    y_test_asd = df_test['diagnosis']
    y_test_hosp = df_test['source']

    # Scale data (critical for Neural Networks)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=features_validas)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=features_validas)

    # 2. Train the Extractor (Neural Network)
    print("\n🧠 Training Multilayer Perceptron (MLP) on the Autism signal...")
    # Architecture: 64 neurons in layer 1, 32 neurons in layer 2 (our latent space)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
    mlp.fit(X_train, y_train_asd)

    asd_acc = accuracy_score(y_test_asd, mlp.predict(X_test))
    print(f"   |-- Clinical Accuracy (ASD vs Control): {asd_acc*100:.2f}%")

    # 3. Extract Latent Space (32D Embeddings)
    print("\n🔬 Performing mathematical neurosurgery: Extracting 32D embeddings...")
    embeddings_train = extraer_espacio_latente(mlp, X_train)
    embeddings_test = extraer_espacio_latente(mlp, X_test)
    print(f"   |-- Shape of the new latent space: {embeddings_test.shape}")

    # 4. The Inquisitor (Adversarial Model)
    print("\n⚖️ Launching the Inquisitor: Random Forest trying to predict the Hospital from the Latent Space...")
    le = LabelEncoder()
    y_train_hosp_enc = le.fit_transform(y_train_hosp)
    y_test_hosp_enc = le.transform(y_test_hosp)

    inquisidor = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    inquisidor.fit(embeddings_train, y_train_hosp_enc)

    hosp_acc = accuracy_score(y_test_hosp_enc, inquisidor.predict(embeddings_test))
    azar_baseline = 100.0 / len(le.classes_)

    print(f"   |-- Adversary Accuracy (Guessing Hospital): {hosp_acc*100:.2f}% (Chance Baseline: {azar_baseline:.2f}%)")

    # Save Numeric Report
    res_df = pd.DataFrame([{
        'Metrica': 'Biological_Accuracy_ASD', 'Valor': asd_acc
    }, {
        'Metrica': 'Adversary_Accuracy_Hospital', 'Valor': hosp_acc
    }])
    res_df.to_csv(os.path.join(DIR_RESULTS, "RESULTADOS_INVARIANZA.csv"), index=False)

    # 5. Visual Proof: 2D t-SNE Projection
    print("\n🎨 Drawing the Neural Network's 'thoughts' with t-SNE...")
    # Merge train and test for the global plot
    embeddings_global = np.vstack((embeddings_train, embeddings_test))
    hosp_global = pd.concat([y_train_hosp, y_test_hosp]).reset_index(drop=True)
    asd_global = pd.concat([y_train_asd, y_test_asd]).reset_index(drop=True)

    # Reduce from 32D to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings_global)

    df_tsne = pd.DataFrame({
        'TSNE_1': emb_2d[:, 0],
        'TSNE_2': emb_2d[:, 1],
        'Hospital': hosp_global,
        'Diagnostico': asd_global.map({0: 'Control', 1: 'Autism'})
    })

    # Plot A: Colored by Diagnosis
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='TSNE_1', y='TSNE_2', hue='Diagnostico', palette=['#3498db', '#e74c3c'], data=df_tsne, alpha=0.7)
    plt.title('Neural Network Latent Space\nColored by Clinical Status (Seeking Biology)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "1_TSNE_Diagnosis.png"), dpi=300)
    plt.close()

    # Plot B: Colored by Hospital
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='TSNE_1', y='TSNE_2', hue='Hospital', palette='tab20', data=df_tsne, alpha=0.8)
    plt.title('Neural Network Latent Space\nColored by Source Hospital (Seeking Batch Effect)', fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_PLOTS, "2_TSNE_Hospital.png"), dpi=300)
    plt.close()

    print(f"✅ Experiment finished! Check the {DIR_PLOTS} folder to see the visual proofs.")
    print("="*65)

if __name__ == "__main__":
    ejecutar_test_invarianza()
