# -*- coding: utf-8 -*-
"""
15_wss_precision_surgery.py

PRECISION SURGERY: WEIGHTED SPARSITY SCORE (WSS) + OMNIPRESENCE
Objective: Identify and neutralize ONLY the bacteria whose zero-pattern
varies artificially between hospitals, protecting true biological rarity.

---------------------------------------------------------------------------
MATHEMATICAL METHOD: WEIGHTED SPARSITY SCORE (WSS)
---------------------------------------------------------------------------
The WSS is designed to differentiate between "True Biological Rarity"
(a microbe that is naturally rare everywhere) and "Geographic Toxicity"
(a microbe that appears in one hospital but is entirely absent in another
due to sequencing artifacts or batch effects).

The equation for a given feature (bacteria) is (Latex format):
$WSS = S_{global} \cdot \text{Var}(S_{batch})$

Where:
- $S_{global}$: The global sparsity (proportion of zeros for this bacteria across the entire dataset).
- $\text{Var}(S_{batch})$: The variance of the sparsity proportions across the different hospitals (batches).

EXAMPLE WITH 6 VARIABLES ACROSS 3 HOSPITALS:
Imagine we calculate the sparsity (proportion of zeros) for 6 bacteria:

| Bacteria | Hosp_1 Zeros | Hosp_2 Zeros | Hosp_3 Zeros | S_global | Var(S_batch) | WSS Score | Conclusion |
|----------|--------------|--------------|--------------|----------|--------------|-----------|------------|
| Bact_A   | 0.90         | 0.10         | 0.95         | ~0.65    | 0.150        | 0.0975    | TOXIC (Batch Effect) |
| Bact_B   | 0.85         | 0.82         | 0.88         | 0.85     | 0.0006       | 0.0005    | SAFE (True Rarity) |
| Bact_C   | 0.05         | 0.02         | 0.08         | 0.05     | 0.0006       | 0.00003   | SAFE (Common/Stable)|
| Bact_D   | 0.00         | 1.00         | 0.00         | ~0.33    | 0.222        | 0.0732    | TOXIC (Missing in Hosp2)|
| Bact_E   | 0.50         | 0.55         | 0.45         | 0.50     | 0.0016       | 0.0008    | SAFE (Stable noise) |
| Bact_F   | 1.00         | 0.20         | 0.90         | 0.70     | 0.126        | 0.0882    | TOXIC (Batch Effect) |

Action: We rank by WSS. Bacteria A, D, and F are destroyed (Gaussian replacement)
because their variance is artificial. Bacteria B (even though it's very sparse)
is protected because its rarity is consistent everywhere.

PREREQUISITES:
- Python 3.8+
- Libraries: pandas, numpy, lightgbm, matplotlib, seaborn, scikit-learn
- Required Input Files (in DIR_INPUT):
  * DATA_TRAIN_READY.csv
  * DATA_TEST_READY.csv
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

DIR_INPUT = "1_INPUT_DATA"
DIR_PLOTS = "3_PLOTS_DANN"
os.makedirs(DIR_PLOTS, exist_ok=True)

def ejecutar_cirugia_wss():
    print("\n" + "🧬"*30)
    print(" STARTING WSS SURGERY: WEIGHTED SPARSITY SCORE ")
    print(" Objective: Punish technical variance, protect biological rarity.")
    print("🧬"*30 + "\n")

    # 1. Load Data
    df_train = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TRAIN_READY.csv"))
    df_test = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TEST_READY.csv"))

    excluir = ['diagnosis', 'source', 'Batch', 'SampleID']
    features = [c for c in df_train.columns if c not in excluir]

    hospitales = df_train['source'].unique()
    n_hospitales = len(hospitales)

    y_test_asd = df_test['diagnosis']
    y_test_hosp = df_test['source']
    azar_hosp = 1.0 / n_hospitales
    epsilon_cero = 1e-5

    # ==========================================
    # STEP 1: WSS CALCULATION FOR EACH BACTERIA
    # ==========================================
    print("🔍 Calculating the Geographic Toxicity Index (WSS)...")
    wss_resultados = []

    for feat in features:
        # A. Global Sparsity
        s_global = (df_train[feat] <= epsilon_cero).mean()

        # B. Sparsity per Hospital (Batch)
        s_batch = []
        for hosp in hospitales:
            df_hosp = df_train[df_train['source'] == hosp]
            if len(df_hosp) > 0:
                s_batch.append((df_hosp[feat] <= epsilon_cero).mean())

        # C. Sparsity variance between hospitals
        var_batch = np.var(s_batch)

        # WSS Equation
        wss_val = s_global * var_batch
        wss_resultados.append({'Bacteria': feat, 'WSS': wss_val, 'S_Global': s_global, 'Var_Batch': var_batch})

    df_wss = pd.DataFrame(wss_resultados).sort_values(by='WSS', ascending=False)

    print("\n🚨 TOP 5 MOST TOXIC BACTERIA (Highest WSS - Strong Barcode):")
    print(df_wss.head(5).to_string(index=False))

    print("\n🛡️ TOP 5 MOST HONEST BACTERIA (Lowest WSS - Stable Rarity):")
    print(df_wss.tail(5).to_string(index=False))

    # Define dynamic thresholds based on WSS percentiles
    # 50th Percentile: Neutralize the most toxic half. 75th Percentile: Only the top 25% most toxic.
    percentiles_corte = [50, 75, 90]

    for p in percentiles_corte:
        umbral_wss = np.percentile(df_wss['WSS'], p)
        print("\n" + "="*70)
        print(f"🚀 EXPERIMENT: NEUTRALIZING BACTERIA ABOVE THE {p}th WSS PERCENTILE")
        print("="*70)

        df_train_mod = df_train.copy()
        bacterias_gaussianas = 0

        # ==========================================
        # STEP 2: THE SURGICAL GAUSSIAN RAY
        # ==========================================
        for feat in features:
            wss_actual = df_wss[df_wss['Bacteria'] == feat]['WSS'].values[0]

            # If the WSS exceeds the threshold, the bacteria is toxic: Agnostic Imputation
            if wss_actual >= umbral_wss:
                mu = df_train_mod[feat].mean()
                sigma = df_train_mod[feat].std()

                # TOTAL replacement with global Gaussian noise
                valores_sinteticos = np.random.normal(mu, sigma, size=len(df_train_mod))
                valores_sinteticos = np.maximum(0, valores_sinteticos)
                df_train_mod[feat] = valores_sinteticos
                bacterias_gaussianas += 1

        print(f"💉 Bacteria cured with Gaussian Noise (Highly toxic): {bacterias_gaussianas} out of {len(features)}")
        print(f"✅ Biologically intact bacteria (Low WSS): {len(features) - bacterias_gaussianas}")

        # ==========================================
        # STEP 3: OMNIPRESENCE (N-TIMES CLONING)
        # ==========================================
        print(f"🥷 Sealing the vault: Applying Label Collision (x{n_hospitales})...")
        clones = []
        for _, row in df_train_mod.iterrows():
            for hosp in hospitales:
                clon = row.copy()
                clon['source'] = hosp
                clon['Batch'] = 'SINTETICO_WSS' # SYNTHETIC_WSS
                clones.append(clon)

        df_train_omni = pd.DataFrame(clones)

        # ==========================================
        # STEP 4: TRAINING AND TESTING
        # ==========================================
        X_train_final = df_train_omni[features]
        X_test_final = df_test[features]

        modelo_clinico = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
        modelo_clinico.fit(X_train_final, df_train_omni['diagnosis'])
        acc_asd = accuracy_score(y_test_asd, modelo_clinico.predict(X_test_final))

        modelo_hosp = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
        modelo_hosp.fit(X_train_final, df_train_omni['source'])
        acc_hosp = accuracy_score(y_test_hosp, modelo_hosp.predict(X_test_final))

        print(f"\n📊 FINAL RESULTS (WSS > {p}th Percentile):")
        print(f"   |-- Clinical Accuracy (Autism): {acc_asd*100:.2f}%")
        print(f"   |-- Inquisitor Accuracy (Hospital): {acc_hosp*100:.2f}% (Chance: {azar_hosp*100:.2f}%)")

if __name__ == "__main__":
    ejecutar_cirugia_wss()
