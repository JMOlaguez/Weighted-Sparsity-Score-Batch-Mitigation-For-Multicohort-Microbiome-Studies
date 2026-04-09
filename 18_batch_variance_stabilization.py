# -*- coding: utf-8 -*-
"""
18_batch_variance_stabilization.py

PHASE 2: BATCH VARIANCE STABILIZATION (RECODE-Inspired)
Objective: Align local distributions with the global one (Z-score matching)
for toxic bacteria, preserving intra-patient covariance.

---------------------------------------------------------------------------
THE NOVELTY: WHY Z-SCORE MATCHING? (COVARIANCE PRESERVATION)
---------------------------------------------------------------------------
While Phase 1 (Empirical MDS) is excellent at destroying the geographic barcode,
it randomly shuffles values, which can break the natural biological correlation
(covariance) between different bacteria within the same patient.

Variance Stabilization (Z-score matching) takes a more surgical approach:
1. It calculates the Global Mean ($\mu_G$) and Global Standard Deviation ($\sigma_G$).
2. For each hospital, it calculates the Local Mean ($\mu_L$) and Local Std Dev ($\sigma_L$).
3. It maps every patient's value ($x$) to the global distribution using:
   $x_{new} = \left( \frac{x - \mu_L}{\sigma_L} \right) \cdot \sigma_G + \mu_G$

EXPLANATION:
If Patient A had a naturally high abundance of a bacteria compared to their
peers in Hospital 1, they will STILL have a high abundance after the
transformation. We merely "shift and stretch" the hospital's entire
distribution to match the global average, erasing the Batch Effect
without destroying the internal biological rankings.

PREREQUISITES:
- Python 3.8+
- Libraries: pandas, numpy, lightgbm, scikit-learn
- Required Input Files (in DIR_INPUT):
  * DATA_TRAIN_READY.csv
  * DATA_TEST_READY.csv
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

DIR_INPUT = "1_INPUT_DATA"

def ejecutar_estabilizacion_varianza():
    print("\n" + "🧬"*30)
    print(" STARTING PHASE 2: BATCH VARIANCE STABILIZATION ")
    print(" Strategy: Z-score matching to flatten technical biases.")
    print("🧬"*30 + "\n")

    df_train = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TRAIN_READY.csv"))
    df_test = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TEST_READY.csv"))

    excluir = ['diagnosis', 'source', 'Batch', 'SampleID']
    features = [c for c in df_train.columns if c not in excluir]

    hospitales = df_train['source'].unique()
    n_hospitales = len(hospitales)
    y_test_asd = df_test['diagnosis']
    y_test_hosp = df_test['source']
    epsilon_cero = 1e-5

    # --- 1. WSS CALCULATION (Identify Toxicity) ---
    wss_resultados = []
    for feat in features:
        s_global = (df_train[feat] <= epsilon_cero).mean()
        s_batch = [(df_train[df_train['source'] == h][feat] <= epsilon_cero).mean() for h in hospitales if len(df_train[df_train['source'] == h]) > 0]
        var_batch = np.var(s_batch)
        wss_resultados.append({'Bacteria': feat, 'WSS': s_global * var_batch})

    df_wss = pd.DataFrame(wss_resultados)

    # We maintain the rigorous standard: 75th Percentile
    umbral_wss = np.percentile(df_wss['WSS'], 75)
    print(f"📊 Aligning variance for the top 25% most toxic bacteria (WSS > {umbral_wss:.4f})...\n")

    df_train_mod = df_train.copy()
    bacterias_estabilizadas = 0
    bacterias_colapsadas = 0

    # --- 2. THE VARIANCE STABILIZATION ENGINE ---
    for feat in features:
        wss_actual = df_wss[df_wss['Bacteria'] == feat]['WSS'].values[0]

        if wss_actual >= umbral_wss:
            # Global Statistics
            mu_global = df_train[feat].mean()
            std_global = df_train[feat].std()

            # Hospital by Hospital Adjustment
            for hosp in hospitales:
                idx_hosp = df_train_mod['source'] == hosp
                if not idx_hosp.any():
                    continue

                valores_locales = df_train_mod.loc[idx_hosp, feat]
                mu_local = valores_locales.mean()
                std_local = valores_locales.std()

                # If the local deviation is > 0, we apply the Z-score projection
                if std_local > 1e-8:
                    valores_transformados = ((valores_locales - mu_local) / std_local) * std_global + mu_global
                else:
                    # If the hospital has zero variance (e.g., the entire hospital is 0 for this bacteria),
                    # we push it towards the global mean with a tiny noise to break the block.
                    ruido = np.random.normal(0, std_global * 0.05, size=len(valores_locales))
                    valores_transformados = mu_global + ruido
                    bacterias_colapsadas += 1

                # Biological rectifier (we do not allow negative abundances)
                df_train_mod.loc[idx_hosp, feat] = np.maximum(0, valores_transformados)

            bacterias_estabilizadas += 1

    print(f"✔️ Stabilized bacteria (Variance Alignment): {bacterias_estabilizadas}")
    if bacterias_colapsadas > 0:
        print(f"⚠️ Emergency corrections (hospitals with 0 variance): {bacterias_colapsadas} interventions.")
    print(f"🛡️ Biologically intact bacteria (low WSS): {len(features) - bacterias_estabilizadas}")

    # --- 3. OMNIPRESENCE (The Impurity Bomb) ---
    print(f"\n🥷 Applying Label Collision (Cloning x{n_hospitales})...")
    clones = []
    for _, row in df_train_mod.iterrows():
        for hosp in hospitales:
            clon = row.copy()
            clon['source'] = hosp
            clon['Batch'] = 'SINTETICO_VARIANZA' # SYNTHETIC_VARIANCE
            clones.append(clon)

    df_train_omni = pd.DataFrame(clones)

    # --- 4. TRAINING AND TESTING ON FRESH DATA ---
    print("⚙️ Training NEW models on the stabilized dataset...")
    X_train_final = df_train_omni[features]
    X_test_final = df_test[features]

    modelo_clinico = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
    modelo_clinico.fit(X_train_final, df_train_omni['diagnosis'])
    acc_asd = accuracy_score(y_test_asd, modelo_clinico.predict(X_test_final))

    modelo_hosp = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
    modelo_hosp.fit(X_train_final, df_train_omni['source'])
    acc_hosp = accuracy_score(y_test_hosp, modelo_hosp.predict(X_test_final))

    azar_hosp = 1.0 / n_hospitales

    print(f"\n" + "="*60)
    print(f"📊 FINAL RESULTS (VARIANCE STABILIZATION - P75):")
    print(f"   |-- Clinical Accuracy (Autism): {acc_asd*100:.2f}%")
    print(f"   |-- Inquisitor Accuracy (Hospital): {acc_hosp*100:.2f}% (Chance: {azar_hosp*100:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    ejecutar_estabilizacion_varianza()
