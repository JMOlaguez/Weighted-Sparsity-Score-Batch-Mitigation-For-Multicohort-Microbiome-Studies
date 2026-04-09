# -*- coding: utf-8 -*-
"""
17_empirical_mds_mitigation.py

PHASE 1: MARGINAL DISTRIBUTION SAMPLING (EMPIRICAL MDS)
Objective: Neutralize toxic bacteria by resampling their values directly
from the global empirical distribution, training fresh and clean models.

---------------------------------------------------------------------------
THE NOVELTY: WHY EMPIRICAL MDS? (NON-PARAMETRIC HEALING)
---------------------------------------------------------------------------
Previous methods (Gaussian, Negative Binomial) are "Parametric". They force
the noise to follow a specific mathematical curve. However, microbiome data
can be wildly unpredictable (multimodal, extreme zero-inflation, etc.).

Empirical Marginal Distribution Sampling (MDS) makes zero assumptions:
1. It takes all the values of a toxic bacteria across the entire dataset
   (the global empirical distribution).
2. It tosses them all into a virtual "bag".
3. It draws them out randomly with replacement to overwrite the patient data.

THE MATHEMATICAL EXPLANATION (Latex format):
Let $X$ be the feature vector. By sampling $x_i \sim P(X)$ purely randomly,
we perfectly preserve the global mean ($\mu$), variance ($\sigma^2$), skewness,
and exact zero-inflation of the bacteria.
However, we completely annihilate the correlation between the count $x_i$
and the hospital $h_i$ or patient $y_i$. It destroys the geographic barcode
while guaranteeing zero Data Leakage (since we don't look at the ASD/NT label
during the resampling).

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

def ejecutar_mds_empirico():
    print("\n" + "🧬"*30)
    print(" STARTING PHASE 1: MARGINAL DISTRIBUTION SAMPLING (MDS) ")
    print(" Strategy: Pure empirical resampling for toxic bacteria.")
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

    # --- 1. WSS CALCULATION ---
    wss_resultados = []
    for feat in features:
        s_global = (df_train[feat] <= epsilon_cero).mean()
        s_batch = []
        for h in hospitales:
            df_hosp = df_train[df_train['source'] == h]
            if len(df_hosp) > 0:
                s_batch.append((df_hosp[feat] <= epsilon_cero).mean())
        var_batch = np.var(s_batch)
        wss_resultados.append({'Bacteria': feat, 'WSS': s_global * var_batch})

    df_wss = pd.DataFrame(wss_resultados)

    # Threshold at the 75th Percentile (we only punish the top 25% most toxic)
    umbral_wss = np.percentile(df_wss['WSS'], 75)
    print(f"📊 Neutralizing the top 25% of bacteria with the highest geographic toxicity (WSS > {umbral_wss:.4f})...\n")

    df_train_mod = df_train.copy()
    bacterias_imputadas = 0

    # --- 2. THE MDS ENGINE (PURE EMPIRICAL SAMPLING) ---
    for feat in features:
        wss_actual = df_wss[df_wss['Bacteria'] == feat]['WSS'].values[0]

        if wss_actual >= umbral_wss:
            # MDS: We extract the entire column (the global empirical distribution)
            distribucion_global = df_train[feat].values

            # Random resampling with replacement to break the link with the patient and the hospital
            # We do not look at ASD/NT to maintain methodological purity (Zero Data Leakage)
            valores_mds = np.random.choice(distribucion_global, size=len(df_train_mod), replace=True)

            df_train_mod[feat] = valores_mds
            bacterias_imputadas += 1

    print(f"✔️ Bacteria sterilized with Global Empirical MDS: {bacterias_imputadas}")
    print(f"🛡️ Biologically intact bacteria (low WSS): {len(features) - bacterias_imputadas}")

    # --- 3. OMNIPRESENCE (The Impurity Bomb) ---
    print(f"\n🥷 Applying Label Collision (Cloning x{n_hospitales})...")
    clones = []
    for _, row in df_train_mod.iterrows():
        for hosp in hospitales:
            clon = row.copy()
            clon['source'] = hosp
            clon['Batch'] = 'SINTETICO_MDS'
            clones.append(clon)

    df_train_omni = pd.DataFrame(clones)

    # --- 4. TRAINING FRESH MODELS ---
    print("⚙️ Training NEW models from scratch on the sterilized dataset...")
    X_train_final = df_train_omni[features]
    X_test_final = df_test[features]

    # Clinical Model (Learns from the clean environment)
    modelo_clinico = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
    modelo_clinico.fit(X_train_final, df_train_omni['diagnosis'])
    acc_asd = accuracy_score(y_test_asd, modelo_clinico.predict(X_test_final))

    # Inquisitor Model (Tries to learn geography in the clean environment)
    modelo_hosp = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
    modelo_hosp.fit(X_train_final, df_train_omni['source'])
    acc_hosp = accuracy_score(y_test_hosp, modelo_hosp.predict(X_test_final))

    azar_hosp = 1.0 / n_hospitales

    print(f"\n" + "="*60)
    print(f"📊 FINAL RESULTS (EMPIRICAL MDS - P75):")
    print(f"   |-- Clinical Accuracy (Autism): {acc_asd*100:.2f}%")
    print(f"   |-- Inquisitor Accuracy (Hospital): {acc_hosp*100:.2f}% (Chance: {azar_hosp*100:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    ejecutar_mds_empirico()
