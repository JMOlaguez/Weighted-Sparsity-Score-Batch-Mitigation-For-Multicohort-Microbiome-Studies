# -*- coding: utf-8 -*-
"""
19b_causal_metrics_export.py

PHASE 3.1: CAUSAL ALIGNMENT AND FORENSIC METRICS EXPORT
Objective: Impute pseudo-zeros, collapse hospital prediction, and
save the isolated results for comparative analysis.

---------------------------------------------------------------------------
THE TACTICAL BRIDGE: WHY THIS SCRIPT EXISTS
---------------------------------------------------------------------------
Before deploying the massive Orchestrators (which test dozens of models and
take significant compute time), we need a lightweight, reproducible way to
prove that Phase 3 (Causal Alignment of Structural Zeros) actually worked.

This script acts as a "Forensic Exporter":
1. It applies the surgical WSS + Positive MDS imputation exclusively to dropouts.
2. It floods the data with Omnipresent clones (Label Collision).
3. It immediately tests this sterilized data using a fast, robust baseline
   model (LightGBM).
4. Most importantly: It EXPORTS these metrics to a CSV file (`sterilized_metrics.csv`).

This exported artifact is critical. It allows you to quickly build visual
comparisons (e.g., Bar charts showing the Inquisitor's accuracy dropping from
99% to chance levels) for your research paper or presentation without having
to re-run the heavy, computationally expensive orchestrator suites.

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
DIR_OUTPUT = "19_OUTPUT_RESULTS"

def ejecutar_alineacion_causal_v19_2():
    # Create output folder if it doesn't exist
    if not os.path.exists(DIR_OUTPUT):
        os.makedirs(DIR_OUTPUT)
        print(f"📁 Folder '{DIR_OUTPUT}' created.")

    print("\n" + "🧬"*30)
    print(" STARTING PHASE 3.1: CAUSAL ALIGNMENT + EXPORT ")
    print("🧬" * 30 + "\n")

    # 1. Load Data
    df_train = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TRAIN_READY.csv"))
    df_test = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TEST_READY.csv"))

    excluir = ['diagnosis', 'source', 'Batch', 'SampleID']
    features = [c for c in df_train.columns if c not in excluir]

    hospitales = df_train['source'].unique()
    n_hospitales = len(hospitales)
    y_test_asd = df_test['diagnosis']
    y_test_hosp = df_test['source']
    epsilon_cero = 1e-5

    # --- 2. WSS CALCULATION (Technical Toxicity Identification) ---
    wss_resultados = []
    for feat in features:
        s_global = (df_train[feat] <= epsilon_cero).mean()
        s_batch = [(df_train[df_train['source'] == h][feat] <= epsilon_cero).mean()
                   for h in hospitales if len(df_train[df_train['source'] == h]) > 0]
        var_batch = np.var(s_batch)
        wss_resultados.append({'Bacteria': feat, 'WSS': s_global * var_batch})

    df_wss = pd.DataFrame(wss_resultados)
    umbral_wss = np.percentile(df_wss['WSS'], 75)

    print(f"📊 WSS Threshold (P75): {umbral_wss:.4f}")

    # --- 3. POSITIVE MDS IMPUTATION ENGINE ---
    df_train_mod = df_train.copy()
    bacterias_intervenidas = 0

    for feat in features:
        wss_actual = df_wss[df_wss['Bacteria'] == feat]['WSS'].values[0]
        if wss_actual >= umbral_wss:
            idx_ceros = df_train_mod[feat] <= epsilon_cero
            idx_positivos = df_train_mod[feat] > epsilon_cero

            if idx_ceros.any() and idx_positivos.any():
                dist_positiva = df_train_mod.loc[idx_positivos, feat].values
                relleno = np.random.choice(dist_positiva, size=idx_ceros.sum(), replace=True)
                df_train_mod.loc[idx_ceros, feat] = relleno
                bacterias_intervenidas += 1

    # --- 4. LABEL COLLISION (OMNIPRESENCE) ---
    print(f"🥷 Applying Label Collision (Cloning x{n_hospitales})...")
    clones = []
    for _, row in df_train_mod.iterrows():
        for hosp in hospitales:
            clon = row.copy()
            clon['source'] = hosp
            clones.append(clon)
    df_train_omni = pd.DataFrame(clones)

    # --- 5. VALIDATION MODELS TRAINING ---
    X_train_final = df_train_omni[features]
    X_test_final = df_test[features]

    # Clinical Model (ASD)
    m_clinico = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    m_clinico.fit(X_train_final, df_train_omni['diagnosis'])
    acc_asd = accuracy_score(y_test_asd, m_clinico.predict(X_test_final))

    # Inquisitor Model (Hospital)
    m_hosp = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    m_hosp.fit(X_train_final, df_train_omni['source'])
    acc_hosp = accuracy_score(y_test_hosp, m_hosp.predict(X_test_final))

    # --- 6. SAVING METRICS FOR THE COMPARATOR ---
    metricas = pd.DataFrame([
        {"Metric": "Biological_Accuracy_ASD", "Value": acc_asd},
        {"Metric": "Adversary_Accuracy_Hospital", "Value": acc_hosp}
    ])

    ruta_csv = os.path.join(DIR_OUTPUT, "sterilized_metrics.csv")
    metricas.to_csv(ruta_csv, index=False)

    print(f"\n✅ Metrics saved at: {ruta_csv}")
    print("="*60)
    print(f"📊 FINAL RESULTS:")
    print(f"   |-- Clinical Accuracy (ASD): {acc_asd*100:.2f}%")
    print(f"   |-- Inquisitor Accuracy (Hosp): {acc_hosp*100:.2f}% (Chance: {(1/n_hospitales)*100:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    ejecutar_alineacion_causal_v19_2()
