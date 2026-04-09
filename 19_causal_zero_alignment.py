# -*- coding: utf-8 -*-
"""
19_causal_zero_alignment.py

PHASE 3: CAUSAL ALIGNMENT AND STRUCTURAL ZEROS
Objective: EXCLUSIVELY impute the pseudo-zeros of toxic bacteria
using the positive empirical distribution, protecting real measurements.

---------------------------------------------------------------------------
THE NOVELTY: WHY CAUSAL ZERO ALIGNMENT? (TARGETING DROPOUTS)
---------------------------------------------------------------------------
In microbiome sequencing, a "zero" can mean two things:
1. Biological Zero: The bacteria is truly absent in the patient.
2. Structural Zero (Dropout): The bacteria was there, but the sequencer
   in a specific hospital missed it due to kit differences or shallow depth.

Previous methods (like Variance Stabilization or full MDS) modify the entire
column of a toxic bacteria. But what if the positive measurements (values > 0)
are actually highly accurate, and the Batch Effect is solely driven by the
abundance of dropouts (pseudo-zeros) in specific hospitals?

The Causal Alignment Strategy:
1. Identify the toxic bacteria using WSS (High geographic variance).
2. Isolate ONLY the zero values (the "black holes").
3. Extract the global empirical distribution of ONLY the POSITIVE values.
4. Resample from these positive values to fill in the zeroes.

Explanation: We preserve 100% of the real, positive biological measurements.
We only surgically repair the technical "blind spots" (dropouts) that the
Inquisitor model was using to cheat.

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

def ejecutar_alineacion_causal():
    print("\n" + "🧬"*30)
    print(" STARTING PHASE 3: CAUSAL ALIGNMENT OF STRUCTURAL ZEROS ")
    print(" Strategy: Exclusive MDS imputation for pseudo-zeros (black holes).")
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

    umbral_wss = np.percentile(df_wss['WSS'], 75)
    print(f"📊 Targeting the zeros of the top 25% most toxic bacteria (WSS > {umbral_wss:.4f})...\n")

    df_train_mod = df_train.copy()
    bacterias_intervenidas = 0
    pseudo_ceros_imputados = 0

    # --- 2. THE CAUSAL IMPUTATION ENGINE (POSITIVE MDS) ---
    for feat in features:
        wss_actual = df_wss[df_wss['Bacteria'] == feat]['WSS'].values[0]

        if wss_actual >= umbral_wss:
            # Boolean masks
            idx_ceros = df_train_mod[feat] <= epsilon_cero
            idx_positivos = df_train_mod[feat] > epsilon_cero

            n_ceros = idx_ceros.sum()

            if n_ceros > 0 and idx_positivos.sum() > 0:
                # We extract the global distribution of ONLY the real/positive values
                distribucion_positiva = df_train_mod.loc[idx_positivos, feat].values

                # We resample to fill the zeros (without touching the positive ones)
                valores_relleno = np.random.choice(distribucion_positiva, size=n_ceros, replace=True)

                # We inject the filler
                df_train_mod.loc[idx_ceros, feat] = valores_relleno

                bacterias_intervenidas += 1
                pseudo_ceros_imputados += n_ceros

    print(f"✔️ Causally intervened bacteria: {bacterias_intervenidas}")
    print(f"🕳️ Technical pseudo-zeros eliminated and replaced with global biology: {pseudo_ceros_imputados}")
    print(f"🛡️ Biologically intact bacteria (low WSS): {len(features) - bacterias_intervenidas}")

    # --- 3. OMNIPRESENCE (Label Collision) ---
    print(f"\n🥷 Applying Label Collision (Cloning x{n_hospitales})...")
    clones = []
    for _, row in df_train_mod.iterrows():
        for hosp in hospitales:
            clon = row.copy()
            clon['source'] = hosp
            clon['Batch'] = 'SINTETICO_CAUSAL' # SYNTHETIC_CAUSAL
            clones.append(clon)

    df_train_omni = pd.DataFrame(clones)

    # --- 4. TRAINING AND TESTING ---
    print("⚙️ Training NEW models on the causally cured dataset...")
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
    print(f"📊 FINAL RESULTS (CAUSAL ALIGNMENT - P75):")
    print(f"   |-- Clinical Accuracy (Autism): {acc_asd*100:.2f}%")
    print(f"   |-- Inquisitor Accuracy (Hospital): {acc_hosp*100:.2f}% (Chance: {azar_hosp*100:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    ejecutar_alineacion_causal()
