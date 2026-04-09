# -*- coding: utf-8 -*-
"""
14_sparsity_mitigation.py

SPARSITY FOOTPRINT ELIMINATION V4 (X-RAY AND GAUSSIAN RAY)
Objective: Map real sparsity, drop hyper-sparse columns, and
overwrite moderately sparse columns with Gaussian distributions.

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

def ejecutar_mitigacion_v4():
    print("\n" + "🦠"*30)
    print(" STARTING PIPELINE V4: SPARSITY X-RAY AND GAUSSIAN RAY ")
    print("🦠"*30 + "\n")

    df_train = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TRAIN_READY.csv"))
    df_test = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TEST_READY.csv"))

    excluir = ['diagnosis', 'source', 'Batch', 'SampleID']
    features_originales = [c for c in df_train.columns if c not in excluir]

    hospitales = df_train['source'].unique()
    n_hospitales = len(hospitales)

    y_test_asd = df_test['diagnosis']
    y_test_hosp = df_test['source']
    azar_hosp = 1.0 / n_hospitales
    epsilon_cero = 1e-5

    # --- STEP 0: ORIGINAL DATASET X-RAY ---
    print("🔍 ORIGINAL DATASET X-RAY:")
    esparsidad_dict = {}
    for feat in features_originales:
        prop_ceros = (df_train[feat] <= epsilon_cero).mean()
        esparsidad_dict[feat] = prop_ceros

    df_esp = pd.DataFrame(list(esparsidad_dict.items()), columns=['Bacteria', 'Sparsity'])

    print(f"   |-- Bacteria with >90% pseudo-zeros: {len(df_esp[df_esp['Sparsity'] > 0.90])}")
    print(f"   |-- Bacteria with 70-90% pseudo-zeros: {len(df_esp[(df_esp['Sparsity'] > 0.70) & (df_esp['Sparsity'] <= 0.90)])}")
    print(f"   |-- Bacteria with 50-70% pseudo-zeros: {len(df_esp[(df_esp['Sparsity'] > 0.50) & (df_esp['Sparsity'] <= 0.70)])}")
    print(f"   |-- Bacteria with 30-50% pseudo-zeros: {len(df_esp[(df_esp['Sparsity'] > 0.30) & (df_esp['Sparsity'] <= 0.50)])}")
    print(f"   |-- Bacteria with <30% pseudo-zeros: {len(df_esp[df_esp['Sparsity'] <= 0.30])}")

    # ADJUSTED THRESHOLDS (To avoid deleting everything)
    umbrales_corte = [0.95, 0.80, 0.60]

    # "Gaussian Ray" Range: We overwrite columns that have between 15% and the cutoff threshold
    umbral_gaussiano_min = 0.15

    for umbral_X in umbrales_corte:
        print("\n" + "="*65)
        print(f"🚀 EXPERIMENT: CUTOFF THRESHOLD = {umbral_X * 100}% ZEROS")
        print("="*65)

        df_train_mod = df_train.copy()

        # --- LAYER 1: CUT AND GAUSSIAN RAY ---
        features_a_mantener = []
        columnas_borradas = 0

        for feat in features_originales:
            prop_ceros = esparsidad_dict[feat]
            if prop_ceros > umbral_X:
                columnas_borradas += 1
            else:
                features_a_mantener.append(feat)

        print(f"🗑️ Dropped columns (>{umbral_X*100}% zeros): {columnas_borradas}")

        columnas_imputadas = 0
        for feat in features_a_mantener:
            prop_ceros = esparsidad_dict[feat]

            # If it's in the danger zone (e.g. between 15% and 80% zeros)
            if prop_ceros > umbral_gaussiano_min:
                mu = df_train_mod[feat].mean()
                sigma = df_train_mod[feat].std()

                # TOTAL replacement of the column with global Gaussian noise
                valores_sinteticos = np.random.normal(mu, sigma, size=len(df_train_mod))
                valores_sinteticos = np.maximum(0, valores_sinteticos)
                df_train_mod[feat] = valores_sinteticos
                columnas_imputadas += 1

        print(f"💉 Columns overwritten with total Gaussian noise: {columnas_imputadas}")

        if len(features_a_mantener) == 0:
            print("⚠️ WARNING: ALL columns were dropped. Skipping...")
            continue

        # --- LAYER 2: OMNIPRESENCE (N-TIMES CLONING) ---
        print(f"🥷 Applying Label Collision (x{n_hospitales})...")
        clones = []
        for _, row in df_train_mod.iterrows():
            for hosp in hospitales:
                clon = row.copy()
                clon['source'] = hosp
                clon['Batch'] = 'SINTETICO_OMNI' # SYNTHETIC_OMNI
                clones.append(clon)

        df_train_omni = pd.DataFrame(clones)

        # --- TRAINING AND TESTING ---
        X_train_final = df_train_omni[features_a_mantener]
        X_test_final = df_test[features_a_mantener]

        modelo_clinico = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
        modelo_clinico.fit(X_train_final, df_train_omni['diagnosis'])
        acc_asd = accuracy_score(y_test_asd, modelo_clinico.predict(X_test_final))

        modelo_hosp = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
        modelo_hosp.fit(X_train_final, df_train_omni['source'])
        acc_hosp = accuracy_score(y_test_hosp, modelo_hosp.predict(X_test_final))

        print(f"\n📊 FINAL RESULTS FOR THRESHOLD {umbral_X * 100}%:")
        print(f"   |-- Clinical Accuracy (Autism): {acc_asd*100:.2f}%")
        print(f"   |-- Inquisitor Accuracy (Hospital): {acc_hosp*100:.2f}% (Chance: {azar_hosp*100:.2f}%)")

if __name__ == "__main__":
    ejecutar_mitigacion_v4()
