# -*- coding: utf-8 -*-
"""
11_synthetic_tabular_generator.py

SYNTHETIC TABULAR GENERATOR V3 (CLONING + STRICT CLINICAL VALIDATION)
Objective: Annihilate batch identification using clones, but rigorously
evaluating that the ASD vs NT prediction remains intact on a real test set.

PREREQUISITES:
- Python 3.8+
- Libraries: pandas, numpy, scikit-learn
- Required Input Files (in DIR_INPUT):
  * DATA_TRAIN_READY.csv
  * DATA_TEST_READY.csv
  * CONSENSO_BIOMARCADORES.csv
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

DIR_INPUT = "1_INPUT_DATA"

def ejecutar_inyeccion_clonada():
    print("\n" + "☢️"*30)
    print(" STARTING TABULAR IMPURITY BOMB (DOUBLE LAYER) ")
    print(" Objective 1: Protect Clinical Accuracy (ASD vs NT)")
    print(" Objective 2: Destroy Inquisitor Accuracy (Hospital)")
    print("☢️"*30 + "\n")

    # 1. Load Data (Real Train and Test sets)
    df_train = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TRAIN_READY.csv"))
    df_test = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TEST_READY.csv"))
    df_consenso = pd.read_csv(os.path.join(DIR_INPUT, "CONSENSO_BIOMARCADORES.csv"))

    features = [f for f in df_consenso[df_consenso.columns[0]].head(20).tolist() if f in df_train.columns]
    hospitales = df_train['source'].unique()
    diagnosticos = df_train['diagnosis'].unique()

    n_original = len(df_train)
    n_quimeras_base = int(n_original * 0.20)

    print(f"📊 Original patients (Train): {n_original}")
    print(f"🧬 Base chimeras to fabricate: {n_quimeras_base}")
    print(f"👯 Total clones to inject (x4): {n_quimeras_base * 4}")

    # 2. Sparsity Mapping (Only on Train)
    mapa_esparsidad = {}
    for hosp in hospitales:
        df_hosp = df_train[df_train['source'] == hosp][features]
        mapa_esparsidad[hosp] = (df_hosp == 0).mean().to_dict()

    # 3. Means and Standard Deviations (ASD/NT Biology)
    stats_bio = {}
    for diag in diagnosticos:
        df_diag = df_train[df_train['diagnosis'] == diag][features]
        stats_bio[diag] = {
            'mean': df_diag.mean().to_dict(),
            'std': df_diag.std().to_dict()
        }

    # 4. Fabrication and Cloning
    print("\n🧪 Generating clones and protecting the biological label...")
    datos_sinteticos = []

    for _ in range(n_quimeras_base):
        # CLINICAL PROTECTION: The chimera is born with a real diagnosis and coherent biological values
        diag_elegido = np.random.choice(diagnosticos)
        hosp_mascara = np.random.choice(hospitales)

        perfil_base = {'diagnosis': diag_elegido, 'Batch': 'SINTETICO_CLON'}

        for feat in features:
            mu = stats_bio[diag_elegido]['mean'][feat]
            sigma = stats_bio[diag_elegido]['std'][feat]
            valor = np.random.normal(mu, sigma)
            valor = np.maximum(0, valor)

            if np.random.rand() < mapa_esparsidad[hosp_mascara][feat]:
                valor = 0.0

            perfil_base[feat] = valor

        # ADVERSARIAL CLONING: We copy the same biology, but lie about the hospital
        etiquetas_falsas = np.random.choice(hospitales, size=4, replace=False)
        for etiqueta in etiquetas_falsas:
            clon = perfil_base.copy()
            clon['source'] = etiqueta
            datos_sinteticos.append(clon)

    df_sintetico = pd.DataFrame(datos_sinteticos)
    for col in df_train.columns:
        if col not in df_sintetico.columns:
            df_sintetico[col] = np.nan

    # 5. Merge (Poisoned Training Dataset)
    df_train_ampliado = pd.concat([df_train, df_sintetico], ignore_index=True)

    # ==========================================
    # DOUBLE LAYER TEST (ON THE REAL TEST SET)
    # ==========================================
    print("\n⚖️ Evaluating on 100% Real and Unseen patients (DATA_TEST_READY.csv)...")

    X_test_real = df_test[features]
    y_test_asd = df_test['diagnosis']
    y_test_hosp = df_test['source']

    # --- MODEL A: THE CLINICAL ONE (Predicts ASD vs NT) ---
    rf_clinico_orig = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clinico_orig.fit(df_train[features], df_train['diagnosis'])
    acc_asd_orig = accuracy_score(y_test_asd, rf_clinico_orig.predict(X_test_real))

    rf_clinico_sint = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clinico_sint.fit(df_train_ampliado[features], df_train_ampliado['diagnosis'])
    acc_asd_sint = accuracy_score(y_test_asd, rf_clinico_sint.predict(X_test_real))

    # --- MODEL B: THE INQUISITOR (Predicts Hospital) ---
    rf_hosp_orig = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_hosp_orig.fit(df_train[features], df_train['source'])
    acc_hosp_orig = accuracy_score(y_test_hosp, rf_hosp_orig.predict(X_test_real))

    rf_hosp_sint = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_hosp_sint.fit(df_train_ampliado[features], df_train_ampliado['source'])
    acc_hosp_sint = accuracy_score(y_test_hosp, rf_hosp_sint.predict(X_test_real))

    azar = 1.0 / len(hospitales)

    print("\n" + "="*50)
    print("📊 CLINICAL RESULTS (ASD vs NT):")
    print(f"   |-- Original Accuracy:  {acc_asd_orig*100:.2f}%")
    print(f"   |-- Synthetic Accuracy: {acc_asd_sint*100:.2f}% (We want it to maintain or increase!)")

    print("\n📊 INQUISITOR RESULTS (Hospital):")
    print(f"   |-- Original Accuracy:  {acc_hosp_orig*100:.2f}%")
    print(f"   |-- Synthetic Accuracy: {acc_hosp_sint*100:.2f}% (We want it to drop towards {azar*100:.2f}%)")
    print("="*50)

if __name__ == "__main__":
    ejecutar_inyeccion_clonada()
