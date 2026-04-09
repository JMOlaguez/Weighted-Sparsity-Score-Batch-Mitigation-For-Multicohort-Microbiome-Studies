# -*- coding: utf-8 -*-
"""
12_label_anonymizer.py

LABEL ANONYMIZER (ADVERSARIAL COLLISION)
Objective: Clone real patients changing ONLY their hospital label.
Destroys the model's ability to associate biological profiles with geography,
without altering the Autism biological signal.

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

def ejecutar_anonimizacion():
    print("\n" + "🛡️"*30)
    print(" STARTING ADVERSARIAL ANONYMIZATION (LABEL COLLISION) ")
    print(" Strategy: Duplicate real patients and lie about their Hospital")
    print("🛡️"*30 + "\n")

    # 1. Load Data (Pure Train and Test sets)
    df_train = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TRAIN_READY.csv"))
    df_test = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TEST_READY.csv"))

    # We will use all available biological columns to recover that ~80%
    excluir = ['diagnosis', 'source', 'Batch', 'SampleID']
    features = [c for c in df_train.columns if c not in excluir]

    hospitales = df_train['source'].unique()

    print(f"📊 Original patients (Train): {len(df_train)}")

    # 2. The User's Strategy: Cloning and Geographic Lying
    print("🥷 Anonymizing: Cloning patients and altering the zip code...")
    clones = []

    for _, row in df_train.iterrows():
        hosp_real = row['source']
        otros_hospitales = [h for h in hospitales if h != hosp_real]

        # We randomly choose 2 different hospitals for this patient
        hospitales_falsos = np.random.choice(otros_hospitales, size=2, replace=False)

        for hosp_falso in hospitales_falsos:
            clon = row.copy()
            clon['source'] = hosp_falso
            clon['Batch'] = 'ANONIMIZADO'  # ANONYMIZED
            clones.append(clon)

    df_clones = pd.DataFrame(clones)

    # The new training dataset has the real ones + the clones
    df_train_anonimizado = pd.concat([df_train, df_clones], ignore_index=True)

    print(f"👯 Injected clones: {len(df_clones)}")
    print(f"💾 New Anonymized Train Set: {len(df_train_anonimizado)} patients")

    # ==========================================
    # DOUBLE LAYER TEST (ON THE PURE TEST SET)
    # ==========================================
    print("\n⚖️ Evaluating on TEST SET (Pure, Real, Unseen Patients)...")

    X_test = df_test[features]
    y_test_asd = df_test['diagnosis']
    y_test_hosp = df_test['source']

    # We use LightGBM, which gave the best global results (~80%)
    modelo_clinico_orig = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    modelo_clinico_anon = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

    modelo_hosp_orig = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    modelo_hosp_anon = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

    # --- ORIGINAL TRAINING (Unanonymized) ---
    X_train_orig = df_train[features]
    modelo_clinico_orig.fit(X_train_orig, df_train['diagnosis'])
    acc_asd_orig = accuracy_score(y_test_asd, modelo_clinico_orig.predict(X_test))

    modelo_hosp_orig.fit(X_train_orig, df_train['source'])
    acc_hosp_orig = accuracy_score(y_test_hosp, modelo_hosp_orig.predict(X_test))

    # --- ANONYMIZED TRAINING (The user's antidote) ---
    X_train_anon = df_train_anonimizado[features]
    modelo_clinico_anon.fit(X_train_anon, df_train_anonimizado['diagnosis'])
    acc_asd_anon = accuracy_score(y_test_asd, modelo_clinico_anon.predict(X_test))

    modelo_hosp_anon.fit(X_train_anon, df_train_anonimizado['source'])
    acc_hosp_anon = accuracy_score(y_test_hosp, modelo_hosp_anon.predict(X_test))

    azar = 1.0 / len(hospitales)

    print("\n" + "="*55)
    print("📊 OBJECTIVE 1: RECOVER AND PROTECT DIAGNOSIS (ASD vs NT)")
    print(f"   |-- Original Accuracy:   {acc_asd_orig*100:.2f}%")
    print(f"   |-- Anonymized Accuracy: {acc_asd_anon*100:.2f}%")

    print("\n📊 OBJECTIVE 2: DESTROY THE INQUISITOR (Guess Hospital)")
    print(f"   |-- Original Accuracy:   {acc_hosp_orig*100:.2f}%")
    print(f"   |-- Anonymized Accuracy: {acc_hosp_anon*100:.2f}% (Chance: {azar*100:.2f}%)")
    print("="*55)

    if acc_hosp_anon < 30.0:
        print("\n✅ BRUTAL! The hospital algorithm collapsed due to label collision.")

if __name__ == "__main__":
    ejecutar_anonimizacion()
