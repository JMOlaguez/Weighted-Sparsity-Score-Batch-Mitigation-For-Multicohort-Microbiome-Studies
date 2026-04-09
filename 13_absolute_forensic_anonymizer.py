# -*- coding: utf-8 -*-
"""
13_absolute_forensic_anonymizer.py

ABSOLUTE ANONYMIZER AND FORENSIC ANALYZER (OMNIPRESENT POISONING)
Objective: Clone each patient N times (where N = total hospitals).
Extract and plot exactly which variables the Inquisitor uses to cheat.

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

def ejecutar_anonimizacion_absoluta():
    print("\n" + "☢️"*30)
    print(" STARTING OMNIPRESENT POISONING (N-CLONES) ")
    print(" Strategy: Each patient will exist in ALL hospitals simultaneously.")
    print("☢️"*30 + "\n")

    # 1. Load Data
    df_train = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TRAIN_READY.csv"))
    df_test = pd.read_csv(os.path.join(DIR_INPUT, "DATA_TEST_READY.csv"))

    excluir = ['diagnosis', 'source', 'Batch', 'SampleID']
    features = [c for c in df_train.columns if c not in excluir]

    hospitales = df_train['source'].unique()
    n_hospitales = len(hospitales)

    print(f"📊 Original patients (Train): {len(df_train)}")
    print(f"🏥 Number of independent hospitals (N): {n_hospitales}")

    # 2. The Omnipresent Poisoning
    print(f"🥷 Multiplying the dataset by {n_hospitales}... (This might take a few seconds)")
    clones = []

    # We iterate row by row and create a copy for EACH hospital
    for _, row in df_train.iterrows():
        for hosp in hospitales:
            clon = row.copy()
            clon['source'] = hosp
            clon['Batch'] = 'OMNIPRESENTE' # OMNIPRESENT
            clones.append(clon)

    df_train_omni = pd.DataFrame(clones)
    print(f"💾 New Absolute Train Set: {len(df_train_omni)} patients")

    # ==========================================
    # DOUBLE LAYER TEST ON THE TEST SET
    # ==========================================
    print("\n⚖️ Training models and evaluating on TEST SET (Pure Patients)...")

    X_test = df_test[features]
    y_test_asd = df_test['diagnosis']
    y_test_hosp = df_test['source']

    # --- CLINICAL TRAINING (ASD vs NT) ---
    modelo_clinico = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
    modelo_clinico.fit(df_train_omni[features], df_train_omni['diagnosis'])
    acc_asd = accuracy_score(y_test_asd, modelo_clinico.predict(X_test))

    # --- INQUISITOR TRAINING (Hospital) ---
    modelo_hosp = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
    modelo_hosp.fit(df_train_omni[features], df_train_omni['source'])
    acc_hosp = accuracy_score(y_test_hosp, modelo_hosp.predict(X_test))

    azar = 1.0 / n_hospitales

    print("\n" + "="*55)
    print("📊 OBJECTIVE 1: DIAGNOSIS (ASD vs NT) IN OMNIPRESENT DATASET")
    print(f"   |-- Clinical Accuracy: {acc_asd*100:.2f}%")

    print("\n📊 OBJECTIVE 2: DESTROY THE INQUISITOR")
    print(f"   |-- Hospital Accuracy: {acc_hosp*100:.2f}% (Chance Limit: {azar*100:.2f}%)")
    print("="*55)

    if acc_hosp < (azar + 0.10): # If it falls near chance
        print("\n✅ KNOCKOUT! The math collapsed. The model is unable to read the geography.")
    else:
        print("\n⚠️ The Inquisitor is still resisting. Check the forensic plot.")

    # ==========================================
    # FORENSIC ANALYSIS (WHAT THE HELL IS IT LOOKING AT?)
    # ==========================================
    print("\n🔬 Executing Inquisitor's Forensic Autopsy...")

    # We extract the Inquisitor's importances
    importancias = modelo_hosp.feature_importances_
    df_imp = pd.DataFrame({
        'Bacteria/Pathway': features,
        'Importance': importancias
    }).sort_values(by='Importance', ascending=False).head(20) # Top 20

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Bacteria/Pathway', data=df_imp, palette='magma')
    plt.title('FORENSIC ANALYSIS: Top 20 Variables the Inquisitor uses to guess the Hospital', fontweight='bold')
    plt.xlabel('LightGBM Importance (Gain/Splits)')
    plt.ylabel('')
    plt.tight_layout()

    ruta_grafico = os.path.join(DIR_PLOTS, "9_Forensic_Inquisitor.png")
    plt.savefig(ruta_grafico, dpi=300)
    plt.close()

    print(f"📸 X-Ray saved at: {ruta_grafico}")
    print("   (Open this image to see the bacteria that are giving us away)")

if __name__ == "__main__":
    ejecutar_anonimizacion_absoluta()
