# -*- coding: utf-8 -*-
"""
16_negative_binomial_mitigation.py

BIOINFORMATIC MITIGATION V2: WSS + NEGATIVE BINOMIAL
Objective: Impute toxic bacteria respecting natural overdispersion,
with safe handling of NaNs and zero-means.

---------------------------------------------------------------------------
THE NOVELTY: WHY NEGATIVE BINOMIAL?
---------------------------------------------------------------------------
Unlike a standard Gaussian distribution, microbiome sequencing data
consists of counts that are naturally "overdispersed" (variance > mean).
The Negative Binomial (NB) distribution is the gold standard in bioinformatics
(used in tools like DESeq2) to model this specific noise.

MATHEMATICAL ENGINE & FALLBACKS:
1. Overdispersion Check: We calculate the mean ($\mu$) and variance ($\sigma^2$).
2. NB Parameters: If $\sigma^2 > \mu$, we calculate success probability ($p$)
   and number of successes ($n$):
   $p = \frac{\mu}{\sigma^2}$
   $n = \frac{\mu^2}{\sigma^2 - \mu}$
3. Poisson Fallback: If the bacteria has low variance ($\sigma^2 \le \mu$),
   the NB math breaks. The script safely catches this and falls back to a
   Poisson($\mu$) distribution.
4. Zero-Mean Fallback: If the mean is 0 or NaN, it forces absolute zeros
   to prevent computational crashes.

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

def ejecutar_binomial_negativa():
    print("\n" + "🧬"*30)
    print(" STARTING BIOINFORMATIC IMPUTATION (NEGATIVE BINOMIAL) ")
    print(" Modeling natural sequencing overdispersion...")
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
        s_batch = [(df_train[df_train['source'] == h][feat] <= epsilon_cero).mean() for h in hospitales if len(df_train[df_train['source'] == h]) > 0]
        var_batch = np.var(s_batch)
        wss_resultados.append({'Bacteria': feat, 'WSS': s_global * var_batch})

    df_wss = pd.DataFrame(wss_resultados)

    # Based on previous success, we will use the 75th Percentile as the cutoff point
    umbral_wss = np.percentile(df_wss['WSS'], 75)
    print(f"📊 Evaluating imputation for bacteria above the 75th Percentile of WSS toxicity...\n")

    df_train_mod = df_train.copy()
    bacterias_imputadas = 0
    bacterias_fallback = 0
    bacterias_vacias = 0

    # --- 2. THE SHIELDED NEGATIVE BINOMIAL ENGINE ---
    for feat in features:
        wss_actual = df_wss[df_wss['Bacteria'] == feat]['WSS'].values[0]

        if wss_actual >= umbral_wss:
            # Extract mean and variance handling NaNs
            mu = np.nanmean(df_train_mod[feat])
            var = np.nanvar(df_train_mod[feat])

            # Safety filter: If the bacteria is virtually absent
            if np.isnan(mu) or mu <= 0.0:
                df_train_mod[feat] = 0.0
                bacterias_vacias += 1
                continue

            factor_escala = 100000.0 if np.nanmax(df_train_mod[feat]) <= 1.5 else 1.0

            mu_scaled = mu * factor_escala
            var_scaled = var * (factor_escala ** 2)

            if var_scaled > mu_scaled and mu_scaled > 0:
                p = mu_scaled / var_scaled
                n = (mu_scaled ** 2) / (var_scaled - mu_scaled)

                # Generate synthetic counts
                valores_sinteticos = np.random.negative_binomial(n, p, size=len(df_train_mod))
                df_train_mod[feat] = valores_sinteticos / factor_escala
                bacterias_imputadas += 1
            else:
                # Safe fallback with Poisson
                mu_scaled = max(1e-8, mu_scaled) # Never let it be exactly 0 or negative here
                valores_sinteticos = np.random.poisson(mu_scaled, size=len(df_train_mod))
                df_train_mod[feat] = valores_sinteticos / factor_escala
                bacterias_fallback += 1

    print(f"✔️ Bacteria imputed with pure Negative Binomial: {bacterias_imputadas}")
    print(f"⚠️ Bacteria imputed with Poisson (low variance fallback): {bacterias_fallback}")
    print(f"💀 Empty bacteria (mean 0): {bacterias_vacias}")
    print(f"🛡️ Biologically intact bacteria (low WSS): {len(features) - (bacterias_imputadas + bacterias_fallback + bacterias_vacias)}")

    # --- 3. OMNIPRESENCE ---
    print(f"\n🥷 Sealing the vault: Applying Label Collision (x{n_hospitales})...")
    clones = []
    for _, row in df_train_mod.iterrows():
        for hosp in hospitales:
            clon = row.copy()
            clon['source'] = hosp
            clon['Batch'] = 'SINTETICO_NB'
            clones.append(clon)

    df_train_omni = pd.DataFrame(clones)

    # --- 4. TRAINING AND TESTING ---
    print("⚙️ Training evaluation models (LightGBM)...")
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
    print(f"📊 FINAL RESULTS (NEGATIVE BINOMIAL IMPUTATION - P75):")
    print(f"   |-- Clinical Accuracy (Autism): {acc_asd*100:.2f}%")
    print(f"   |-- Inquisitor Accuracy (Hospital): {acc_hosp*100:.2f}% (Chance: {azar_hosp*100:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    ejecutar_binomial_negativa()
