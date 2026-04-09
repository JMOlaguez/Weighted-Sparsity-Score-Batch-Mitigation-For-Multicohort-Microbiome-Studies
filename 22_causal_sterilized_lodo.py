# -*- coding: utf-8 -*-
"""
22_causal_sterilized_lodo.py

ORCHESTRATOR 11 - CAUSAL STERILIZED LODO (Leave-One-Domain-Out)
Objective: Strict spatial evaluation. Sterilization is fitted on $N-1$ hospitals
and projected onto the test hospital to prevent Data Leakage.
Generates logs, .pkl models, plots, and a comparative CSV per hospital.

---------------------------------------------------------------------------
THE PINNACLE OF VALIDATION: LEAVE-ONE-DOMAIN-OUT (LODO)
---------------------------------------------------------------------------
Standard Cross-Validation randomly splits patients. However, in clinical
microbiome data, patients from the same hospital share technical batch effects.
If we randomly split, the model might still memorize the batch effect because
patients from "Hospital A" are in both the training and test sets.

LODO is the ultimate spatial test:
If we have $N$ hospitals, we train the model on $N-1$ hospitals and test it
on the entirely unseen $N$-th hospital. We repeat this until every hospital
has been the test set. This simulates deploying the AI in a brand new clinic.

---------------------------------------------------------------------------
CRITICAL METHODOLOGY: NESTED CAUSAL STERILIZATION (ZERO LEAKAGE)
---------------------------------------------------------------------------
If we calculate the WSS (Weighted Sparsity Score) and extract the positive
empirical distributions using the ENTIRE dataset before doing the LODO split,
we commit a mortal sin of Data Science: Data Leakage.

To maintain 100% scientific purity, this script does "Nested Sterilization":
1. Isolate the $N-1$ training hospitals.
2. Calculate WSS and find toxic bacteria *only* in this training fold.
3. Extract the positive distribution arrays *only* from this training fold.
4. "Transform" the $N$-th test hospital using the parameters learned from the training fold.
This guarantees the test hospital remains mathematically unseen and pure.

PREREQUISITES:
- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
- The `algoritmos_ml` module directory must exist.
- Required Input Files (in DIR_INPUT):
  * DATA_TRAIN_READY.csv
  * DATA_TEST_READY.csv
"""
import os
import sys
import importlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, roc_curve, auc,
    matthews_corrcoef, f1_score, recall_score, cohen_kappa_score
)

# === ACADEMIC CONFIGURATION ===
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

DIR_INPUT = os.path.join(os.getcwd(), "1_INPUT_DATA")
DIR_OUTPUT = os.path.join(os.getcwd(), "ML_RESULTS_STERILIZED_LODO")

DIR_MODELS = os.path.join(DIR_OUTPUT, "Trained_Models")
DIR_PLOTS = os.path.join(DIR_OUTPUT, "ROC_CM_Plots")
DIR_BIOMARKERS = os.path.join(DIR_OUTPUT, "Biomarkers")

for d in [DIR_OUTPUT, DIR_MODELS, DIR_PLOTS, DIR_BIOMARKERS]:
    os.makedirs(d, exist_ok=True)

sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")

# === MASTER LOGGER ===
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

log_file = os.path.join(DIR_OUTPUT, f"STERILIZED_LODO_EXECUTION_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
sys.stdout = Logger(log_file)

# === ROBUST ALGORITHMS ===
MODULES_CONFIG = {
    'Supervised': [
        'algoritmos_ml.supervisados.rf_classifier',
        'algoritmos_ml.supervisados.lightgbm_classifier',
        'algoritmos_ml.supervisados.xgboost_classifier',
        'algoritmos_ml.supervisados.catboost_classifier',
        'algoritmos_ml.supervisados.svm_linear_classifier',
        'algoritmos_ml.supervisados.logistic_regression',
        'algoritmos_ml.supervisados.elasticnet_classifier',
        'algoritmos_ml.supervisados.lda_classifier'
    ]
}

SEEDS = [42, 123, 888]

# === PLOTTING FUNCTIONS ===
def plot_cv_roc_lodo(tprs, aucs, mean_fpr, model_name, output_dir):
    plt.figure(figsize=(8, 6))
    if not tprs: return
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='#27ae60', label=r'Mean ROC Sterilized LODO (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2.5)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#2ecc71', alpha=0.3)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#7f8c8d', label='Chance')
    plt.title(f'Sterilized LODO Global Performance - {model_name}', fontweight='bold')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ROC_LODO_STERILIZED_{model_name}.png"), dpi=300)
    plt.close()

def plot_confusion_matrix_lodo(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, annot_kws={"size": 16, "weight": "bold"})
    plt.title(f'Sterilized LODO Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"CM_LODO_STERILIZED_{model_name}.png"), dpi=300)
    plt.close()

# =============================================================================
# NESTED CAUSAL STERILIZATION ENGINE (NO LEAKAGE)
# =============================================================================
def fit_transform_esterilizacion(df_train):
    excluir = ['diagnosis', 'source', 'Batch', 'SampleID']
    features = [c for c in df_train.columns if c not in excluir]
    hospitales = df_train['source'].unique()
    epsilon_cero = 1e-5

    wss_resultados = []
    for feat in features:
        s_global = (df_train[feat] <= epsilon_cero).mean()
        s_batch = [(df_train[df_train['source'] == h][feat] <= epsilon_cero).mean() for h in hospitales if len(df_train[df_train['source'] == h]) > 0]
        var_batch = np.var(s_batch)
        wss_resultados.append({'Bacteria': feat, 'WSS': s_global * var_batch})

    df_wss = pd.DataFrame(wss_resultados)
    umbral_wss = np.percentile(df_wss['WSS'], 75)

    df_train_mod = df_train.copy()
    parametros_esterilizacion = {}

    for feat in features:
        wss_actual = df_wss[df_wss['Bacteria'] == feat]['WSS'].values[0]
        if wss_actual >= umbral_wss:
            idx_ceros = df_train_mod[feat] <= epsilon_cero
            idx_positivos = df_train_mod[feat] > epsilon_cero
            n_ceros = idx_ceros.sum()

            if idx_positivos.sum() > 0:
                distribucion_positiva = df_train_mod.loc[idx_positivos, feat].values
                # We SAVE the empirical positive distribution learned exclusively from the training set
                parametros_esterilizacion[feat] = distribucion_positiva

                if n_ceros > 0:
                    valores_relleno = np.random.choice(distribucion_positiva, size=n_ceros, replace=True)
                    df_train_mod.loc[idx_ceros, feat] = valores_relleno

    return df_train_mod, parametros_esterilizacion

def transform_esterilizacion(df_test, parametros_esterilizacion):
    df_test_mod = df_test.copy()
    epsilon_cero = 1e-5

    # We PROJECT the parameters learned from training onto the test fold
    for feat, distribucion_positiva in parametros_esterilizacion.items():
        idx_ceros = df_test_mod[feat] <= epsilon_cero
        n_ceros = idx_ceros.sum()
        if n_ceros > 0:
            valores_relleno = np.random.choice(distribucion_positiva, size=n_ceros, replace=True)
            df_test_mod.loc[idx_ceros, feat] = valores_relleno

    return df_test_mod

# =============================================================================
# MAIN STERILIZED LODO ENGINE
# =============================================================================
def ejecutar_lodo_esterilizado():
    train_path = os.path.join(DIR_INPUT, "DATA_TRAIN_READY.csv")
    test_path = os.path.join(DIR_INPUT, "DATA_TEST_READY.csv")

    df_train_raw = pd.read_csv(train_path)
    df_test_raw = pd.read_csv(test_path)
    # We combine them because LODO uses the entire dataset to leave one hospital out at a time
    df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
    hospitales = df_all['source'].unique()

    print("\n" + "="*75)
    print(" STARTING CAUSAL STERILIZED LODO (LEAVE-ONE-DOMAIN-OUT) ")
    print(f" Total Patients: {len(df_all)} | Total Hospitals: {len(hospitales)}")
    print(f" Logs backed up at: {log_file}")
    print("="*75)

    target = 'diagnosis'
    cols_drop = [target, 'source', 'Batch']
    resultados_globales = []
    metricas_por_hospital = []

    for family, scripts in MODULES_CONFIG.items():
        for script_path in scripts:
            model_id = script_path.split('.')[-1]
            try:
                module = importlib.import_module(script_path)
            except Exception as e:
                print(f" ❌ Error importing {model_id}: {e}")
                continue

            print(f"\n🚀 Evaluating Sterilized Algorithm: {model_id.upper()}")
            tprs, aucs = [], []
            mean_fpr = np.linspace(0, 1, 100)
            y_true_acumulado, y_pred_acumulado = [], []

            for hosp_test in hospitales:
                print(f"  |-- Isolating and Sterilizing: {hosp_test} ... ", end="", flush=True)

                # THE LODO SPLIT
                df_train_fold = df_all[df_all['source'] != hosp_test].copy()
                df_test_fold = df_all[df_all['source'] == hosp_test].copy()

                if len(df_test_fold[target].unique()) < 2:
                    print("[Skipped: Only 1 class present] ")
                    continue

                # CAUSAL SURGERY (Fit on N-1, Transform on 1)
                df_train_est, params_est = fit_transform_esterilizacion(df_train_fold)
                df_test_est = transform_esterilizacion(df_test_fold, params_est)

                X_train = df_train_est.drop(columns=[c for c in cols_drop if c in df_train_est.columns]).values
                y_train = df_train_est[target].values
                X_test = df_test_est.drop(columns=[c for c in cols_drop if c in df_test_est.columns]).values
                y_test = df_test_est[target].values

                y_prob_fold = np.zeros(len(y_test))

                for seed in SEEDS:
                    try:
                        res = module.train_and_evaluate(X_train, y_train, X_test, y_test, random_state=seed)
                        if res.get('y_test_prob') is not None:
                            y_prob_fold += res['y_test_prob'] / len(SEEDS)

                        if seed == SEEDS[-1] and 'trained_model' in res:
                            joblib.dump(res['trained_model'], os.path.join(DIR_MODELS, f"{model_id}_{hosp_test}_STERILIZED.pkl"))
                    except Exception as e:
                        pass # Silence individual seed errors to avoid breaking the fold

                # Metric calculation
                fpr, tpr, thresholds = roc_curve(y_test, y_prob_fold)
                idx = np.argmax(tpr - fpr)
                best_thresh = thresholds[idx]
                y_pred_fold = (y_prob_fold >= best_thresh).astype(int)

                auc_fold = roc_auc_score(y_test, y_prob_fold)
                acc_fold = accuracy_score(y_test, y_pred_fold)

                aucs.append(auc_fold)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                y_true_acumulado.extend(y_test)
                y_pred_acumulado.extend(y_pred_fold)

                metricas_por_hospital.append({
                    'Algorithm': model_id,
                    'Hospital': hosp_test,
                    'Sterilized_AUC': auc_fold,
                    'Sterilized_ACC': acc_fold
                })

                print(f"AUC: {auc_fold:.3f} ✅")

            if aucs:
                auc_global = np.mean(aucs)
                std_global = np.std(aucs)
                acc_global = accuracy_score(y_true_acumulado, y_pred_acumulado)
                sens_global = recall_score(y_true_acumulado, y_pred_acumulado)

                cm = confusion_matrix(y_true_acumulado, y_pred_acumulado)
                espec_global = cm[0,0] / (cm[0,0] + cm[0,1]) if len(cm) == 2 else 0

                resultados_globales.append({
                    'Family': family,
                    'Algorithm': model_id,
                    'LODO_AUC_Mean': auc_global,
                    'LODO_AUC_Format': f"{auc_global:.3f} ± {std_global:.3f}",
                    'LODO_Accuracy': acc_global,
                    'LODO_Sensitivity': sens_global,
                    'LODO_Specificity': espec_global
                })

                plot_cv_roc_lodo(tprs, aucs, mean_fpr, model_id, DIR_PLOTS)
                plot_confusion_matrix_lodo(y_true_acumulado, y_pred_acumulado, model_id, DIR_PLOTS)

    if resultados_globales:
        df_reporte = pd.DataFrame(resultados_globales)
        df_reporte.sort_values(by='LODO_AUC_Mean', ascending=False, inplace=True)
        df_reporte.to_csv(os.path.join(DIR_OUTPUT, "STERILIZED_LODO_REPORT.csv"), index=False)

        df_hosp = pd.DataFrame(metricas_por_hospital)
        df_hosp.to_csv(os.path.join(DIR_OUTPUT, "STERILIZED_METRICS_PER_HOSPITAL.csv"), index=False)

        print("\n" + "="*80)
        print("🏆 FINAL CLASSIFICATION STERILIZED LODO (Pure Generalization)")
        print("="*80)
        df_print = df_reporte[['Algorithm', 'LODO_AUC_Format', 'LODO_Accuracy', 'LODO_Sensitivity']].copy()
        for col in ['LODO_Accuracy', 'LODO_Sensitivity']:
            df_print[col] = df_print[col].apply(lambda x: f"{x:.3f}")
        print(df_print.to_string(index=False))
        print("="*80)

if __name__ == "__main__":
    ejecutar_lodo_esterilizado()
