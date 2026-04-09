# -*- coding: utf-8 -*-
"""
23_dirty_lodo_baseline.py

ORCHESTRATOR 12 - DIRTY LODO (LITERATURE BASELINE)
Objective: Spatial evaluation using RAW CLR transformed data. Exposes the model collapse
due to "Shortcut Learning". Generates logs, plots, and a comparative CSV.

---------------------------------------------------------------------------
SCIENTIFIC JUSTIFICATION: THE ILLUSION OF HIGH ACCURACY
---------------------------------------------------------------------------
Machine Learning models are fundamentally "lazy" optimizers; they look for
the easiest path to minimize error. This is known as "Shortcut Learning".

Instead of learning the incredibly complex biological signature of Autism,
the model takes a shortcut: it memorizes the geographic barcode. It essentially
learns: "If the sample looks like it was sequenced in a given hospital, apply
local prediction capabilities."

HOW BATCH EFFECTS ARTIFICIALLY INFLATE PREDICTION:
During standard Random Cross-Validation, patients from Hospital A are present
in both the training and testing sets. The model successfully uses the
memorized barcode to cheat on the test set, reporting artificially inflated
metrics (e.g., AUC > 0.92).

PREREQUISITES:
- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
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
    accuracy_score, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, auc, recall_score,
    cohen_kappa_score, log_loss, matthews_corrcoef
)

# === ACADEMIC STYLE ===
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# === PATH CONFIGURATION ===
DIR_INPUT = os.path.join(os.getcwd(), "1_INPUT_DATA")
DIR_OUTPUT = os.path.join(os.getcwd(), "ML_RESULTS_DIRTY_LODO")

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

log_file = os.path.join(DIR_OUTPUT, f"DIRTY_LODO_EXECUTION_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
sys.stdout = Logger(log_file)

# === ROBUST ALGORITHMS (To ensure a clean execution) ===
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

    plt.plot(mean_fpr, mean_tpr, color='#c0392b', label=r'Mean Dirty LODO ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2.5)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#e74c3c', alpha=0.3)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#7f8c8d', label='Chance')
    plt.title(f'Global LODO Performance (RAW Data) - {model_name}', fontweight='bold')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ROC_LODO_DIRTY_{model_name}.png"), dpi=300)
    plt.close()

def plot_confusion_matrix_lodo(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False, annot_kws={"size": 16, "weight": "bold"})
    plt.title(f'LODO Confusion Matrix (RAW Data) - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"CM_LODO_DIRTY_{model_name}.png"), dpi=300)
    plt.close()

# === MAIN DIRTY LODO ENGINE ===
def ejecutar_benchmark_lodo_sucio():
    train_path = os.path.join(DIR_INPUT, "DATA_TRAIN_READY.csv")
    test_path = os.path.join(DIR_INPUT, "DATA_TEST_READY.csv")

    if not os.path.exists(train_path):
        print("❌ Files not found.")
        return

    # Dynamic fusion
    df_train_raw = pd.read_csv(train_path)
    df_test_raw = pd.read_csv(test_path)
    df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

    hospitales = df_all['source'].unique()
    target = 'diagnosis'
    cols_drop = [target, 'source', 'Batch']

    print("\n" + "="*75)
    print(" STARTING LODO ORCHESTRATOR V.12 (DIRTY / RAW BASELINE) ")
    print(f" Total Patients: {len(df_all)} | Total Hospitals (Folds): {len(hospitales)}")
    print(f" Logs backed up at: {log_file}")
    print("="*75)

    resultados_globales = []
    metricas_por_hospital = []

    for family, scripts in MODULES_CONFIG.items():
        print(f"\n🚀 DEPLOYING FAMILY: {family.upper()}")

        for script_path in scripts:
            model_id = script_path.split('.')[-1]
            try:
                module = importlib.import_module(script_path)
                tprs, aucs = [], []
                mean_fpr = np.linspace(0, 1, 100)
                y_true_acumulado, y_pred_acumulado = [], []

                print(f"  |-- Evaluating Dirty LODO: {model_id} ", end="\n", flush=True)

                for hosp_test in hospitales:
                    print(f"      > Isolating: {hosp_test} ... ", end="", flush=True)

                    df_train_fold = df_all[df_all['source'] != hosp_test].copy()
                    df_test_fold = df_all[df_all['source'] == hosp_test].copy()

                    if len(df_test_fold[target].unique()) < 2:
                        print("[Skipped: Only 1 class present] ")
                        continue

                    # NO STERILIZATION. PASSING RAW DATA DIRECTLY.
                    X_train = df_train_fold.drop(columns=[c for c in cols_drop if c in df_train_fold.columns]).values
                    y_train = df_train_fold[target].values
                    X_test = df_test_fold.drop(columns=[c for c in cols_drop if c in df_test_fold.columns]).values
                    y_test = df_test_fold[target].values

                    y_prob_fold = np.zeros(len(y_test))

                    for seed in SEEDS:
                        res = module.train_and_evaluate(X_train, y_train, X_test, y_test, random_state=seed)
                        if res.get('y_test_prob') is not None:
                            y_prob_fold += res['y_test_prob'] / len(SEEDS)

                        if seed == SEEDS[-1] and 'trained_model' in res:
                            joblib.dump(res['trained_model'], os.path.join(DIR_MODELS, f"{model_id}_{hosp_test}.pkl"))

                    # Fold calculation
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

                    # Save for comparison
                    metricas_por_hospital.append({
                        'Algorithm': model_id,
                        'Hospital': hosp_test,
                        'Dirty_AUC': auc_fold,
                        'Dirty_ACC': acc_fold
                    })

                    print(f"AUC: {auc_fold:.3f} ✅")

                # Cumulative Metrics
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
                    'LODO_Specificity': espec_global,
                })

                plot_cv_roc_lodo(tprs, aucs, mean_fpr, model_id, DIR_PLOTS)
                plot_confusion_matrix_lodo(y_true_acumulado, y_pred_acumulado, model_id, DIR_PLOTS)

            except Exception as e:
                print(f" ❌ Error in {model_id}: {e}")

    # Save Reports
    df_reporte = pd.DataFrame(resultados_globales)
    df_reporte.sort_values(by='LODO_AUC_Mean', ascending=False, inplace=True)
    df_reporte.to_csv(os.path.join(DIR_OUTPUT, "DIRTY_LODO_CLINICAL_REPORT.csv"), index=False)

    df_hosp = pd.DataFrame(metricas_por_hospital)
    df_hosp.to_csv(os.path.join(DIR_OUTPUT, "DIRTY_METRICS_PER_HOSPITAL.csv"), index=False)

    print("\n" + "="*85)
    print("🏆 GLOBAL DIRTY LODO RESULTS (Shortcut Learning Effect)")
    print("="*85)
    df_print = df_reporte[['Algorithm', 'LODO_AUC_Format', 'LODO_Accuracy', 'LODO_Sensitivity']].copy()
    for col in ['LODO_Accuracy', 'LODO_Sensitivity']:
        df_print[col] = df_print[col].apply(lambda x: f"{x:.3f}")
    print(df_print.to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    ejecutar_benchmark_lodo_sucio()
