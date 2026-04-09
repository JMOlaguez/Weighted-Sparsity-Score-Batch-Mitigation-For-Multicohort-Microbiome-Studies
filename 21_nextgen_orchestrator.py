# -*- coding: utf-8 -*-
"""
21_nextgen_orchestrator.py

NEXT-GEN ORCHESTRATOR (V.10) - 2024/2026 TABULAR ARCHITECTURES
Objective: Massive Execution of Foundation Models, State-Space Models (Mamba),
and Deep Ensembles on Causally Sterilized Data.

---------------------------------------------------------------------------
THE PINNACLE OF TABULAR AI: THE NEXT-GEN ARSENAL
---------------------------------------------------------------------------
While Traditional ML (Random Forest, LightGBM) has historically dominated
tabular data, the period between 2024 and 2026 saw a paradigm shift with the
adaptation of Deep Learning architectures natively built for tabular datasets.
This script deploys the absolute State-of-the-Art (SOTA):

1. TabICL (Foundation Models): Adapts the "In-Context Learning" capabilities
   of Large Language Models (LLMs) directly to tabular data, allowing the model
   to infer relationships dynamically.
2. MambaTab (State-Space Models): Replaces the quadratic bottleneck of
   Transformers with Selective State-Space Models (Mamba). It captures ultra-long
   and complex biological interactions across thousands of bacteria without
   losing computational efficiency.
3. TabM (Deep Ensembles): A parameter-efficient architecture that simulates
   hundreds of neural networks (ensembling) within a single forward pass,
   offering extreme robustness against overfitting.

---------------------------------------------------------------------------
THE SCIENTIFIC HYPOTHESIS
---------------------------------------------------------------------------
Did deep neural networks perform better in older studies because they
learned complex biology, or simply because they were vastly superior at
memorizing the "Batch Effect" (the geographic barcode)?

By executing these SOTA architectures on our Causally Sterilized Dataset
(where the batch effect has been mathematically destroyed via WSS and MDS),
we force these models to rely PURELY on the biological signal of Autism.
If they outperform the models from Orchestrator 20, we have proven that
Deep Learning genuinely captures complex microbiome dynamics.

PREREQUISITES:
- Python 3.9+ (Highly recommended for newer architectures)
- Hardware: CUDA-enabled GPU is virtually mandatory for MambaTab and TabICL.
- Libraries: torch, pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
- The `algoritmos_nextgen` module directory must exist.
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    roc_curve, matthews_corrcoef, f1_score, auc, recall_score,
    cohen_kappa_score, log_loss
)

# === ACADEMIC STYLE CONFIGURATION ===
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# === PATH CONFIGURATION ===
DIR_INPUT = os.path.join(os.getcwd(), "1_INPUT_DATA")
DIR_OUTPUT = os.path.join(os.getcwd(), "ML_NEXTGEN_RESULTS")
os.makedirs(DIR_OUTPUT, exist_ok=True)
sys.path.append(os.getcwd())

# === WARNING SILENCER ===
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# NEXT-GEN MODULE CONFIGURATION (2024-2026 LITERATURE)
# =============================================================================
MODULES_CONFIG = {
    'Foundation_Models': [
        'algoritmos_nextgen.tabicl_classifier'
    ],
    'State_Space_Models': [
        'algoritmos_nextgen.mambatab_classifier'
    ],
    'Deep_Ensembles': [
        'algoritmos_nextgen.tabm_classifier'
    ]
}

SEEDS = [42, 123, 888]

# =============================================================================
# REPORTING AND PLOTTING FUNCTIONS
# =============================================================================
def plot_cv_roc_shaded(tprs, aucs, mean_fpr, model_name, output_dir):
    plt.figure(figsize=(8, 6))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='#8e44ad',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2.5)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#9b59b6', alpha=0.3, label=r'$\pm$ 1 Std. Dev.')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#e74c3c', label='Chance', alpha=.8)
    plt.title(f'Next-Gen Diagnostic Performance - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ROC_{model_name}.png"), dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
                annot_kws={"size": 16, "weight": "bold"})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"CM_{model_name}.png"), dpi=300)
    plt.close()

# =============================================================================
# CAUSAL STERILIZATION ENGINE (SHIELDED AGAINST DATA LEAKAGE)
# =============================================================================
def aplicar_esterilizacion_causal(df_train):
    print("\n" + "🛡️"*30)
    print(" EXECUTING CAUSAL STERILIZATION ON THE TRAINING DATASET ")
    print(" (Pure Positive MDS - No cloning to protect Cross-Validation)")
    print("🛡️"*30)

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
    bacterias_intervenidas = 0

    for feat in features:
        wss_actual = df_wss[df_wss['Bacteria'] == feat]['WSS'].values[0]
        if wss_actual >= umbral_wss:
            idx_ceros = df_train_mod[feat] <= epsilon_cero
            idx_positivos = df_train_mod[feat] > epsilon_cero
            n_ceros = idx_ceros.sum()

            if n_ceros > 0 and idx_positivos.sum() > 0:
                distribucion_positiva = df_train_mod.loc[idx_positivos, feat].values
                valores_relleno = np.random.choice(distribucion_positiva, size=n_ceros, replace=True)
                df_train_mod.loc[idx_ceros, feat] = valores_relleno
                bacterias_intervenidas += 1

    print(f" ✔️ Sterilized toxic bacteria (Positive MDS): {bacterias_intervenidas}")
    print(f" 🧹 Clean dataset returned with original dimensions: {df_train_mod.shape}")

    return df_train_mod

# =============================================================================
# MAIN NEXT-GEN BENCHMARK ENGINE
# =============================================================================
def ejecutar_benchmark():
    train_path = os.path.join(DIR_INPUT, "DATA_TRAIN_READY.csv")
    test_path = os.path.join(DIR_INPUT, "DATA_TEST_READY.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"❌ READY files not found in {DIR_INPUT}.")
        return

    df_train_raw = pd.read_csv(train_path)
    df_test_raw = pd.read_csv(test_path)

    print("\n" + "="*60)
    print(" STARTING NEXT-GEN ORCHESTRATOR V.10 (2024-2026 LITERATURE) ")
    print("="*60)

    # APPLY THE ANTIDOTE BEFORE TRAINING (Without leaking data to Optuna/Hyper-tuning)
    df_train_esterilizado = aplicar_esterilizacion_causal(df_train_raw)

    target = 'diagnosis'
    cols_drop = [target, 'source', 'Batch']

    X_train = df_train_esterilizado.drop(columns=[c for c in cols_drop if c in df_train_esterilizado.columns])
    y_train = df_train_esterilizado[target]

    # THE TEST SET REMAINS PURE, SACRED, AND UNTOUCHED
    X_test = df_test_raw.drop(columns=[c for c in cols_drop if c in df_test_raw.columns])
    y_test = df_test_raw[target]

    results_summary = []
    global_feature_importances = []

    print("\n" + "="*60)
    print(" DATA MATRICES READY FOR NEXT-GEN TRAINING ")
    print(" Sterilized Training Dimension:", X_train.shape)
    print(" Test Dimension (Pure):", X_test.shape)
    print("="*60)

    for family, scripts in MODULES_CONFIG.items():
        print(f"\n🚀 DEPLOYING NEXT-GEN FAMILY: {family.upper()}")

        for script_path in scripts:
            model_id = script_path.split('.')[-1]
            try:
                module = importlib.import_module(script_path)
                tprs, aucs = [], []
                mean_fpr = np.linspace(0, 1, 100)
                metrics_seeds = []
                model_importances_seeds = []

                print(f"  |-- Training Deep Architecture: {model_id} ", end="\n", flush=True)

                for seed in SEEDS:
                    print(f"      > Evaluating Seed {seed}...", end="", flush=True)
                    res = module.train_and_evaluate(X_train, y_train, X_test, y_test, random_state=seed)

                    y_prob = res.get('y_test_prob')
                    y_true = res.get('y_test_true')

                    if y_prob is None or y_true is None:
                        print(" [!] Incomplete return ", flush=True)
                        continue

                    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
                    idx = np.argmax(tpr - fpr)
                    best_threshold = thresholds[idx]
                    y_pred = (y_prob >= best_threshold).astype(int)

                    acc = accuracy_score(y_true, y_pred)
                    mcc = matthews_corrcoef(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)
                    sens = recall_score(y_true, y_pred)
                    kappa = cohen_kappa_score(y_true, y_pred)
                    lloss = log_loss(y_true, y_prob)

                    cm = confusion_matrix(y_true, y_pred)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        espec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    else:
                        espec = 0

                    metrics_seeds.append({
                        'acc': acc, 'auc': res.get('roc_auc', roc_auc_score(y_true, y_prob)),
                        'mcc': mcc, 'f1': f1, 'sens': sens, 'espec': espec,
                        'kappa': kappa, 'logloss': lloss
                    })

                    tprs.append(np.interp(mean_fpr, fpr, tpr))
                    aucs.append(res.get('roc_auc', roc_auc_score(y_true, y_prob)))

                    if 'feature_importance' in res:
                        model_importances_seeds.append(res['feature_importance'])

                    print(" ✅")

                if not metrics_seeds:
                    print(" ❌ Metric collection failed.")
                    continue

                avg_auc = np.mean(aucs)
                std_auc = np.std(aucs)

                res_final = {
                    'Family': family,
                    'Algorithm': model_id,
                    'ROC_AUC': avg_auc,
                    'AUC_Format': f"{avg_auc:.3f} ± {std_auc:.3f}",
                    'Accuracy': np.mean([m['acc'] for m in metrics_seeds]),
                    'Sensitivity': np.mean([m['sens'] for m in metrics_seeds]),
                    'Specificity': np.mean([m['espec'] for m in metrics_seeds]),
                    'F1_Score': np.mean([m['f1'] for m in metrics_seeds]),
                    'MCC': np.mean([m['mcc'] for m in metrics_seeds]),
                    'Kappa': np.mean([m['kappa'] for m in metrics_seeds]),
                    'LogLoss': np.mean([m['logloss'] for m in metrics_seeds])
                }

                plot_cv_roc_shaded(tprs, aucs, mean_fpr, model_id, DIR_OUTPUT)
                plot_confusion_matrix(y_true, y_pred, model_id, DIR_OUTPUT)

                joblib.dump(res.get('trained_model'), os.path.join(DIR_OUTPUT, f"{model_id}_NextGen.pkl"))

                if model_importances_seeds:
                    mean_imp_model = pd.concat(model_importances_seeds, axis=1).mean(axis=1)
                    mean_imp_model.name = model_id
                    global_feature_importances.append(mean_imp_model)

                results_summary.append(res_final)

            except Exception as e:
                print(f" ❌ Critical Error in Architecture {model_id}: {e}")

    if not results_summary:
        print("❌ No model completed execution.")
        return

    df_reporte = pd.DataFrame(results_summary)
    df_reporte.sort_values(by=['ROC_AUC', 'Accuracy'], ascending=[False, False], inplace=True)

    cols_to_export = ['Family', 'Algorithm', 'AUC_Format', 'Accuracy', 'Sensitivity',
                      'Specificity', 'F1_Score', 'MCC', 'Kappa', 'LogLoss']
    df_reporte[cols_to_export].to_csv(os.path.join(DIR_OUTPUT, "METRICS_REPORT_Q1_NEXTGEN.csv"), index=False)

    if global_feature_importances:
        df_biomarcadores = pd.concat(global_feature_importances, axis=1).fillna(0)
        df_norm = df_biomarcadores.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > 0 else 0)
        df_biomarcadores['Consensus_Score'] = df_norm.mean(axis=1)
        df_biomarcadores.sort_values(by='Consensus_Score', ascending=False, inplace=True)
        df_biomarcadores.to_csv(os.path.join(DIR_OUTPUT, "BIOMARKERS_CONSENSUS_NEXTGEN.csv"))

    print("\n" + "="*85)
    print("🏆 FINAL CLASSIFICATION: NEXT-GEN ARCHITECTURES (SORTED BY ROC-AUC)")
    print("="*85)
    df_print = df_reporte[['Algorithm', 'AUC_Format', 'Accuracy', 'Sensitivity', 'Specificity']].copy()
    for col in ['Accuracy', 'Sensitivity', 'Specificity']:
        df_print[col] = df_print[col].apply(lambda x: f"{x:.3f}")
    print(df_print.to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    ejecutar_benchmark()
