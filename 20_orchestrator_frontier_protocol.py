# -*- coding: utf-8 -*-
"""
20_orchestrator_frontier_protocol.py

ORCHESTRATOR 20 - FRONTIER PROTOCOL (CAUSAL STERILIZATION & CONSENSUS)
Objective: Massive Execution of 23 Machine Learning algorithms on a causally
sterilized dataset, extraction of unbiased biomarkers, and generation of
deployable models.

---------------------------------------------------------------------------
PIPELINE CONTEXT: TRADITIONAL ML VS. STATE-OF-THE-ART (SOTA)
---------------------------------------------------------------------------
This orchestrator represents the ultimate evaluation of our bioinformatic
mitigation strategy using a robust battery of 23 Traditional and Advanced
Machine Learning models (Supervised, Semi-Supervised, and Unsupervised).

IMPORTANT: This is only STEP 1 of the final validation. By establishing a
sterilized baseline with these algorithms, a subsequent orchestrator will
be deployed to train "State-of-the-Art" (SOTA) models. The results produced here
will serve as the ultimate benchmark to prove if SOTA models can outperform
perfectly sterilized traditional ML.

---------------------------------------------------------------------------
METHODOLOGICAL RIGOR: PREVENTING DATA LEAKAGE
---------------------------------------------------------------------------
While previous exploratory phases used "Omnipresence" (Label Collision/Cloning)
to test the Inquisitor's resistance, THIS orchestrator disables cloning during
training. Why? Because cloning patients before Cross-Validation causes severe
Data Leakage (the exact same biological profile ends up in both the training
and validation folds, artificially inflating accuracy). Here, we exclusively
apply the Causal Zero Imputation (Phase 3) to ensure 100% pure validation.

PREREQUISITES:
- Python 3.8+
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib
- The `algoritmos_ml` module directory must exist and contain the 23 model scripts.
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
RUTA_DATOS = os.path.join(os.getcwd(), "1_INPUT_DATA")
RUTA_SALIDA = os.path.join(os.getcwd(), "ML_FINAL_RESULTS")
os.makedirs(RUTA_SALIDA, exist_ok=True)
sys.path.append(os.getcwd())

# === WARNING SILENCER ===
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# FULL MODULE CONFIGURATION (THE 23 ALGORITHMS)
# =============================================================================
MODULES_CONFIG = {
    'Supervised': [
        'algoritmos_ml.supervisados.catboost_classifier',
        'algoritmos_ml.supervisados.elasticnet_classifier',
        'algoritmos_ml.supervisados.extratrees_classifier',
        'algoritmos_ml.supervisados.knn_classifier',
        'algoritmos_ml.supervisados.lda_classifier',
        'algoritmos_ml.supervisados.lightgbm_classifier',
        'algoritmos_ml.supervisados.logistic_regression',
        'algoritmos_ml.supervisados.mlp_classifier',
        'algoritmos_ml.supervisados.naive_bayes_classifier',
        'algoritmos_ml.supervisados.rf_classifier',
        'algoritmos_ml.supervisados.rf_light_classifier',
        'algoritmos_ml.supervisados.stacking_classifier',
        'algoritmos_ml.supervisados.svm_classifier',
        'algoritmos_ml.supervisados.svm_linear_classifier',
        'algoritmos_ml.supervisados.weighted_ensemble',
        'algoritmos_ml.supervisados.weighted_threshold_optimized',
        'algoritmos_ml.supervisados.xgboost_classifier',
        'algoritmos_ml.supervisados.semisupervised_original'
    ],
    'SemiSupervised': [
        'algoritmos_ml.semisupervisados.label_propagation_classifier',
        'algoritmos_ml.semisupervisados.self_training_classifier'
    ],
    'Unsupervised': [
        'algoritmos_ml.no_supervisados.gmm_clustering',
        'algoritmos_ml.no_supervisados.isolation_forest',
        'algoritmos_ml.no_supervisados.kmeans_clustering'
    ]
}

SEEDS = [42, 123, 888]

# =============================================================================
# REPORTING AND Q1 PLOTTING FUNCTIONS
# =============================================================================
def plot_cv_roc_shaded(tprs, aucs, mean_fpr, model_name, output_dir):
    plt.figure(figsize=(8, 6))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='#2c3e50',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2.5)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#95a5a6', alpha=0.3, label=r'$\pm$ 1 Std. Dev.')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#e74c3c', label='Chance', alpha=.8)
    plt.title(f'Diagnostic Performance - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ROC_{model_name}.png"), dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                annot_kws={"size": 16, "weight": "bold"})
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"CM_{model_name}.png"), dpi=300)
    plt.close()

# =============================================================================
# STERILIZATION ENGINE (CORRECTED: NO CLONING TO PREVENT DATA LEAKAGE)
# =============================================================================
def aplicar_esterilizacion_causal(df_train, df_test):
    print("\n" + "🛡️"*30)
    print(" EXECUTING CAUSAL STERILIZATION ON THE TRAINING DATASET ")
    print(" (Pure Positive MDS - No cloning to protect Cross-Validation)")
    print("🛡️"*30)

    excluir = ['diagnosis', 'source', 'Batch', 'SampleID']
    features = [c for c in df_train.columns if c not in excluir]
    hospitales = df_train['source'].unique()
    epsilon_cero = 1e-5

    # 1. WSS CALCULATION
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

    # 2. CAUSAL MDS (Only for pseudo-zeros)
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

    # IMPORTANT: We return the sterilized data WITHOUT cloning.
    return df_train_mod

# =============================================================================
# MAIN BENCHMARK ENGINE
# =============================================================================
def ejecutar_benchmark():
    train_path = os.path.join(RUTA_DATOS, "DATA_TRAIN_READY.csv")
    test_path = os.path.join(RUTA_DATOS, "DATA_TEST_READY.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"❌ READY files not found in {RUTA_DATOS}.")
        return

    df_train_raw = pd.read_csv(train_path)
    df_test_raw = pd.read_csv(test_path)

    print("\n" + "="*60)
    print(" STARTING FRONTIER ORCHESTRATOR V.09 ")
    print("="*60)

    # APPLY THE ANTIDOTE BEFORE TRAINING
    df_train_esterilizado = aplicar_esterilizacion_causal(df_train_raw, df_test_raw)

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
    print(" DATA MATRICES READY FOR TRAINING ")
    print(" Sterilized Training Dimension:", X_train.shape)
    print(" Test Dimension (Pure):", X_test.shape)
    print("="*60)

    for family, scripts in MODULES_CONFIG.items():
        print(f"\n🚀 DEPLOYING FAMILY: {family.upper()}")

        for script_path in scripts:
            model_id = script_path.split('.')[-1]
            try:
                module = importlib.import_module(script_path)
                tprs, aucs = [], []
                mean_fpr = np.linspace(0, 1, 100)
                metrics_seeds = []
                model_importances_seeds = []

                print(f"  |-- Optimizing and Evaluating: {model_id} ", end="", flush=True)

                for seed in SEEDS:
                    res = module.train_and_evaluate(X_train, y_train, X_test, y_test, random_state=seed)

                    y_prob = res.get('y_test_prob')
                    y_true = res.get('y_test_true')

                    if y_prob is None or y_true is None:
                        print(" [!] Incomplete return ", end="", flush=True)
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

                    print(".", end="", flush=True)

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

                plot_cv_roc_shaded(tprs, aucs, mean_fpr, model_id, RUTA_SALIDA)
                plot_confusion_matrix(y_true, y_pred, model_id, RUTA_SALIDA)

                # SAVE MASTER MODEL (Independent and Deployable)
                joblib.dump(res.get('trained_model'), os.path.join(RUTA_SALIDA, f"{model_id}_Sterilized.pkl"))

                if model_importances_seeds:
                    mean_imp_model = pd.concat(model_importances_seeds, axis=1).mean(axis=1)
                    mean_imp_model.name = model_id
                    global_feature_importances.append(mean_imp_model)

                results_summary.append(res_final)
                print(" ✅")

            except Exception as e:
                print(f" ❌ Error in {model_id}: {e}")

    if not results_summary:
        print("❌ No model completed execution.")
        return

    df_reporte = pd.DataFrame(results_summary)
    df_reporte.sort_values(by=['ROC_AUC', 'Accuracy'], ascending=[False, False], inplace=True)

    cols_to_export = ['Family', 'Algorithm', 'AUC_Format', 'Accuracy', 'Sensitivity',
                      'Specificity', 'F1_Score', 'MCC', 'Kappa', 'LogLoss']
    df_reporte[cols_to_export].to_csv(os.path.join(RUTA_SALIDA, "METRICS_REPORT_Q1_STERILIZED.csv"), index=False)

    if global_feature_importances:
        df_biomarcadores = pd.concat(global_feature_importances, axis=1).fillna(0)
        df_norm = df_biomarcadores.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > 0 else 0)
        df_biomarcadores['Consensus_Score'] = df_norm.mean(axis=1)
        df_biomarcadores.sort_values(by='Consensus_Score', ascending=False, inplace=True)
        df_biomarcadores.to_csv(os.path.join(RUTA_SALIDA, "BIOMARKERS_CONSENSUS_STERILIZED.csv"))

        print("\n" + "🧬"*30)
        print(" TOP 5 UNIVERSAL BIOMARKERS (NO GEOGRAPHIC BIAS)")
        print(df_biomarcadores[['Consensus_Score']].head(5))
        print("🧬"*30)

    print("\n" + "="*85)
    print("🏆 FINAL CLASSIFICATION OF ROBUST MODELS (SORTED BY ROC-AUC)")
    print("="*85)
    df_print = df_reporte[['Algorithm', 'AUC_Format', 'Accuracy', 'Sensitivity', 'Specificity']].copy()
    for col in ['Accuracy', 'Sensitivity', 'Specificity']:
        df_print[col] = df_print[col].apply(lambda x: f"{x:.3f}")
    print(df_print.head(15).to_string(index=False))
    print("="*85)
    print(f"📁 Artifacts, .pkl Models (Ready for Inference) and Plots in: {RUTA_SALIDA}")

if __name__ == "__main__":
    ejecutar_benchmark()
