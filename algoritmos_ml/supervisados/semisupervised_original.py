# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import optuna
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, f1_score

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    """
    Versión PROTOCOLO DE FRONTERA: Label Spreading (SSL).
    Optimiza la difusión de etiquetas mediante geometría de grafos.
    """
    model_name = "LabelSpreading_SSL_Frontier"

    try:
        # 1. Preparación de etiquetas
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Objetivo de Optimización
        def objective(trial):
            # Hiperparámetros de difusión
            kernel = trial.suggest_categorical('kernel', ['rbf', 'knn'])
            alpha = trial.suggest_float('alpha', 0.01, 0.9) # Factor de clampeo

            if kernel == 'rbf':
                gamma = trial.suggest_float('gamma', 1e-3, 20, log=True)
                clf = LabelSpreading(kernel='rbf', gamma=gamma, alpha=alpha, n_jobs=-1)
            else:
                n_neighbors = trial.suggest_int('n_neighbors', 3, 21)
                clf = LabelSpreading(kernel='knn', n_neighbors=n_neighbors, alpha=alpha, n_jobs=-1)

            # Ratio de enmascaramiento semi-supervisado
            pct = trial.suggest_categorical('ratio', [0.6, 0.8, 0.9])

            # Crear máscara SSL
            rng = np.random.RandomState(random_state)
            y_semi = y_train_enc.copy()
            n_unlabeled = int(len(y_semi) * (1 - pct))
            unlabeled_idx = rng.choice(len(y_semi), n_unlabeled, replace=False)
            y_semi[unlabeled_idx] = -1

            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler()),
                ('classifier', clf)
            ])

            # En SSL no solemos usar CV estándar porque el -1 rompe algunas métricas,
            # evaluamos sobre la capacidad de reconstrucción de las etiquetas originales
            pipeline.fit(X_train, y_semi)
            y_pred_internal = pipeline.predict(X_train)

            return f1_score(y_train_enc, y_pred_internal, average='weighted')

        # 3. Búsqueda de la estructura de grafo óptima
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, timeout=300)

        # 4. Re-entrenamiento Final
        best_p = study.best_params
        ratio = best_p.pop('ratio')

        # Generar máscara óptima
        rng = np.random.RandomState(random_state)
        y_semi_final = y_train_enc.copy()
        n_unlabeled = int(len(y_semi_final) * (1 - ratio))
        y_semi_final[rng.choice(len(y_semi_final), n_unlabeled, replace=False)] = -1

        final_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('classifier', LabelSpreading(**best_p, n_jobs=-1))
        ])

        final_pipe.fit(X_train, y_semi_final)

        # 5. Evaluación Blind Test
        y_prob = final_pipe.predict_proba(X_test)[:, 1]
        y_pred = final_pipe.predict(X_test)

        acc = accuracy_score(y_test_enc, y_pred)
        auc = roc_auc_score(y_test_enc, y_prob)
        cm = confusion_matrix(y_test_enc, y_pred)

        sens, spec = 0.0, 0.0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        logger.info(f"SSL Frontier -> AUC: {auc:.4f} | Ratio: {ratio} | Kernel: {best_p['kernel']}")

        return {
            'model_name': model_name,
            'accuracy': acc,
            'roc_auc': auc,
            'sensitivity': sens,
            'specificity': spec,
            'best_params': study.best_params,
            'trained_model': final_pipe,
            'y_test_true': y_test_enc,
            'y_test_prob': y_prob
        }

    except Exception as e:
        logger.error(f"Fatal error in SSL module: {e}")
        raise e
