# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    """
    Versión PROTOCOLO DE FRONTERA.
    Optimización Bayesiana avanzada para CatBoost.
    Aprovecha árboles simétricos y regularización L2 dinámica para datos metabólicos.
    """
    model_name = "CatBoost_Optimized"

    try:
        # 1. Codificación de etiquetas
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Objetivo de Optimización (Investigación de Vanguardia)
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 200, 1500),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-1, 10.0),
                'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
                'od_type': 'Iter',
                'od_wait': 50,
                'random_seed': random_state,
                'auto_class_weights': 'Balanced',
                'logging_level': 'Silent',
                'thread_count': -1
            }

            if params['bootstrap_type'] == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 10)
            elif params['bootstrap_type'] == 'Bernoulli':
                params['subsample'] = trial.suggest_float('subsample', 0.1, 1)

            # CV interna
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            aucs = []

            for train_idx, val_idx in cv.split(X_train, y_train_enc):
                X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_cv_train, y_cv_val = y_train_enc[train_idx], y_train_enc[val_idx]

                model = CatBoostClassifier(**params)
                model.fit(X_cv_train, y_cv_train, eval_set=(X_cv_val, y_cv_val))

                preds = model.predict_proba(X_cv_val)[:, 1]
                aucs.append(roc_auc_score(y_cv_val, preds))

            return np.mean(aucs)

        # 3. Búsqueda con Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, timeout=900) # 15 min para este peso pesado

        # 4. Entrenamiento Final con el "Overfitting Detector" activo
        best_params = study.best_params
        # Asegurar parámetros base en el modelo final
        best_params.update({
            'random_seed': random_state,
            'auto_class_weights': 'Balanced',
            'logging_level': 'Silent',
            'od_type': 'Iter',
            'od_wait': 50
        })

        final_cb = CatBoostClassifier(**best_params)
        final_cb.fit(X_train, y_train_enc)

        # 5. Evaluación
        y_test_prob = final_cb.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test_enc, y_test_prob)

        results = {
            'model_name': model_name,
            'trained_model': final_cb,
            'y_test_prob': y_test_prob,
            'y_test_true': y_test_enc,
            'roc_auc': test_auc,
            'best_params': best_params
        }

        logger.info(f"CatBoost Frontera -> Test AUC: {test_auc:.4f} | Best Iter: {best_params.get('iterations')}")
        return results

    except Exception as e:
        logger.error(f"Error in CatBoost Optimized: {e}")
        raise e
