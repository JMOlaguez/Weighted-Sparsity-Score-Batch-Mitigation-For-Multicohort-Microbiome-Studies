# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    """
    Versión PROTOCOLO DE FRONTERA.
    Optimización Bayesiana para Random Forest.
    Introduce Cost-Complexity Pruning (ccp_alpha) para manejar ruido metabólico.
    """
    model_name = "Random_Forest_Optimized"

    try:
        # 1. Codificación
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Objetivo de Optimización (Nivel Senior)
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
                # Parámetro de frontera: poda por complejidad de costo
                'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-5, 1e-2, log=True),
                'class_weight': 'balanced',
                'random_state': random_state,
                'n_jobs': -1
            }

            rf = RandomForestClassifier(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            # Optimizamos por AUC para balancear sensibilidad/especificidad
            score = cross_val_score(rf, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()
            return score

        # 3. Búsqueda Bayesiana (Frontera)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=40, timeout=600)

        # 4. Entrenamiento Final
        best_params = study.best_params
        final_rf = RandomForestClassifier(**best_params, random_state=random_state, n_jobs=-1)
        final_rf.fit(X_train, y_train_enc)

        # 5. Evaluación de Desempeño
        train_probs = final_rf.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train_enc, train_probs)

        y_test_prob = final_rf.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test_enc, y_test_prob)

        results = {
            'model_name': model_name,
            'trained_model': final_rf,
            'y_test_prob': y_test_prob,
            'y_test_true': y_test_enc,
            'roc_auc': test_auc,
            'train_roc_auc': train_auc,
            'best_params': best_params
        }

        logger.info(f"RF Optimized -> Test AUC: {test_auc:.4f} | ccp_alpha: {best_params['ccp_alpha']:.5f}")
        return results

    except Exception as e:
        logger.error(f"Error in RF Optimized: {e}")
        raise e
