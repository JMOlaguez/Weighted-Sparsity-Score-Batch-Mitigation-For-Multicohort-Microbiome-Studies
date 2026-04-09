# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import optuna
from sklearn.svm import SVC
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
    Optimización Bayesiana para SVC. Explora kernels y regularización continua.
    Maneja el desbalance mediante pesos de clase dinámicos.
    """
    model_name = "SVM_Optimized"

    try:
        # 1. Codificación
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Objetivo de Optimización (Nivel Investigador)
        def objective(trial):
            # Exploramos el kernel primero
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])

            # C es la penalización por error: búsqueda en escala logarítmica
            c_param = trial.suggest_float('C', 1e-3, 1e2, log=True)

            params = {
                'kernel': kernel,
                'C': c_param,
                'probability': True,
                'class_weight': 'balanced',
                'random_state': random_state
            }

            # Gamma y Degree solo aplican a ciertos kernels
            if kernel in ['rbf', 'poly', 'sigmoid']:
                params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
            if kernel == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)

            svc = SVC(**params)

            # CV Estratificada para robustez
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            score = cross_val_score(svc, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()
            return score

        # 3. Búsqueda Bayesiana
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=40, timeout=600)

        # 4. Entrenamiento Final
        best_params = study.best_params
        final_svc = SVC(**best_params, probability=True, class_weight='balanced', random_state=random_state)
        final_svc.fit(X_train, y_train_enc)

        # 5. Evaluación de Desempeño
        train_probs = final_svc.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train_enc, train_probs)

        y_test_prob = final_svc.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test_enc, y_test_prob)

        # Retorno compatible con el Orquestador 08 y tus métricas extra
        results = {
            'model_name': model_name,
            'trained_model': final_svc,
            'y_test_prob': y_test_prob,
            'y_test_true': y_test_enc,
            'roc_auc': test_auc,
            'train_roc_auc': train_auc,
            'best_params': best_params
        }

        logger.info(f"SVM Frontera -> Test AUC: {test_auc:.4f} | Kernel: {best_params['kernel']} | C: {best_params['C']:.4f}")
        return results

    except Exception as e:
        logger.error(f"Error in SVM Optimized: {e}")
        raise e
