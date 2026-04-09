# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import optuna
from sklearn.naive_bayes import GaussianNB
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
    Eleva Naive Bayes mediante la optimización bayesiana de var_smoothing.
    Esto permite que el modelo sea robusto ante variables metabólicas con
    distribuciones no perfectas.
    """
    model_name = "Naive_Bayes_Optimized"

    try:
        # 1. Codificación
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Configuración de la Optimización (Protocolo de Frontera)
        def objective(trial):
            # var_smoothing es el parámetro clave: controla la estabilidad
            # Se busca en escala logarítmica desde 1e-11 hasta 1e-1
            var_smoothing = trial.suggest_float('var_smoothing', 1e-11, 1e-1, log=True)

            gnb = GaussianNB(var_smoothing=var_smoothing)

            # Validación cruzada para asegurar que el suavizado sea generalizable
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            score = cross_val_score(gnb, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()
            return score

        # 3. Búsqueda inteligente
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, timeout=120) # Muy rápido, NB es ligero

        # 4. Entrenamiento Final
        best_params = study.best_params
        final_gnb = GaussianNB(**best_params)
        final_gnb.fit(X_train, y_train_enc)

        # 5. Evaluación
        train_probs = final_gnb.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train_enc, train_probs)

        y_test_prob = final_gnb.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test_enc, y_test_prob)

        results = {
            'model_name': model_name,
            'trained_model': final_gnb,
            'y_test_prob': y_test_prob,
            'y_test_true': y_test_enc,
            'roc_auc': test_auc,
            'train_roc_auc': train_auc,
            'best_params': best_params
        }

        logger.info(f"NB Frontera -> Test AUC: {test_auc:.4f} | Smoothing: {best_params['var_smoothing']:.2e}")
        return results

    except Exception as e:
        logger.error(f"Error in Naive Bayes Optimized: {e}")
        raise e
