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
    Sustituye la regulación manual (Light) por una optimización bayesiana robusta.
    Permite capturar interacciones metabólicas complejas sin caer en overfitting.
    """
    model_name = "RF_Optimized"

    try:
        # 1. Codificación
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Objetivo de Optimización
        def objective(trial):
            # Espacio de búsqueda Senior:
            # Dejamos que el modelo crezca si los datos lo justifican
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': True,
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
                'random_state': random_state,
                'n_jobs': -1
            }

            rf = RandomForestClassifier(**params)

            # Validación Cruzada de Frontera (5-fold Stratified)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            # Optimizamos por AUC para asegurar separación de clases
            score = cross_val_score(rf, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()
            return score

        # 3. Ejecución de la búsqueda inteligente
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=40, timeout=600) # 10 min de exploración profunda

        # 4. Entrenamiento Final con los parámetros ganadores
        best_params = study.best_params
        final_rf = RandomForestClassifier(**best_params, random_state=random_state, n_jobs=-1)
        final_rf.fit(X_train, y_train_enc)

        # 5. Evaluación de resultados
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

        logger.info(f"RF Frontera -> Test AUC: {test_auc:.4f} | Depth: {best_params['max_depth']} | Est: {best_params['n_estimators']}")
        return results

    except Exception as e:
        logger.error(f"Error in RF Optimized: {e}")
        raise e
