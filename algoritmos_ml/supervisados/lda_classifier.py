# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import optuna
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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
    Implementa LDA con Regularización (Shrinkage) optimizada mediante TPE (Optuna).
    Ideal para manejar la colinealidad y mejorar la generalización.
    """
    model_name = "LDA_Optimized"

    try:
        # 1. Codificación
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Configuración de la Optimización Bayesiana
        def objective(trial):
            # En LDA, si usamos shrinkage, el solver debe ser 'lsqr' o 'eigen'
            solver = trial.suggest_categorical('solver', ['lsqr', 'eigen'])
            # El valor de shrinkage: 0 es LDA puro, 1 es regularización total
            shrinkage = trial.suggest_float('shrinkage', 0.0, 1.0)

            model = LinearDiscriminantAnalysis(
                solver=solver,
                shrinkage=shrinkage
            )

            # Validación cruzada interna para asegurar robustez
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            score = cross_val_score(model, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()
            return score

        # 3. Ejecución de la búsqueda (Frontera: 50 trials rápidos)
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, timeout=300) # Máximo 5 min

        # 4. Entrenamiento Final con los parámetros de élite
        best_params = study.best_params
        final_lda = LinearDiscriminantAnalysis(**best_params)
        final_lda.fit(X_train, y_train_enc)

        # 5. Evaluación de Desempeño
        train_probs = final_lda.predict_proba(X_train)[:, 1]
        train_auc = roc_auc_score(y_train_enc, train_probs)

        y_test_prob = final_lda.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test_enc, y_test_prob)

        results = {
            'model_name': model_name,
            'trained_model': final_lda,
            'y_test_prob': y_test_prob,
            'y_test_true': y_test_enc,
            'roc_auc': test_auc,
            'train_roc_auc': train_auc,
            'best_params': best_params
        }

        logger.info(f"LDA Frontera -> Test AUC: {test_auc:.4f} | Solver: {best_params['solver']} | Shrinkage: {best_params['shrinkage']:.4f}")
        return results

    except Exception as e:
        logger.error(f"Error in LDA Optimized: {e}")
        raise e
