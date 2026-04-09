# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import optuna
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    """
    Versión PROTOCOLO DE FRONTERA.
    KNN con Optimización Bayesiana de métricas de distancia y pesos.
    Usa RobustScaler para minimizar el impacto de outliers en el cálculo de distancias.
    """
    model_name = "KNN_Optimized"

    try:
        # 1. Codificación
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Objetivo de Optimización
        def objective(trial):
            # Exploramos un rango más amplio y continuo
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 31, step=2),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'metric': 'minkowski',
                'p': trial.suggest_float('p', 1.0, 3.0), # De Manhattan a Minkowski superior
                'leaf_size': trial.suggest_int('leaf_size', 10, 50),
                'n_jobs': -1
            }

            pipeline = Pipeline([
                ('scaler', RobustScaler()), # Fundamental para KNN en bio-datos
                ('knn', KNeighborsClassifier(**params))
            ])

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            score = cross_val_score(pipeline, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()
            return score

        # 3. Búsqueda con Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=40, timeout=300)

        # 4. Entrenamiento Final
        best_params = study.best_params
        final_pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('knn', KNeighborsClassifier(**best_params, n_jobs=-1))
        ])

        final_pipeline.fit(X_train, y_train_enc)

        # 5. Evaluación
        y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test_enc, y_test_prob)

        results = {
            'model_name': model_name,
            'trained_model': final_pipeline,
            'y_test_prob': y_test_prob,
            'y_test_true': y_test_enc,
            'roc_auc': test_auc,
            'best_params': best_params
        }

        logger.info(f"KNN Frontera -> Test AUC: {test_auc:.4f} | K: {best_params['n_neighbors']} | p: {best_params['p']:.2f}")
        return results

    except Exception as e:
        logger.error(f"Error in KNN Optimized: {e}")
        raise e
