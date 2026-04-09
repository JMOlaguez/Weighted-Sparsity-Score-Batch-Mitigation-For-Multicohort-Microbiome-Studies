# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import optuna
from sklearn.neural_network import MLPClassifier
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
    MLP con optimización bayesiana de topología y regularización.
    Usa RobustScaler para manejar outliers metabólicos.
    """
    model_name = "MLP_NeuralNet_Optimized"

    try:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        def objective(trial):
            # Definición dinámica de la arquitectura
            n_layers = trial.suggest_int('n_layers', 1, 3)
            layers = []
            for i in range(n_layers):
                layers.append(trial.suggest_int(f'n_units_l{i}', 16, 256, step=16))

            params = {
                'hidden_layer_sizes': tuple(layers),
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
                'solver': 'adam',
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True),
                'max_iter': 1000,
                'early_stopping': True,
                'n_iter_no_change': 20,
                'random_state': random_state
            }

            # Pipeline de frontera
            pipeline = Pipeline([
                ('scaler', RobustScaler()), # Más estable que StandardScaler para bio-datos
                ('mlp', MLPClassifier(**params))
            ])

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            return cross_val_score(pipeline, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()

        # Búsqueda Bayesiana
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, timeout=900)

        # Entrenamiento Final
        best_params = study.best_params
        # Reconstruir hidden_layer_sizes de los resultados de trial
        layers = [best_params[f'n_units_l{i}'] for i in range(best_params['n_layers'])]

        final_params = {
            'hidden_layer_sizes': tuple(layers),
            'activation': best_params['activation'],
            'alpha': best_params['alpha'],
            'learning_rate_init': best_params['learning_rate_init'],
            'random_state': random_state,
            'max_iter': 1500
        }

        final_pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('mlp', MLPClassifier(**final_params))
        ])

        final_pipeline.fit(X_train, y_train_enc)

        y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test_enc, y_test_prob)

        results = {
            'model_name': model_name,
            'trained_model': final_pipeline,
            'y_test_prob': y_test_prob,
            'y_test_true': y_test_enc,
            'roc_auc': test_auc,
            'best_params': final_params
        }

        logger.info(f"MLP Frontera -> Test AUC: {test_auc:.4f} | Arch: {final_params['hidden_layer_sizes']}")
        return results

    except Exception as e:
        logger.error(f"Error in MLP Optimized: {e}")
        raise e
