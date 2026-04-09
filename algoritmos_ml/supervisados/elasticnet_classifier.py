# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
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
    Optimización Bayesiana de Elastic Net (L1 + L2).
    Busca el punto de equilibrio exacto para la selección de biomarcadores.
    """
    model_name = "ElasticNet_LogReg_Optimized"

    try:
        # 1. Codificación
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Objetivo de Optimización (Frontera)
        def objective(trial):
            # C: Penalización inversa (menor C -> mayor regularización)
            c_param = trial.suggest_float('C', 1e-4, 10.0, log=True)
            # l1_ratio: 0 es Ridge (L2), 1 es Lasso (L1)
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

            pipeline = Pipeline([
                ('scaler', RobustScaler()), # Protección contra outliers metabólicos
                ('classifier', LogisticRegression(
                    penalty='elasticnet',
                    solver='saga',
                    C=c_param,
                    l1_ratio=l1_ratio,
                    class_weight='balanced',
                    max_iter=5000,
                    random_state=random_state,
                    n_jobs=-1
                ))
            ])

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            # Optimizamos por AUC para asegurar la capacidad discriminatoria
            score = cross_val_score(pipeline, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()
            return score

        # 3. Búsqueda Inteligente
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, timeout=300)

        # 4. Entrenamiento Final
        best_params = study.best_params
        final_pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('classifier', LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                **best_params,
                class_weight='balanced',
                max_iter=10000,
                random_state=random_state,
                n_jobs=-1
            ))
        ])

        final_pipeline.fit(X_train, y_train_enc)

        # 5. Evaluación de Desempeño
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

        logger.info(f"ElasticNet Frontera -> Test AUC: {test_auc:.4f} | C: {best_params['C']:.4f} | L1 Ratio: {best_params['l1_ratio']:.4f}")
        return results

    except Exception as e:
        logger.error(f"Error in Elastic Net Optimized: {e}")
        raise e
