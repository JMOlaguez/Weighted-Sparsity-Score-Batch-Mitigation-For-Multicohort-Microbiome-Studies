# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
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
    ExtraTrees con Optimización Bayesiana (Optuna) y análisis de biomarcadores.
    """
    model_name = "ExtraTrees_Frontier_v08"

    try:
        # 1. Codificación
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Objetivo de Optimización Bayesiana
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_float('max_features', 0.1, 0.9),
                'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-5, 1e-2, log=True),
                'bootstrap': trial.suggest_bool('bootstrap'),
                'class_weight': 'balanced',
                'random_state': random_state,
                'n_jobs': -1
            }

            # Pipeline interno para asegurar que el escalado/selección sea parte del CV
            pipeline = Pipeline([
                ('selector', VarianceThreshold(threshold=1e-6)),
                ('classifier', ExtraTreesClassifier(**params))
            ])

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            return cross_val_score(pipeline, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()

        # 3. Búsqueda de Hiperparámetros
        logger.info("Starting Bayesian Optimization for ExtraTrees...")
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=40, timeout=600)

        # 4. Entrenamiento Final con los Mejores Parámetros
        best_params = study.best_params
        final_pipe = Pipeline([
            ('selector', VarianceThreshold(threshold=1e-6)),
            ('classifier', ExtraTreesClassifier(
                **best_params,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            ))
        ])

        final_pipe.fit(X_train, y_train_enc)

        # 5. Importancia de Variables (Biomarcadores)
        clf = final_pipe.named_steps['classifier']
        sel = final_pipe.named_steps['selector']
        # Recuperar nombres de columnas tras el VarianceThreshold
        feature_names = X_train.columns[sel.get_support()]
        feature_importances = pd.Series(
            clf.feature_importances_,
            index=feature_names
        ).sort_values(ascending=False)

        # 6. Evaluación en Blind Test
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

        logger.info(f"ET Frontier Final -> AUC: {auc:.4f} | Accuracy: {acc:.4f}")

        return {
            'model_name': model_name,
            'best_params': best_params,
            'accuracy': acc,
            'roc_auc': auc,
            'sensitivity': sens,
            'specificity': spec,
            'feature_importance': feature_importances,
            'trained_model': final_pipe,
            'y_test_true': y_test_enc,
            'y_test_prob': y_prob
        }

    except Exception as e:
        logger.error(f"Error in ExtraTrees Frontier: {e}")
        raise e
