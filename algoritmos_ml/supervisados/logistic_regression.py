# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
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
    Optimización de ElasticNet para identificación de biomarcadores.
    Implementa RobustScaler y optimización continua de hiperparámetros.
    """
    model_name = "Logistic_ElasticNet_Frontier"

    try:
        # 1. Codificación
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Objetivo de Optimización Bayesiana
        def objective(trial):
            # C: Menor valor = más regularización (más variables eliminadas)
            c_param = trial.suggest_float('C', 1e-4, 10.0, log=True)
            # l1_ratio: 1 es Lasso puro, 0 es Ridge puro
            l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)

            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('variance', VarianceThreshold()),
                ('scaler', RobustScaler()),
                ('classifier', LogisticRegression(
                    penalty='elasticnet',
                    solver='saga',
                    C=c_param,
                    l1_ratio=l1_ratio,
                    class_weight='balanced',
                    max_iter=10000,
                    random_state=random_state,
                    n_jobs=-1
                ))
            ])

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            return cross_val_score(pipeline, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()

        # 3. Ejecución de la búsqueda
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=40, timeout=600)

        # 4. Re-entrenamiento con los mejores parámetros
        best_params = study.best_params
        final_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('variance', VarianceThreshold()),
            ('scaler', RobustScaler()),
            ('classifier', LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                **best_params,
                class_weight='balanced',
                max_iter=15000,
                random_state=random_state,
                n_jobs=-1
            ))
        ])

        final_pipeline.fit(X_train, y_train_enc)

        # 5. Extracción de Coeficientes (Biomarcadores)
        classifier = final_pipeline.named_steps['classifier']
        selector = final_pipeline.named_steps['variance']
        # Recuperamos nombres de columnas post-varianza
        valid_features = X_train.columns[selector.get_support()]

        feature_importance = pd.Series(
            classifier.coef_[0],
            index=valid_features
        ).sort_values(ascending=False)

        # 6. Evaluación Blind Test
        y_prob = final_pipeline.predict_proba(X_test)[:, 1]
        y_pred = final_pipeline.predict(X_test)

        auc = roc_auc_score(y_test_enc, y_prob)
        acc = accuracy_score(y_test_enc, y_pred)
        cm = confusion_matrix(y_test_enc, y_pred)

        sens = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
        spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0

        logger.info(f"LR Frontier -> AUC: {auc:.4f} | Variables activas: {sum(classifier.coef_[0] != 0)}")

        return {
            'model_name': model_name,
            'accuracy': acc,
            'roc_auc': auc,
            'sensitivity': sens,
            'specificity': spec,
            'feature_importance': feature_importance,
            'best_params': best_params,
            'trained_model': final_pipeline,
            'y_test_true': y_test_enc,
            'y_test_prob': y_prob
        }

    except Exception as e:
        logger.error(f"Error in Logistic Frontier: {e}")
        raise e
