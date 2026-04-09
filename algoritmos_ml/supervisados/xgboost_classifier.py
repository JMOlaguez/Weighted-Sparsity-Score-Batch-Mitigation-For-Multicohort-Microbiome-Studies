# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Habilitar búsqueda rápida
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    """
    Versión Adaptada  Entrena XGBoost con Halving Search para optimizar hiperparámetros.
    """
    logger.info("Initializing XGBoost Classifier with Halving 5-Fold Stratified CV...")

    try:
        import xgboost as xgb
    except ImportError:
        logger.error("The 'xgboost' library is not installed.")
        raise ImportError("Missing xgboost library.")

    try:
        # 1. Codificación de Etiquetas
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Espacio de búsqueda (Grid)
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'gamma': [0, 1],
            'min_child_weight': [1, 5]
        }

        # 3. Configurar balanceo de clases automático
        num_neg = np.sum(y_train_enc == 0)
        num_pos = np.sum(y_train_enc == 1)
        scale_weight = num_neg / num_pos if num_pos > 0 else 1.0

        # 4. Configurar el modelo base
        # Se elimina use_label_encoder para evitar DeprecationWarnings en versiones nuevas
        xgb_base = xgb.XGBClassifier(
            scale_pos_weight=scale_weight,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1
        )

        # 5. Estrategia de Validación Cruzada interna
        cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        # 6. Ejecutar el HalvingGridSearchCV
        # Usamos n_estimators como recurso para que el halving sea eficiente
        logger.info(f"Running Halving CV on {len(X_train)} samples...")
        grid_search = HalvingGridSearchCV(
            estimator=xgb_base,
            param_grid=param_grid,
            cv=cv_strat,
            scoring='roc_auc',
            factor=3,
            resource='n_estimators',
            min_resources=50,
            max_resources=500,
            n_jobs=-1,
            verbose=0,
            random_state=random_state
        )

        grid_search.fit(X_train, y_train_enc)
        best_model = grid_search.best_estimator_

        # 7. Evaluación final en Blind Test
        y_pred_enc = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # 8. Calcular Métricas
        acc = accuracy_score(y_test_enc, y_pred_enc)
        roc = roc_auc_score(y_test_enc, y_prob)
        cm = confusion_matrix(y_test_enc, y_pred_enc)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            sensitivity, specificity = 0.0, 0.0

        # 9. Retorno estandarizado
        return {
            'model_name': 'XGBoost',
            'best_params': grid_search.best_params_,
            'accuracy': acc,
            'roc_auc': roc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'trained_model': best_model,
            'y_test_true': y_test_enc,
            'y_test_prob': y_prob
        }

    except Exception as e:
        logger.error(f"Fatal error in XGBoost module: {e}")
        raise e
