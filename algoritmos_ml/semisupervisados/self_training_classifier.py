# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import optuna
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
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
    Versión PROTOCOLO DE FRONTERA - SSL Self-Training.
    Optimización Bayesiana de la estrategia de auto-etiquetado.
    Garantiza la trazabilidad de biomarcadores metabólicos.
    """
    model_name = "SelfTraining_RF_Frontier_v08"
    logger.info(f"Iniciando {model_name}...")

    try:
        # 1. Preparación de Etiquetas (Soportando el formato -1 para unlabeled)
        le = LabelEncoder()
        # Ajustamos con las etiquetas reales para conocer las clases
        y_train_full = pd.Series(y_train).reset_index(drop=True)
        y_test_enc = le.fit_transform(y_test)

        # 2. Objetivo de Optimización (Optuna)
        def objective(trial):
            # Hiperparámetros del Bosque Base (Exploración de profundidad y número de árboles)
            rf_params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                'class_weight': 'balanced',
                'random_state': random_state,
                'n_jobs': -1
            }

            # Parámetros críticos del Self-Training
            # 'threshold' es la probabilidad mínima para que el modelo acepte una predicción propia como verdad
            st_threshold = trial.suggest_float('st_threshold', 0.70, 0.95)
            st_criterion = trial.suggest_categorical('criterion', ['threshold', 'k_best'])

            base_rf = RandomForestClassifier(**rf_params)
            st_clf = SelfTrainingClassifier(
                base_estimator=base_rf,
                threshold=st_threshold,
                criterion=st_criterion,
                max_iter=10
            )

            # Pipeline Robusto
            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('selector', VarianceThreshold(threshold=1e-6)),
                ('scaler', RobustScaler()), # Protección contra outliers metabólicos
                ('classifier', st_clf)
            ])

            # Usamos Validación Cruzada sobre los datos etiquetados originales
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            # Evaluamos el desempeño en los datos conocidos
            return cross_val_score(pipeline, X_train, y_train_full, cv=cv, scoring='roc_auc').mean()

        # 3. Ejecución de la Búsqueda de Frontera
        study = optuna.create_study(direction='maximize')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            study.optimize(objective, n_trials=30, timeout=400)

        # 4. Re-entrenamiento con los Mejores Parámetros
        best_params = study.best_params

        # Separamos parámetros de RF y ST
        st_keys = ['st_threshold', 'criterion']
        rf_p = {k: v for k, v in best_params.items() if k not in st_keys}
        st_p = {k.replace('st_', ''): v for k, v in best_params.items() if k in st_keys}

        final_rf = RandomForestClassifier(**rf_p, class_weight='balanced', random_state=random_state)
        final_st = SelfTrainingClassifier(base_estimator=final_rf, **st_p)

        final_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('selector', VarianceThreshold(threshold=1e-6)),
            ('scaler', RobustScaler()),
            ('classifier', final_st)
        ])

        # Entrenamiento final
        final_pipe.fit(X_train, y_train_full)

        # 5. EXTRACCIÓN DE BIOMARCADORES (Feature Importance)
        # Accedemos al base_estimator_ que ya ha sido entrenado con las etiquetas propagadas
        trained_rf = final_pipe.named_steps['classifier'].base_estimator_
        selector = final_pipe.named_steps['selector']

        valid_features = X_train.columns[selector.get_support()]
        feature_importance = pd.Series(
            trained_rf.feature_importances_,
            index=valid_features
        ).sort_values(ascending=False)

        # 6. Evaluación Blind Test
        y_prob = final_pipe.predict_proba(X_test)[:, 1]
        y_pred = final_pipe.predict(X_test)

        acc = accuracy_score(y_test_enc, y_pred)
        auc = roc_auc_score(y_test_enc, y_prob)
        cm = confusion_matrix(y_test_enc, y_pred)

        sens = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
        spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0

        logger.info(f"✅ Self-Training Optimizado: AUC={auc:.4f} | Umbral={best_params.get('st_threshold', 'N/A')}")

        return {
            'model_name': model_name,
            'accuracy': acc,
            'roc_auc': auc,
            'sensitivity': sens,
            'specificity': spec,
            'feature_importance': feature_importance,
            'best_params': best_params,
            'trained_model': final_pipe,
            'y_test_true': y_test_enc,
            'y_test_prob': y_prob
        }

    except Exception as e:
        logger.error(f"Error crítico en módulo Self-Training: {e}")
        raise e
