# -*- coding: utf-8 -*-
import numpy as np
import optuna
from sklearn.svm import SVC
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
    Optimización Bayesiana de SVM (C y Gamma).
    Utiliza RobustScaler para proteger el hiperplano de metabolitos extremos.
    """
    model_name = "SVM_Frontier_v08"

    try:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        def objective(trial):
            # Exploramos tanto lineal como RBF por si la estructura es compleja
            kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
            c_param = trial.suggest_float('C', 1e-4, 10.0, log=True)

            # Gamma solo importa en kernels no lineales (RBF)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto']) if kernel == 'rbf' else 'scale'

            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('variance', VarianceThreshold()),
                ('scaler', RobustScaler()), # El SVM es extremadamente sensible a la escala
                ('classifier', SVC(
                    C=c_param,
                    kernel=kernel,
                    gamma=gamma,
                    probability=True, # Calibración nativa (Platt scaling)
                    class_weight='balanced',
                    random_state=random_state,
                    max_iter=5000
                ))
            ])

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            return cross_val_score(pipeline, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()

        # Búsqueda inteligente
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, timeout=600)

        # Entrenamiento Final
        best_params = study.best_params
        final_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('variance', VarianceThreshold()),
            ('scaler', RobustScaler()),
            ('classifier', SVC(
                **best_params,
                probability=True,
                class_weight='balanced',
                random_state=random_state,
                max_iter=10000
            ))
        ])

        final_pipeline.fit(X_train, y_train_enc)

        # Evaluación
        y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]
        y_test_pred = final_pipeline.predict(X_test)

        auc = roc_auc_score(y_test_enc, y_test_prob)
        acc = accuracy_score(y_test_enc, y_test_pred)
        cm = confusion_matrix(y_test_enc, y_test_pred)

        sens = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
        spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0

        logger.info(f"SVM Frontier -> AUC: {auc:.4f} | Kernel: {best_params['kernel']} | C: {best_params['C']:.4f}")

        return {
            'model_name': model_name,
            'accuracy': acc,
            'roc_auc': auc,
            'sensitivity': sens,
            'specificity': spec,
            'best_params': best_params,
            'trained_model': final_pipeline,
            'y_test_true': y_test_enc,
            'y_test_prob': y_test_prob
        }

    except Exception as e:
        logger.error(f"Error in SVM Frontier: {e}")
        raise e
