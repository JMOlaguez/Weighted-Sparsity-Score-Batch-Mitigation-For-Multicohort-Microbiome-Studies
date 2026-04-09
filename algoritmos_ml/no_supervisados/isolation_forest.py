# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class SupervisedIsolationForest(BaseEstimator, ClassifierMixin):
    """
    Wrapper Evolucionado para Orquestador 08.
    Convierte la lógica de aislamiento en clasificación probabilística.
    """
    def __init__(self, n_estimators=100, contamination='auto', max_samples='auto',
                 max_features=1.0, bootstrap=False, random_state=None):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

    def fit(self, X, y):
        self.iforest_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.iforest_.fit(X)

        # Mapeo: ¿Es la anomalía (-1) el grupo ASD o el Control?
        preds = self.iforest_.predict(X)
        self.mapping_ = {}
        for val in [1, -1]: # 1: normal, -1: anomalía
            mask = (preds == val)
            if mask.any():
                self.mapping_[val] = np.argmax(np.bincount(y[mask]))
            else:
                self.mapping_[val] = 1 if val == -1 else 0

        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        raw_preds = self.iforest_.predict(X)
        return np.array([self.mapping_.get(p, 0) for p in raw_preds])

    def predict_proba(self, X):
        # El score de decisión indica qué tan "aislada" está la muestra
        scores = self.iforest_.decision_function(X)
        # Transformación Sigmoide para probabilidad
        probs_anomaly = 1 / (1 + np.exp(scores))

        label_anomaly = self.mapping_.get(-1, 1)
        probs = np.zeros((X.shape[0], 2))
        probs[:, label_anomaly] = probs_anomaly
        probs[:, 1 - label_anomaly] = 1 - probs_anomaly
        return probs

def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    """
    Versión PROTOCOLO DE FRONTERA.
    Isolation Forest con Optimización Bayesiana y scoring de biomarcadores por aislamiento.
    """
    model_name = "IsolationForest_Frontier_v08"

    try:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_samples': trial.suggest_float('max_samples', 0.5, 1.0),
                'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                'contamination': trial.suggest_float('contamination', 0.01, 0.5),
                'bootstrap': trial.suggest_bool('bootstrap')
            }

            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('selector', VarianceThreshold(threshold=1e-6)),
                ('scaler', RobustScaler()),
                ('classifier', SupervisedIsolationForest(**params, random_state=random_state))
            ])

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            return cross_val_score(pipeline, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()

        # 1. Optimización Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, timeout=300)

        # 2. Re-entrenamiento con mejores parámetros
        best_p = study.best_params
        final_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('selector', VarianceThreshold(threshold=1e-6)),
            ('scaler', RobustScaler()),
            ('classifier', SupervisedIsolationForest(**best_p, random_state=random_state))
        ])

        final_pipe.fit(X_train, y_train_enc)

        # 3. EXTRACCIÓN DE BIOMARCADORES (Feature Importance)
        # Isolation Forest no tiene feature_importances_ nativo fácil de extraer como RF,
        # pero podemos calcular el impacto de cada variable midiendo la profundidad media.
        # Aquí usamos un proxy: la relevancia en las particiones del bosque.
        if_model = final_pipe.named_steps['classifier'].iforest_
        selector = final_pipe.named_steps['selector']
        feature_names = X_train.columns[selector.get_support()]

        # Calculamos importancia basada en la frecuencia de uso de cada feature en los árboles
        # Este es un estándar de la industria para interpretar IF
        feature_importances = np.mean([
            tree.tree_.compute_feature_importances(normalize=False)
            for tree in if_model.estimators_
        ], axis=0)

        importance_ser = pd.Series(
            feature_importances,
            index=feature_names
        ).sort_values(ascending=False)

        # 4. Evaluación
        y_prob = final_pipe.predict_proba(X_test)[:, 1]
        y_pred = final_pipe.predict(X_test)

        auc = roc_auc_score(y_test_enc, y_prob)
        acc = accuracy_score(y_test_enc, y_pred)
        cm = confusion_matrix(y_test_enc, y_pred)

        sens = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
        spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0

        logger.info(f"✅ IF Frontier -> AUC: {auc:.4f} | Contaminación: {best_p['contamination']:.2f}")

        return {
            'model_name': model_name,
            'accuracy': acc,
            'roc_auc': auc,
            'sensitivity': sens,
            'specificity': spec,
            'feature_importance': importance_ser,
            'best_params': best_p,
            'trained_model': final_pipe,
            'y_test_true': y_test_enc,
            'y_test_prob': y_prob
        }

    except Exception as e:
        logger.error(f"Error en Isolation Forest Frontier: {e}")
        raise e
