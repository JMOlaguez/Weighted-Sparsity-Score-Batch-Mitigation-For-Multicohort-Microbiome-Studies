# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import optuna
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler, LabelEncoder
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

class SupervisedGMM(BaseEstimator, ClassifierMixin):
    """
    Wrapper Evolucionado para Orquestador 08.
    Mapea componentes de GMM a etiquetas clínicas con manejo de probabilidades.
    """
    def __init__(self, n_components=2, covariance_type='full', max_iter=500,
                 n_init=10, init_params='kmeans', random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state

    def fit(self, X, y):
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            random_state=self.random_state
        )
        self.gmm_.fit(X)

        clusters = self.gmm_.predict(X)
        self.mapping_ = {}
        for i in range(self.n_components):
            idx = np.where(clusters == i)[0]
            if len(idx) > 0:
                counts = np.bincount(y[idx])
                self.mapping_[i] = np.argmax(counts)
            else:
                self.mapping_[i] = 0
        return self

    def predict(self, X):
        clusters = self.gmm_.predict(X)
        return np.array([self.mapping_[c] for c in clusters])

    def predict_proba(self, X):
        gmm_probs = self.gmm_.predict_proba(X)
        probs = np.zeros((X.shape[0], 2))
        for i in range(self.n_components):
            label = self.mapping_[i]
            if label < 2:
                probs[:, label] += gmm_probs[:, i]
        # Normalización para asegurar que sumen 1
        return probs / probs.sum(axis=1, keepdims=True)

def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    """
    Versión PROTOCOLO DE FRONTERA.
    GMM con optimización Bayesiana de la geometría de covarianza.
    """
    model_name = "GMM_Frontier_v08"

    try:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        def objective(trial):
            n_comp = trial.suggest_int('n_components', 2, 4)
            cov_type = trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical'])

            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('selector', VarianceThreshold(threshold=1e-6)),
                ('scaler', RobustScaler()),
                ('classifier', SupervisedGMM(
                    n_components=n_comp,
                    covariance_type=cov_type,
                    random_state=random_state
                ))
            ])

            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            return cross_val_score(pipeline, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()

        # Optimización Bayesiana
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=25, timeout=300)

        # Re-entrenamiento Final
        best_p = study.best_params
        final_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('selector', VarianceThreshold(threshold=1e-6)),
            ('scaler', RobustScaler()),
            ('classifier', SupervisedGMM(**best_p, random_state=random_state))
        ])

        final_pipe.fit(X_train, y_train_enc)

        # --- EXTRACCIÓN DE BIOMARCADORES PARA GMM ---
        # En GMM, la importancia se mide por la separación de las medias de los clusters
        gmm_model = final_pipe.named_steps['classifier'].gmm_
        selector = final_pipe.named_steps['selector']
        feature_names = X_train.columns[selector.get_support()]

        # Calculamos la diferencia absoluta entre las medias de los componentes
        # Si hay más de 2 componentes, usamos la desviación estándar de las medias
        if best_p['n_components'] == 2:
            importance_values = np.abs(gmm_model.means_[0] - gmm_model.means_[1])
        else:
            importance_values = np.std(gmm_model.means_, axis=0)

        feature_importance = pd.Series(
            importance_values,
            index=feature_names
        ).sort_values(ascending=False)

        # Evaluación
        y_prob = final_pipe.predict_proba(X_test)[:, 1]
        y_pred = final_pipe.predict(X_test)

        auc = roc_auc_score(y_test_enc, y_prob)
        acc = accuracy_score(y_test_enc, y_pred)
        cm = confusion_matrix(y_test_enc, y_pred)

        sens = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
        spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0

        logger.info(f"✅ GMM Frontier -> AUC: {auc:.4f} | Covarianza: {best_p['covariance_type']}")

        return {
            'model_name': model_name,
            'accuracy': acc,
            'roc_auc': auc,
            'sensitivity': sens,
            'specificity': spec,
            'feature_importance': feature_importance,
            'best_params': best_p,
            'trained_model': final_pipe,
            'y_test_true': y_test_enc,
            'y_test_prob': y_prob
        }

    except Exception as e:
        logger.error(f"Error en GMM Frontier: {e}")
        raise e
