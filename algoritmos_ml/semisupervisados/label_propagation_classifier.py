# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import warnings

from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):

    logger.info("Initializing Enhanced Semi-Supervised Label Spreading...")

    try:

        # --------------------------------------------------
        # 1. Encoding de etiquetas
        # --------------------------------------------------

        le = LabelEncoder()

        le.fit(pd.concat([pd.Series(y_train), pd.Series(y_test)]).unique())

        y_train_series = pd.Series(y_train).reset_index(drop=True)
        y_test_enc = le.transform(y_test)

        # --------------------------------------------------
        # 2. Enmascaramiento Estratificado SSL
        # --------------------------------------------------

        y_train_semi = y_train_series.copy()
        mask_fraction = 0.30

        rng = np.random.RandomState(random_state)

        for cls in le.classes_:

            idx = y_train_semi[y_train_semi == cls].index

            if len(idx) < 3:
                continue

            mask_size = max(1, int(len(idx) * mask_fraction))

            mask_idx = rng.choice(idx, size=mask_size, replace=False)

            y_train_semi.loc[mask_idx] = -1

        y_train_enc = np.array([
            -1 if v == -1 else le.transform([v])[0]
            for v in y_train_semi
        ])

        # --------------------------------------------------
        # 3. Pipeline robusto
        # --------------------------------------------------

        pipeline = Pipeline([

            ('imputer', SimpleImputer(strategy='median')),

            ('selector', VarianceThreshold(threshold=1e-6)),

            ('scaler', StandardScaler()),

            ('classifier', LabelSpreading(
                max_iter=3000,
                tol=1e-3
            ))
        ])

        # --------------------------------------------------
        # 4. Espacio de búsqueda estabilizado
        # --------------------------------------------------

        param_grid = [

            {
                'classifier__kernel': ['knn'],

                'classifier__n_neighbors': [
                    5, 9, 15, 25
                ],

                'classifier__alpha': [
                    0.01, 0.2, 0.8
                ]
            },

            {
                'classifier__kernel': ['rbf'],

                'classifier__gamma': [
                    0.01, 0.05, 0.1, 1.0
                ],

                'classifier__alpha': [
                    0.01, 0.2, 0.8
                ]
            }
        ]

        # --------------------------------------------------
        # 5. CV estratificada
        # --------------------------------------------------

        cv_strat = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state
        )

        cv_splits = list(cv_strat.split(X_train, y_train))

        # --------------------------------------------------
        # 6. Halving Search robusto
        # --------------------------------------------------

        with warnings.catch_warnings():

            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide"
            )

            grid_search = HalvingGridSearchCV(

                estimator=pipeline,

                param_grid=param_grid,

                cv=cv_splits,

                scoring='roc_auc',

                factor=3,

                n_jobs=-1,

                random_state=random_state,

                error_score=np.nan
            )

            grid_search.fit(X_train, y_train_enc)

        best_model = grid_search.best_estimator_

        clean_params = {
            k.replace('classifier__', ''): v
            for k, v in grid_search.best_params_.items()
        }

        # --------------------------------------------------
        # 7. Evaluación en Blind Test
        # --------------------------------------------------

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test_enc, y_pred)

        roc = roc_auc_score(y_test_enc, y_prob)

        cm = confusion_matrix(y_test_enc, y_pred)

        if cm.shape == (2, 2):

            tn, fp, fn, tp = cm.ravel()

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        else:

            sens = 0.0
            spec = 0.0

        return {

            'model_name': 'Label_Spreading_Halving',

            'accuracy': acc,

            'roc_auc': roc,

            'sensitivity': sens,

            'specificity': spec,

            'trained_model': best_model,

            'best_params': clean_params,

            'y_test_true': y_test_enc,

            'y_test_prob': y_prob
        }

    except Exception as e:

        logger.error(f"Error in Semi-Supervised module: {e}")
        raise e
