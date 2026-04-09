# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    """
    Versión PROTOCOLO DE FRONTERA: STACKING DE ALTA PRECISIÓN.
    Combina la potencia de ExtraTrees con modelos lineales robustos.
    El meta-modelo usa CV interna para ajustar su propia regularización.
    """
    model_name = "Stacking_Frontera_v08"

    try:
        # 1. Codificación
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Definir Modelos Base de "Frontera"
        # Cada uno ataca el problema desde un ángulo distinto
        base_models = [
            # El especialista en varianza (vanguardia metabólica)
            ('et', ExtraTreesClassifier(
                n_estimators=300,
                max_depth=10,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            )),
            # El especialista en fronteras no lineales
            ('svc', SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                class_weight='balanced',
                random_state=random_state
            )),
            # El especialista en separación lineal pura
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ]

        # 3. Meta-modelo Dinámico (LogisticRegressionCV)
        # En lugar de un C fijo, busca el mejor valor de regularización mediante CV
        meta_clf = LogisticRegressionCV(
            Cs=10,
            penalty='l2',
            cv=5,
            scoring='roc_auc',
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000,
            n_jobs=-1
        )

        # 4. Configurar Stacking
        stacker = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_clf,
            cv=5,
            stack_method='predict_proba',
            passthrough=True, # IMPORTANTE: permite que el meta-modelo vea también los datos originales
            n_jobs=-1
        )

        # 5. Pipeline de Frontera
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('stacking', stacker)
        ])

        # 6. Entrenamiento
        logger.info("Training Frontier Stacking (ET + SVC + LDA)...")
        pipeline.fit(X_train, y_train_enc)

        # 7. Evaluación
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test_enc, y_pred)
        auc = roc_auc_score(y_test_enc, y_prob)

        # Métricas de confusión
        cm = confusion_matrix(y_test_enc, y_pred)
        sens, spec = 0.0, 0.0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        logger.info(f"Stacking Frontera Final -> Accuracy: {acc:.4f} | AUC: {auc:.4f}")

        return {
            'model_name': model_name,
            'accuracy': acc,
            'roc_auc': auc,
            'sensitivity': sens,
            'specificity': spec,
            'trained_model': pipeline,
            'y_test_true': y_test_enc,
            'y_test_prob': y_prob,
            'best_params': 'Stacking_Optimized_Seniors'
        }

    except Exception as e:
        logger.error(f"Error in Stacking Frontera: {e}")
        raise e
