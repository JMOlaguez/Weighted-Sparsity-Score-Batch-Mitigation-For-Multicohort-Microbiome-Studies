# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
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
    Versión PROTOCOLO DE FRONTERA.
    Ensamble Ponderado Calibrado.
    Busca la máxima estabilidad eliminando el ruido mediante ExtraTrees y RBF.
    """
    model_name = "Weighted_Ensemble_Stability_v08"

    try:
        # 1. Codificación
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 2. Modelos Base (Configuración de Frontera para Estabilidad)
        base_models = [
            # ExtraTrees es más estable que RF ante ruido de metabolitos
            ('et', ExtraTreesClassifier(
                n_estimators=250,
                max_depth=8,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            )),
            # RBF para capturar relaciones no lineales complejas
            ('svc', SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                class_weight='balanced',
                random_state=random_state
            )),
            # NB es un excelente regularizador por su simplicidad
            ('nb', GaussianNB()),
            # LDA proyecta los datos para maximizar separación de clases
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ]

        # 3. Pesos Estratégicos (Prioridad ET y SVC)
        # ET (0.4) + SVC (0.3) + LDA (0.2) + NB (0.1)
        pesos = [0.4, 0.3, 0.2, 0.1]

        voter = VotingClassifier(
            estimators=base_models,
            voting='soft',
            weights=pesos,
            n_jobs=-1
        )

        # 4. Pipeline Robusto (Uso de RobustScaler es vital)
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('ensemble', voter)
        ])

        # 5. Entrenamiento
        logger.info(f"Training Stability Ensemble with weights: {pesos}")
        pipeline.fit(X_train, y_train_enc)

        # 6. Evaluación
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test_enc, y_pred)
        auc = roc_auc_score(y_test_enc, y_prob)
        cm = confusion_matrix(y_test_enc, y_pred)

        sens, spec = 0.0, 0.0
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            'model_name': model_name,
            'accuracy': acc,
            'roc_auc': auc,
            'sensitivity': sens,
            'specificity': spec,
            'trained_model': pipeline,
            'y_test_true': y_test_enc,
            'y_test_prob': y_prob,
            'best_params': {'weights': pesos, 'architecture': 'Stability_Frontier'}
        }

    except Exception as e:
        logger.error(f"Error in Stability Ensemble: {e}")
        raise e
