# -*- coding: utf-8 -*-
import numpy as np
import optuna
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    """
    Versión PROTOCOLO DE FRONTERA.
    Voting Classifier con Pesos Optimizados por Optuna y Umbral Clínico de Youden.
    """
    model_name = "Voting_Youden_Frontera"

    try:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        # 1. Definición de Modelos Base de Alta Calidad
        base_models = [
            ('et', ExtraTreesClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=random_state)),
            ('svc', SVC(kernel='rbf', probability=True, C=1.0, class_weight='balanced', random_state=random_state)),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
            ('nb', GaussianNB())
        ]

        # 2. Optimización Bayesiana de PESOS
        def objective(trial):
            w_et = trial.suggest_float('w_et', 0.1, 1.0)
            w_svc = trial.suggest_float('w_svc', 0.1, 1.0)
            w_lda = trial.suggest_float('w_lda', 0.1, 1.0)
            w_nb = trial.suggest_float('w_nb', 0.0, 0.5) # Naive Bayes suele ser más débil

            voter = VotingClassifier(
                estimators=base_models,
                voting='soft',
                weights=[w_et, w_svc, w_lda, w_nb]
            )

            pipeline = Pipeline([('scaler', RobustScaler()), ('voter', voter)])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            return cross_val_score(pipeline, X_train, y_train_enc, cv=cv, scoring='roc_auc').mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)

        # 3. Construcción del Ensamble Ganador
        best_weights = [study.best_params['w_et'], study.best_params['w_svc'],
                        study.best_params['w_lda'], study.best_params['w_nb']]

        winning_voter = VotingClassifier(estimators=base_models, voting='soft', weights=best_weights)
        final_pipe = Pipeline([('scaler', RobustScaler()), ('voter', winning_voter)])

        # 4. Cálculo del Umbral de Youden mediante Cross-Validation
        # Esto evita que el umbral esté "viciado" por el entrenamiento puro
        cv_probs = cross_val_predict(final_pipe, X_train, y_train_enc, cv=5, method='predict_proba')[:, 1]
        fpr, tpr, thresholds = roc_curve(y_train_enc, cv_probs)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # 5. Ajuste Final y Evaluación
        final_pipe.fit(X_train, y_train_enc)
        y_test_prob = final_pipe.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_prob >= optimal_threshold).astype(int)

        # 6. Métricas
        acc = accuracy_score(y_test_enc, y_test_pred)
        auc = roc_auc_score(y_test_enc, y_test_prob)
        cm = confusion_matrix(y_test_enc, y_test_pred)

        sens = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
        spec = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0

        logger.info(f"Voting Frontera -> AUC: {auc:.4f} | Youden Threshold: {optimal_threshold:.3f}")

        return {
            'model_name': model_name,
            'accuracy': acc,
            'roc_auc': auc,
            'sensitivity': sens,
            'specificity': spec,
            'optimal_threshold': optimal_threshold,
            'trained_model': final_pipe,
            'y_test_true': y_test_enc,
            'y_test_prob': y_test_prob
        }

    except Exception as e:
        logger.error(f"Error in Voting Frontera: {e}")
        raise e
