# algoritmos_ml/supervisados/lightgbm_classifier.py
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

def train_and_evaluate(X_train, y_train, X_test, y_test, random_state=42):
    def objective(trial):
        # Espacio de búsqueda de "Frontera"
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'random_state': random_state,
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        }

        # Validación cruzada rápida interna para guiar a Optuna
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)

    # Optimización inteligente: 50 pruebas (mucho más rápido y mejor que GridSearch)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, timeout=600) # Máximo 10 min por modelo

    # Entrenar modelo final con los mejores parámetros encontrados
    best_model = lgb.LGBMClassifier(**study.best_params, random_state=random_state)
    best_model.fit(X_train, y_train)

    y_prob = best_model.predict_proba(X_test)[:, 1]

    return {
        'trained_model': best_model,
        'y_test_prob': y_prob,
        'y_test_true': y_test,
        'roc_auc': roc_auc_score(y_test, y_prob),
        'best_params': study.best_params
    }
