# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import optuna
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

def train_and_evaluate(X_train, y_train=None, X_test=None, y_test=None, random_state=42):
    """
    Versión PROTOCOLO DE FRONTERA - K-Means NO SUPERVISADO.
    Enfoque: Descubrimiento de estructuras latentes y biomarcadores de clusterización.
    """
    model_name = "KMeans_Unsupervised_Frontier"
    logger.info("Ejecutando K-Means No Supervisado para análisis de estructura latente...")

    try:
        # 1. Preparación de datos (Ignoramos etiquetas para el entrenamiento)
        # Combinamos para un análisis de clusterización global si es necesario,
        # o mantenemos X_train para consistencia con el orquestador.
        X_combined = pd.concat([X_train, X_test]) if X_test is not None else X_train

        # 2. Optimización del Número de Clusters (K) y Calidad
        def objective(trial):
            k = trial.suggest_int('n_clusters', 2, 6)
            init_strategy = trial.suggest_categorical('init', ['k-means++', 'random'])

            pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('selector', VarianceThreshold(threshold=1e-6)),
                ('scaler', RobustScaler()),
                ('kmeans', KMeans(n_clusters=k, init=init_strategy, n_init=10, random_state=random_state))
            ])

            # Procesamos datos
            X_proc = pipeline.named_steps['imputer'].fit_transform(X_combined)
            X_proc = pipeline.named_steps['selector'].fit_transform(X_proc)
            X_proc = pipeline.named_steps['scaler'].fit_transform(X_proc)

            # Ajustamos KMeans
            km = pipeline.named_steps['kmeans'].fit(X_proc)
            labels = km.labels_

            # El objetivo es maximizar la cohesión y separación (Silhouette Score)
            score = silhouette_score(X_proc, labels)
            return score

        # 3. Búsqueda de la estructura óptima
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, timeout=300)

        # 4. Ajuste Final con el K óptimo
        best_k = study.best_params['n_clusters']
        final_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('selector', VarianceThreshold(threshold=1e-6)),
            ('scaler', RobustScaler()),
            ('kmeans', KMeans(**study.best_params, n_init=10, random_state=random_state))
        ])

        # Ajuste
        final_pipe.fit(X_combined)
        km_final = final_pipe.named_steps['kmeans']
        selector = final_pipe.named_steps['selector']

        # 5. EXTRACCIÓN DE BIOMARCADORES DE CLUSTERIZACIÓN
        # Identificamos qué metabolitos tienen mayor varianza entre los centroides de los clusters encontrados
        feature_names = X_combined.columns[selector.get_support()]
        # Varianza entre centroides: a mayor varianza, más contribuye el metabolito a definir los grupos
        cluster_importance = np.var(km_final.cluster_centers_, axis=0)

        feature_importance = pd.Series(
            cluster_importance,
            index=feature_names
        ).sort_values(ascending=False)

        # 6. Métricas de Calidad del Clustering
        X_final_proc = final_pipe.named_steps['scaler'].transform(
            final_pipe.named_steps['selector'].transform(
                final_pipe.named_steps['imputer'].transform(X_combined)
            )
        )

        sil_score = silhouette_score(X_final_proc, km_final.labels_)
        ch_score = calinski_harabasz_score(X_final_proc, km_final.labels_)

        logger.info(f"✅ Clustering Finalizado. Clusters óptimos: {best_k} | Silhouette: {sil_score:.4f}")

        return {
            'model_name': model_name,
            'n_clusters': best_k,
            'silhouette_score': sil_score,
            'calinski_harabasz': ch_score,
            'feature_importance': feature_importance,
            'cluster_labels': km_final.labels_,
            'trained_model': final_pipe,
            'best_params': study.best_params
        }

    except Exception as e:
        logger.error(f"Error en K-Means No Supervisado: {e}")
        raise e
