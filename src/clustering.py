"""
Módulo para experimentos de clustering (K-Means) no dataset Olist.

Inclui:
- Função para Elbow Method
- Treino de KMeans
- Atribuição de clusters ao dataframe
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def elbow_method(
    X: pd.DataFrame,
    max_k: int = 10,
    random_state: int = 42,
) -> Dict[str, list]:
    """
    Executa o Elbow Method para auxiliar na escolha do número de clusters.

    Parameters
    ----------
    X : pd.DataFrame
    max_k : int
    random_state : int

    Returns
    -------
    dict
        {"ks": [...], "inertias": [...]}
    """
    # Padronizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ks = list(range(1, max_k + 1))
    inertias = []

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    plt.plot(ks, inertias, marker="o")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inertia (Soma dos Erros Quadráticos)")
    plt.title("Elbow Method")
    plt.grid(True)
    plt.show()

    return {"ks": ks, "inertias": inertias}


def fit_kmeans(
    X: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> KMeans:
    """
    Treina um modelo KMeans em dados padronizados.

    Parameters
    ----------
    X : pd.DataFrame
    n_clusters : int
    random_state : int

    Returns
    -------
    KMeans
    """
    # Garantir que não há valores não finitos
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    X = X[np.isfinite(X).all(axis=1)]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.astype(float))
    # Substituir eventuais NaN/inf pós-escala por 0 para evitar overflow
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(X_scaled).all():
        X_scaled[~np.isfinite(X_scaled)] = 0.0
    # Clipping extra para evitar overflow em distância
    X_scaled = np.clip(X_scaled, -1e6, 1e6)

    # Usar init="random" e algorithm="lloyd" para evitar RuntimeWarnings (overflow/div by zero)
    # observados com k-means++ em algumas arquiteturas/versões do sklearn.
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=random_state, 
        n_init=10, 
        init="random", 
        algorithm="lloyd"
    )
    kmeans.fit(X_scaled)

    # Guardar scaler dentro de atributo customizado (para usar depois)
    kmeans.scaler_ = scaler
    return kmeans


def assign_clusters(
    model: KMeans,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Atribui clusters a um conjunto de dados usando um modelo KMeans
    treinado com StandardScaler armazenado em model.scaler_.

    Parameters
    ----------
    model : KMeans
    X : pd.DataFrame

    Returns
    -------
    np.ndarray
        Vetor de labels de cluster.
    """
    scaler = getattr(model, "scaler_", None)
    if scaler is None:
        raise ValueError("O modelo KMeans não possui atributo 'scaler_'. Use fit_kmeans.")

    X_scaled = scaler.transform(X)
    labels = model.predict(X_scaled)
    return labels
