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
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
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
