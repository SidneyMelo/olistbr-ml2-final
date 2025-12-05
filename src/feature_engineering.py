"""
Módulo de feature engineering para o dataset Olist.

Responsável por:
- Construir targets (binário, multiclasses).
- Selecionar e transformar features numéricas/categóricas.
- Preparar matrizes para modelos supervisionados e clustering.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas de target para diferentes tarefas:

    - review_binary: 1 = review >= 4, 0 = review <= 2, NaN = 3 (neutro)
    - review_positive: 1 = review >= 4, 0 caso contrário
    - review_negative: 1 = review <= 2, 0 caso contrário (opcional)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    df["review_positive"] = (df["review_score"] >= 4).astype(int)
    df["review_negative"] = (df["review_score"] <= 2).astype(int)

    def _binary_strict(score: float) -> float:
        if pd.isna(score):
            return np.nan
        if score >= 4:
            return 1
        if score <= 2:
            return 0
        return np.nan

    df["review_binary"] = df["review_score"].apply(_binary_strict)

    return df


def get_supervised_feature_lists() -> Tuple[List[str], List[str]]:
    """
    Retorna as listas de nomes de colunas numéricas e categóricas
    que serão usadas como entrada nos modelos supervisionados.

    Returns
    -------
    (numeric_features, categorical_features)
    """
    numeric_features = [
        "total_items_price",
        "total_freight_value",
        "payment_installments",
        "payment_value",
        "n_items",
        "delivery_time_days",
        "estimated_delivery_days",
        "delivery_delay_days",
        "review_count",
    ]

    categorical_features = [
        "payment_type",
        "customer_state",
        "product_category_name_english",
    ]

    return numeric_features, categorical_features


class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Clipa valores extremos por feature usando quantis, ajudando na estabilidade numérica.
    Definida no escopo do módulo para permitir pickle dos pipelines.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)


def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Constrói um ColumnTransformer padrão para:
    - Escalar variáveis numéricas
    - One-hot encoding em categóricas

    Parameters
    ----------
    numeric_features : list[str]
    categorical_features : list[str]

    Returns
    -------
    ColumnTransformer
    """

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", QuantileClipper(lower=0.01, upper=0.99)),
            ("scaler", MinMaxScaler()),
        ]
    )
    def _onehot_dense() -> OneHotEncoder:
        # Compatibilidade entre versoes do scikit-learn (sparse_output novo, sparse antigo)
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)

    # Dense output para modelos que nao aceitam matriz esparsa (ex: GaussianNB)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _onehot_dense()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def prepare_supervised_xy(
    df: pd.DataFrame,
    target_col: str = "review_binary",
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Filtra o dataframe para linhas válidas do target, e retorna X, y,
    além das listas de colunas numéricas e categóricas.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
        Nome da coluna de target (default = 'review_binary').

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    numeric_features : list[str]
    categorical_features : list[str]
    """
    df = df.copy()
    df = df[~df[target_col].isna()].copy()

    numeric_features, categorical_features = get_supervised_feature_lists()

    # Garantir que colunas existem (caso exclua algo antes)
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    X = df[numeric_features + categorical_features]
    y = df[target_col].astype(int)

    return X, y, numeric_features, categorical_features


def prepare_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara subconjunto de features para clustering de clientes/pedidos.

    Exemplo de variáveis:
    - total_items_price
    - total_freight_value
    - payment_value
    - n_items
    - delivery_time_days
    - delivery_delay_days

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    cols = [
        "total_items_price",
        "total_freight_value",
        "payment_value",
        "n_items",
        "delivery_time_days",
        "delivery_delay_days",
    ]
    cols = [c for c in cols if c in df.columns]

    cluster_df = df[cols].copy()
    # Garantir tipo numérico
    cluster_df = cluster_df.apply(pd.to_numeric, errors="coerce")
    # Tratar inf/-inf
    cluster_df = cluster_df.replace([np.inf, -np.inf], np.nan)
    # Log1p em colunas monetárias para reduzir magnitude
    for col in ["total_items_price", "total_freight_value", "payment_value"]:
        if col in cluster_df.columns:
            cluster_df[col] = np.log1p(cluster_df[col].clip(lower=0))
    # Winsorização leve para evitar overflow no KMeans
    q_low = cluster_df.quantile(0.01)
    q_high = cluster_df.quantile(0.99)
    cluster_df = cluster_df.clip(lower=q_low, upper=q_high, axis=1)
    # Manter apenas linhas totalmente finitas
    cluster_df = cluster_df.dropna()
    cluster_df = cluster_df[np.isfinite(cluster_df).all(axis=1)]

    return cluster_df
