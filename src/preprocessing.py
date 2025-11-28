"""
Módulo de pré-processamento do dataset Olist.

Responsável por:
- Carregar todos os CSVs.
- Fazer merges principais (pedidos + itens + reviews + clientes + produtos).
- Limpar registros inválidos (ex: pedidos cancelados).
- Construir um dataframe base para modelagem.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def load_olist_data(base_path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    Carrega todos os arquivos CSV do dataset Olist a partir de uma pasta base.

    Espera encontrar arquivos com estes nomes (Kaggle Olist):
        - olist_orders_dataset.csv
        - olist_order_items_dataset.csv
        - olist_order_reviews_dataset.csv
        - olist_customers_dataset.csv
        - olist_products_dataset.csv
        - olist_order_payments_dataset.csv
        - olist_sellers_dataset.csv
        - olist_geolocation_dataset.csv
        - product_category_name_translation.csv (opcional)

    Parameters
    ----------
    base_path : str | Path
        Caminho da pasta contendo os CSVs.

    Returns
    -------
    dict
        Dicionário {nome_tabela: dataframe}.
    """
    base_path = Path(base_path)

    def _read_csv(name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
        fp = base_path / name
        if parse_dates:
            return pd.read_csv(fp, parse_dates=parse_dates)
        return pd.read_csv(fp)

    data = {
        "orders": _read_csv(
            "olist_orders_dataset.csv",
            parse_dates=[
                "order_purchase_timestamp",
                "order_approved_at",
                "order_delivered_carrier_date",
                "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
        ),
        "order_items": _read_csv("olist_order_items_dataset.csv"),
        "order_reviews": _read_csv(
            "olist_order_reviews_dataset.csv",
            parse_dates=["review_creation_date", "review_answer_timestamp"],
        ),
        "customers": _read_csv("olist_customers_dataset.csv"),
        "products": _read_csv("olist_products_dataset.csv"),
        "payments": _read_csv("olist_order_payments_dataset.csv"),
        "sellers": _read_csv("olist_sellers_dataset.csv"),
        "geolocation": _read_csv("olist_geolocation_dataset.csv"),
    }

    # tradução de categoria é opcional
    translation_path = base_path / "product_category_name_translation.csv"
    if translation_path.exists():
        data["product_translation"] = _read_csv(
            "product_category_name_translation.csv"
        )

    return data


def build_base_orders_reviews(
    data: Dict[str, pd.DataFrame],
    drop_canceled: bool = True,
) -> pd.DataFrame:
    """
    Constrói um dataframe base a partir dos principais CSVs:
    orders + order_items + order_reviews + customers + products + payments.

    Essa base é utilizada depois para feature engineering e modelagem.

    Parameters
    ----------
    data : dict
        Dicionário retornado por load_olist_data.
    drop_canceled : bool, default=True
        Se True, remove pedidos com status "canceled".

    Returns
    -------
    pd.DataFrame
        DataFrame consolidado por pedido (order_id).
    """
    orders = data["orders"].copy()
    items = data["order_items"].copy()
    reviews = data["order_reviews"].copy()
    customers = data["customers"].copy()
    products = data["products"].copy()
    payments = data["payments"].copy()

    # 1) Remover pedidos cancelados (opcional)
    if drop_canceled:
        orders = orders[orders["order_status"] != "canceled"].copy()

    # 2) Merge pedidos + reviews (1:N → algumas ordens possuem vários reviews, pegamos a média)
    reviews_agg = (
        reviews.groupby("order_id", as_index=False)
        .agg(
            review_score=("review_score", "mean"),
            review_count=("review_id", "count"),
        )
    )

    df = orders.merge(reviews_agg, on="order_id", how="left")

    # 3) Itens do pedido → agregamos por pedido
    items_agg = (
        items.groupby("order_id", as_index=False)
        .agg(
            n_items=("order_item_id", "count"),
            product_ids=("product_id", lambda x: list(x)),
            total_items_price=("price", "sum"),
            total_freight_value=("freight_value", "sum"),
            sellers_ids=("seller_id", lambda x: list(x)),
        )
    )
    df = df.merge(items_agg, on="order_id", how="left")

    # 4) Pagamentos → agregamos por pedido
    payments_agg = (
        payments.groupby("order_id", as_index=False)
        .agg(
            payment_sequential=("payment_sequential", "max"),
            payment_type=("payment_type", lambda x: x.iloc[0]),
            payment_installments=("payment_installments", "max"),
            payment_value=("payment_value", "sum"),
        )
    )
    df = df.merge(payments_agg, on="order_id", how="left")

    # 5) Clientes
    df = df.merge(customers, on="customer_id", how="left")

    # 6) Produtos → como temos vários produtos por pedido, vamos pegar categoria do 1º item
    first_item = (
        items.sort_values("order_item_id")
        .groupby("order_id", as_index=False)
        .first()[["order_id", "product_id"]]
    )
    first_item = first_item.merge(
        products[["product_id", "product_category_name"]], on="product_id", how="left"
    )
    df = df.merge(first_item[["order_id", "product_category_name"]], on="order_id", how="left")

    # 7) Traduzir categoria (se existir)
    if "product_translation" in data:
        translation = data["product_translation"].copy()
        df = df.merge(
            translation,
            on="product_category_name",
            how="left",
        )

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features temporais básicas de entrega, atraso e datas da compra.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe base contendo colunas de datas.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # Tempo total até entrega (em dias)
    df["delivery_time_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # Prazo estimado (em dias)
    df["estimated_delivery_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.days

    # Atraso (positivo = entregou depois do estimado)
    df["delivery_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days

    # Onde não tem data de entrega (talvez não entregue ou nulo), colocamos NaN
    df["delivery_time_days"] = df["delivery_time_days"].astype("float")
    df["estimated_delivery_days"] = df["estimated_delivery_days"].astype("float")
    df["delivery_delay_days"] = df["delivery_delay_days"].astype("float")

    # Features de data da compra
    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek

    return df


def preprocess_base(
    base_path: str | Path,
    drop_canceled: bool = True,
) -> pd.DataFrame:
    """
    Pipeline completo de pré-processamento:
    - Carrega dados
    - Constrói base consolidada
    - Adiciona features temporais

    Parameters
    ----------
    base_path : str | Path
        Caminho para a pasta com csvs.
    drop_canceled : bool, default=True
        Remove pedidos cancelados.

    Returns
    -------
    pd.DataFrame
    """
    data = load_olist_data(base_path)
    df = build_base_orders_reviews(data, drop_canceled=drop_canceled)
    df = add_time_features(df)

    # Remove registros sem review_score (se objetivo for modelar satisfação)
    df = df[~df["review_score"].isna()].copy()

    return df