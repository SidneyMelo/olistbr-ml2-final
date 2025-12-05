#!/usr/bin/env python
"""
01_data_overview.py

Script para fazer a análise exploratória inicial do dataset de e-commerce brasileiro (Olist).

O que ele faz:
- Carrega todas as tabelas usando src.preprocessing.load_olist_data
- Mostra resumo (linhas, colunas, tipos, nulos)
- Gera gráficos:
    - Distribuição de review_score
    - Número de pedidos por dia
    - Atraso na entrega x nota do review (boxplot)

Uso:
    python 01_data_overview.py --data-dir data

onde "data" é a pasta onde estão os arquivos:
    olist_orders_dataset.csv
    olist_order_items_dataset.csv
    olist_order_reviews_dataset.csv
    olist_customers_dataset.csv
    olist_products_dataset.csv
    olist_order_payments_dataset.csv
    olist_sellers_dataset.csv
    olist_geolocation_dataset.csv
    (e opcionalmente product_category_name_translation.csv)
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Deixar os gráficos com um padrão visual
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)

# Importar função do seu módulo
from src.preprocessing import load_olist_data

# Default data directory (project root / data)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


def summarize_dataframe(name: str, df: pd.DataFrame, n_head: int = 5) -> None:
    """Imprime um resumo textual de um DataFrame."""
    print("\n" + "=" * 80)
    print(f"TABELA: {name}")
    print(f"Formato: {df.shape[0]} linhas x {df.shape[1]} colunas\n")

    print("Tipos de dados:")
    print(df.dtypes)

    print("\nValores ausentes (%):")
    missing = df.isna().mean().sort_values(ascending=False) * 100
    missing = missing[missing > 0].round(2)
    if missing.empty:
        print("Nenhum valor ausente.")
    else:
        print(missing)

    print(f"\nExemplo das primeiras {n_head} linhas:")
    print(df.head(n_head))


def plot_review_distribution(reviews: pd.DataFrame, output_dir: Path) -> None:
    """Plota a distribuição de review_score e salva em PNG."""
    plt.figure()
    order = sorted(reviews["review_score"].dropna().unique())
    color_map = {
        1: "#d73027",
        2: "#fc8d59",
        3: "#fee08b",
        4: "#d9ef8b",
        5: "#1a9850",
    }
    palette = [color_map.get(int(s), sns.color_palette()[0]) for s in order]
    ax = sns.countplot(
        x="review_score",
        data=reviews,
        order=order,
        palette=palette,
        hue="review_score",
        legend=False,
    )

    total = len(reviews)
    mean_count = reviews["review_score"].value_counts().mean()
    for p, score in zip(ax.patches, order):
        count = p.get_height()
        pct = 100 * count / total if total else 0
        ax.annotate(
            f"{pct:.1f}%",
            (p.get_x() + p.get_width() / 2, count),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.axhline(mean_count, linestyle="--", color="gray", label="Média de avaliações por nota")
    plt.title("Distribuição das notas de review - dados desbalanceados")
    plt.xlabel("Nota do cliente")
    plt.ylabel("Quantidade de avaliações")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / "review_score_distribution.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    print(f"Grafico salvo em: {output_path}")

    # Se quiser ver na hora, descomenta:
    # plt.show()
    plt.close()


def plot_orders_per_month(orders: pd.DataFrame, output_dir: Path) -> None:
    """Plota o número de pedidos por mês (melhor para visualizar)."""
    df = orders.copy()
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["purchase_month"] = df["order_purchase_timestamp"].dt.to_period("M")

    orders_per_month = (
        df.groupby("purchase_month")
        .size()
        .rename("n_orders_month")
        .reset_index()
    )
    orders_per_month["purchase_month"] = orders_per_month["purchase_month"].dt.to_timestamp()

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    plt.bar(
        orders_per_month["purchase_month"],
        orders_per_month["n_orders_month"],
        color=sns.color_palette()[0],
        width=25,  # largura aproximada em dias
    )
    plt.title("Volume mensal de pedidos (2016–2018)")
    plt.xlabel("Mês da compra")
    plt.ylabel("Quantidade de pedidos")
    plt.xticks(rotation=45, ha="right")
    # Linha vertical para marcar Black Friday 2017
    ax.axvline(pd.Timestamp("2017-11-01"), linestyle="--", color="red", label="Black Friday 2017")
    # Anotação sobre queda final
    ax.text(
        orders_per_month["purchase_month"].max(),
        orders_per_month["n_orders_month"].min(),
        "Queda no final é esperada\n(dataset termina em 2018-10)",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    ax.legend()
    plt.tight_layout()

    output_path = output_dir / "orders_per_month.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    print(f"Grafico salvo em: {output_path}")
    # plt.show()
    plt.close()

    print("\nPeríodo coberto pela base de pedidos:")
    print("Data mínima:", df["order_purchase_timestamp"].min())
    print("Data máxima:", df["order_purchase_timestamp"].max())


def plot_delay_vs_review(orders: pd.DataFrame, reviews: pd.DataFrame, output_dir: Path) -> None:
    """Plota um boxplot do atraso de entrega por nota de review."""
    df_orders = orders.copy()
    df_reviews = reviews.copy()

    # Garantir tipos datetime
    date_cols = [
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    ]
    for c in date_cols:
        df_orders[c] = pd.to_datetime(df_orders[c])

    # Usar apenas uma review por pedido (primeira ocorrencia) para manter 1..5
    reviews_dedup = (
        df_reviews.sort_values("order_id")
        .drop_duplicates(subset="order_id", keep="first")[['order_id', 'review_score']]
    )

    df_merge = df_orders.merge(reviews_dedup, on="order_id", how="inner")

    # Calcular atraso em dias
    df_merge["delivery_delay_days"] = (
        df_merge["order_delivered_customer_date"]
        - df_merge["order_estimated_delivery_date"]
    ).dt.days

    df_merge = df_merge.dropna(subset=["delivery_delay_days", "review_score"])

    # Tratar review_score como categoria com rótulos 1..5
    df_merge["review_score_label"] = df_merge["review_score"].astype(int).astype(str)
    order_labels = [str(v) for v in sorted(df_merge["review_score"].dropna().unique())]

    plt.figure()
    ax = sns.boxplot(
        data=df_merge,
        x="review_score_label",
        y="delivery_delay_days",
        order=order_labels,
        showfliers=True,
    )
    sns.pointplot(
        data=df_merge,
        x="review_score_label",
        y="delivery_delay_days",
        order=order_labels,
        estimator="mean",
        color="black",
        markers="D",
        linestyles="",
        label="Média",
    )
    plt.title("Impacto do atraso na entrega sobre a nota do cliente")
    plt.xlabel("review_score")
    plt.ylabel("delivery_delay_days (dias)")
    plt.ylim(-50, 50)
    plt.axhline(0, color="red", linestyle="--", label="Entregue no prazo")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', linestyle='None', label='Outliers'))
    labels.append("Outliers (pontos fora da caixa)")
    plt.legend(handles=handles, labels=labels, loc="upper right", bbox_to_anchor=(0.98, 0.95))
    plt.tight_layout()

    output_path = output_dir / "delivery_delay_by_review_score.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    print(f"Grafico salvo em: {output_path}")
    # plt.show()
    plt.close()

    print("\nResumo estatístico do atraso por nota:")
    print(
        df_merge.groupby("review_score")["delivery_delay_days"]
        .describe()
        .round(2)
    )


def analyze_review_drivers(data: dict, top_n: int = 10, fig_dir: Optional[Path] = None) -> None:
    """Responde perguntas exploratorias sobre review_score (imprime e plota)."""
    analysis_dir = Path(fig_dir) / "analysis" if fig_dir else Path("figures/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    reviews = data["order_reviews"].copy()
    reviews = reviews.dropna(subset=["review_score"])
    reviews["review_score"] = reviews["review_score"].astype(float)

    # --- Tempo de entrega x nota ---
    orders = data["orders"].copy()
    date_cols = [
        "order_purchase_timestamp",
        "order_estimated_delivery_date",
        "order_delivered_customer_date",
    ]
    for c in date_cols:
        orders[c] = pd.to_datetime(orders[c])
    orders["delivery_delay_days"] = (
        orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
    ).dt.days

    reviews_orders = reviews.merge(
        orders[["order_id", "delivery_delay_days", "customer_id"]],
        on="order_id",
        how="left",
    )
    corr_delay = (
        reviews_orders[["review_score", "delivery_delay_days"]]
        .dropna()
        .corr()
        .loc["review_score", "delivery_delay_days"]
    )
    print("\n==== Tempo de entrega x nota ====")
    print(f"Correlacao (review_score vs delivery_delay_days): {corr_delay:.3f}")
    print(
        reviews_orders.assign(delayed=reviews_orders["delivery_delay_days"] > 0)
        .groupby("delayed")["review_score"]
        .agg(media="mean", qtd="size")
        .round(2)
    )
    delay_stats = (
        reviews_orders.assign(delayed=reviews_orders["delivery_delay_days"] > 0)
        .groupby("delayed")["review_score"]
        .agg(media="mean", qtd="size")
        .reset_index()
        .assign(delayed=lambda df: df["delayed"].map({True: "Sim", False: "Não"}))
    )
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=delay_stats,
        x="delayed",
        y="media",
        hue="delayed",
        palette=["#4daf4a", "#e41a1c"],  # não atrasado (verde), atrasado (vermelho)
        legend=False,
    )
    for i, row in delay_stats.iterrows():
        plt.text(i, row["media"] + 0.05, f"{row['media']:.1f}", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Média review_score")
    plt.xlabel("Pedido atrasado?")
    plt.title("Impacto do atraso na entrega na satisfação do cliente")
    plt.ylim(0, max(delay_stats["media"]) + 0.5)
    plt.tight_layout()
    delay_plot = analysis_dir / "review_score_by_delay.png"
    plt.savefig(delay_plot, dpi=120)
    plt.close()
    print(f"Grafico: {delay_plot.resolve()}")

    # --- Categoria de produto ---
    items = data["order_items"][["order_id", "product_id"]]
    products = data["products"][["product_id", "product_category_name"]]
    reviews_prod = (
        reviews.merge(items, on="order_id", how="left")
        .merge(products, on="product_id", how="left")
    )
    cat_stats = (
        reviews_prod.groupby("product_category_name")["review_score"]
        .agg(qtd="size", media="mean")
        .assign(pct_baixa=lambda s: reviews_prod.groupby("product_category_name")["review_score"]
                .apply(lambda x: (x <= 2).mean() * 100))
    )
    cat_stats = cat_stats[cat_stats["qtd"] >= 50].sort_values("pct_baixa", ascending=False)
    print("\n==== Categorias com maior insatisfacao (qtd>=50) ====")
    print(cat_stats[["qtd", "media", "pct_baixa"]].round(3).head(top_n))
    if not cat_stats.empty:
        top_cats = cat_stats.head(top_n).reset_index()
        overall_pct_bad = (reviews_prod["review_score"] <= 2).mean() * 100
        plt.figure(figsize=(9, 5))
        # mapa de cores por intensidade
        cmap = sns.color_palette("Reds", as_cmap=True)
        norm = plt.Normalize(top_cats["pct_baixa"].min(), top_cats["pct_baixa"].max())
        colors = [cmap(norm(v)) for v in top_cats["pct_baixa"]]

        sns.barplot(
            data=top_cats,
            y="product_category_name",
            x="pct_baixa",
            hue="product_category_name",
            palette=colors,
            legend=False,
        )
        plt.xlabel("Pct avaliacoes ruins (nota <= 2) (%)")
        plt.ylabel("Categoria")
        plt.title("Categorias com maior taxa de avaliacoes ruins")
        xmax = max(top_cats["pct_baixa"].max() * 1.1, overall_pct_bad * 1.2)
        plt.xlim(0, xmax)
        plt.axvline(overall_pct_bad, linestyle="--", color="gray", label="Média geral")
        plt.legend(loc="upper right")
        plt.tight_layout()
        cat_plot = analysis_dir / "categories_most_negative.png"
        plt.savefig(cat_plot, dpi=120)
        plt.close()
        print(f"Grafico: {cat_plot.resolve()}")
        # salvar tabela limpa (sem poluir o grafico)
        table_file = analysis_dir / "categories_most_negative_top.csv"
        top_cats.to_csv(table_file, index=False)
        print(f"Tabela (top categorias) salva em: {table_file.resolve()}")

    # --- Regiao (estado) ---
    customers = data["customers"][["customer_id", "customer_state", "customer_city"]]
    reviews_geo = (
        reviews_orders.merge(customers, on="customer_id", how="left")
    )
    state_stats = (
        reviews_geo.groupby("customer_state")["review_score"]
        .agg(qtd="size", media="mean")
        .assign(pct_baixa=lambda s: reviews_geo.groupby("customer_state")["review_score"]
                .apply(lambda x: (x <= 2).mean() * 100))
    )
    state_stats = state_stats[state_stats["qtd"] >= 100].sort_values("media")
    print("\n==== Estados com pior media (qtd>=100) ====")
    print(state_stats[["qtd", "media", "pct_baixa"]].round(3).head(top_n))
    if not state_stats.empty:
        top_states = state_stats.head(top_n).reset_index()
        overall_pct_bad_state = (reviews_geo["review_score"] <= 2).mean() * 100
        plt.figure(figsize=(9, 5))
        cmap = sns.color_palette("Reds", as_cmap=True)
        norm = plt.Normalize(top_states["pct_baixa"].min(), top_states["pct_baixa"].max())
        colors = [cmap(norm(v)) for v in top_states["pct_baixa"]]
        sns.barplot(
            data=top_states,
            y="customer_state",
            x="pct_baixa",
            hue="customer_state",
            palette=colors,
            legend=False,
        )
        plt.xlabel("Pct avaliacoes ruins (nota <= 2) (%)")
        plt.ylabel("Estado")
        plt.title("Quais estados têm maior taxa de notas ruins?")
        xmax_s = max(top_states["pct_baixa"].max() * 1.1, overall_pct_bad_state * 1.2)
        plt.xlim(0, xmax_s)
        plt.axvline(overall_pct_bad_state, linestyle="--", color="gray", label="Média geral")
        plt.legend(loc="upper right")
        plt.tight_layout()
        state_plot = analysis_dir / "states_most_negative.png"
        plt.savefig(state_plot, dpi=120)
        plt.close()
        print(f"Grafico: {state_plot.resolve()}")
        state_table = analysis_dir / "states_most_negative_top.csv"
        top_states.to_csv(state_table, index=False)
        print(f"Tabela (top estados) salva em: {state_table.resolve()}")


def main() -> None:
    data_dir = Path("data")
    fig_dir = Path("figures")

    if not data_dir.exists():
        raise SystemExit(f"Pasta de dados não encontrada: {data_dir.resolve()}")

    print("Usando pasta de dados:", data_dir.resolve())

    # Carregar dados
    data = load_olist_data(data_dir)

    print("\nTabelas carregadas:")
    for name, df in data.items():
        print(f"{name:20s} -> {df.shape[0]:7d} linhas | {df.shape[1]:3d} colunas")

    # Resumo das tabelas principais
    summarize_dataframe("orders", data["orders"])
    summarize_dataframe("order_items", data["order_items"])
    summarize_dataframe("order_reviews", data["order_reviews"])
    summarize_dataframe("customers", data["customers"])
    summarize_dataframe("products", data["products"])
    summarize_dataframe("payments", data["payments"])
    summarize_dataframe("sellers", data["sellers"])
    summarize_dataframe("geolocation", data["geolocation"])

    if "product_translation" in data:
        summarize_dataframe("product_translation", data["product_translation"])

    # Gráficos principais
    plot_review_distribution(data["order_reviews"], fig_dir)
    plot_orders_per_month(data["orders"], fig_dir)
    plot_delay_vs_review(data["orders"], data["order_reviews"], fig_dir)

    # Sumários para responder perguntas chave
    analyze_review_drivers(data, fig_dir=fig_dir)

    print("\nAnálise exploratória inicial concluída.")


if __name__ == "__main__":
    main()
