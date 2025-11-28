#!/usr/bin/env python
"""
04_clustering.py

Script para aplicar K-Means no dataset de modelagem Olist e identificar
grupos (clusters) de pedidos/clientes com base em variáveis numéricas
como preço, frete, tempo de entrega e atraso.

Fluxo:
- Lê: data/processed/olist_model_dataset.csv
- Usa: prepare_clustering_features (src.feature_engineering)
- Ajusta KMeans (n_clusters fixo = 3, mas pode ser mudado)
- Atribui rótulos de cluster às linhas
- Salva:
    - data/processed/olist_model_dataset_with_clusters.csv
    - results/clustering/cluster_summary.csv
    - results/clustering/review_score_by_cluster.png
    - results/clustering/boxplots_features_by_cluster.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.feature_engineering import prepare_clustering_features
from src.clustering import fit_kmeans, assign_clusters
from src.utils import set_global_seed

# Estilo dos gráficos
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 5)


def main() -> None:
    set_global_seed(42)
    # Caminhos fixos
    DATA_FILE = Path("data/processed/olist_model_dataset.csv")
    OUT_DATA_FILE = Path("data/processed/olist_model_dataset_with_clusters.csv")
    RESULTS_DIR = Path("results/clustering")

    if not DATA_FILE.exists():
        raise SystemExit(
            f"❌ ERRO: Arquivo de dados não encontrado: {DATA_FILE.resolve()}.\n"
            f"Execute antes o 02_preprocessing.py."
        )

    print("📄 Lendo base de modelagem em:", DATA_FILE.resolve())
    df = pd.read_csv(DATA_FILE)

    # ============================================================
    # 1) Preparar features para clustering
    # ============================================================
    print("\n📌 Preparando features numéricas para clustering...")
    cluster_df = prepare_clustering_features(df)

    print("Formato de cluster_df:", cluster_df.shape)
    print("Colunas usadas no clustering:", list(cluster_df.columns))

    # ============================================================
    # 2) Ajustar K-Means
    # ============================================================
    # Número de clusters fixo, mas você pode ajustar (3, 4, 5...)
    N_CLUSTERS = 3
    print(f"\n🚀 Ajustando KMeans com n_clusters = {N_CLUSTERS}...")

    kmeans_model = fit_kmeans(cluster_df, n_clusters=N_CLUSTERS, random_state=42)

    # Atribuir rótulos
    labels = assign_clusters(kmeans_model, cluster_df)
    print("Clusters encontrados:", np.unique(labels, return_counts=True))

    # ============================================================
    # 3) Adicionar clusters ao dataframe original
    # ============================================================
    print("\n🧩 Adicionando rótulos de cluster ao dataframe original...")

    # Garantir coluna cluster com NaN e depois preencher para as linhas que têm features válidas
    df["cluster"] = np.nan
    df.loc[cluster_df.index, "cluster"] = labels

    # Converter para inteiro (nullable)
    df["cluster"] = df["cluster"].astype("Int64")

    # ============================================================
    # 4) Resumo por cluster
    # ============================================================
    print("\n📊 Gerando resumo por cluster...")

    # Escolher algumas colunas chave para resumir
    cols_summary = [
        "total_items_price",
        "total_freight_value",
        "payment_value",
        "n_items",
        "delivery_time_days",
        "delivery_delay_days",
        "review_score",
    ]
    cols_summary = [c for c in cols_summary if c in df.columns]

    cluster_summary = (
        df.dropna(subset=["cluster"])
        .groupby("cluster")[cols_summary]
        .agg(["mean", "median", "std", "min", "max", "count"])
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = RESULTS_DIR / "cluster_summary.csv"
    cluster_summary.to_csv(summary_file)
    print("Resumo por cluster salvo em:", summary_file.resolve())

    # ============================================================
    # 5) Gráfico: média de review_score por cluster
    # ============================================================
    if "review_score" in df.columns:
        print("\n📈 Gerando gráfico de média de review_score por cluster...")

        review_by_cluster = (
            df.dropna(subset=["cluster"])
            .groupby("cluster")["review_score"]
            .mean()
            .reset_index()
        )

        plt.figure()
        sns.barplot(data=review_by_cluster, x="cluster", y="review_score")
        plt.title("Média de review_score por cluster")
        plt.xlabel("Cluster")
        plt.ylabel("Média de review_score")
        plt.ylim(1, 5)
        plt.tight_layout()

        plot_file = RESULTS_DIR / "review_score_by_cluster.png"
        plt.savefig(plot_file, dpi=120)
        plt.close()
        print("Gráfico salvo em:", plot_file.resolve())

    # ============================================================
    # 6) Boxplots de algumas features por cluster
    # ============================================================
    print("\n📈 Gerando boxplots de features por cluster...")

    features_for_plot = [
        "total_items_price",
        "total_freight_value",
        "payment_value",
        "n_items",
        "delivery_time_days",
        "delivery_delay_days",
    ]
    features_for_plot = [c for c in features_for_plot if c in df.columns]

    df_plot = df.dropna(subset=["cluster"] + features_for_plot).copy()
    df_plot["cluster"] = df_plot["cluster"].astype(int)

    # Fazer um boxplot por feature, todos no mesmo figure usando subplots
    n_features = len(features_for_plot)
    if n_features > 0:
        n_cols = 2
        n_rows = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten()

        for i, feature in enumerate(features_for_plot):
            ax = axes[i]
            sns.boxplot(
                data=df_plot,
                x="cluster",
                y=feature,
                ax=ax,
            )
            ax.set_title(f"{feature} por cluster")

        # Esconder eixos sobrando
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        boxplot_file = RESULTS_DIR / "boxplots_features_by_cluster.png"
        plt.savefig(boxplot_file, dpi=120)
        plt.close()
        print("Boxplots salvos em:", boxplot_file.resolve())
    else:
        print("⚠ Nenhuma feature disponível para boxplot.")

    # ============================================================
    # 7) Salvar dataset com clusters
    # ============================================================
    OUT_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DATA_FILE, index=False)
    print("\n💾 Dataset com coluna 'cluster' salvo em:")
    print("   ", OUT_DATA_FILE.resolve())

    print("\n✅ Clustering concluído com sucesso.")


if __name__ == "__main__":
    main()
