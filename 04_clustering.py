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
    METRICS_DIR = RESULTS_DIR / "metrics"
    PLOTS_DIR = RESULTS_DIR / "plots"

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

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_file = METRICS_DIR / "cluster_summary.csv"
    cluster_summary.to_csv(summary_file)
    print("Resumo por cluster salvo em:", summary_file.resolve())

    # ============================================================
    # 5) Gráfico: média de review_score por cluster
    # ============================================================
    if "review_score" in df.columns:
        print("\n📈 Gerando gráfico de média de review_score por cluster...")

        # Nomes explicativos (podem variar conforme a rodada, mas seguem o perfil médio observado):
        # 0: Ticket médio-alto, 1 produto, entrega adiantada -> "Alto Valor (1 produto, adiantado)"
        # 1: Ticket baixo, 1 produto, entrega adiantada, nota mais alta -> "Baixo Valor (adiantado, 1 produto)"
        # 2: Ticket médio-alto, múltiplos produtos, nota mais baixa -> "Multi-produtos (médio)"
        cluster_names = {
            0: "Alto Valor (1 produto, adiantado)",
            1: "Baixo Valor (adiantado, 1 produto)",
            2: "Multi-produtos (médio)",
        }
        
        # Criar coluna com nomes
        df_plot = df.dropna(subset=["cluster", "review_score"]).copy()
        df_plot["cluster_name"] = df_plot["cluster"].map(cluster_names)

        # 2. Usar Cores para Distinguir
        palette = {
            "Alto Valor (1 produto, adiantado)": "#95a5a6",              # Cinza
            "Baixo Valor (adiantado, 1 produto)": "#2ecc71",             # Verde
            "Multi-produtos (médio)": "#e67e22",                         # Laranja
        }

        plt.figure(figsize=(8, 5))
        
        # 5. Mostrar a Variabilidade (Barra de Erro)
        # Usando o dataset completo, o seaborn calcula o IC (linha preta)
        order_plot = (
            df_plot.groupby("cluster_name")["review_score"]
            .mean()
            .sort_values(ascending=False)
            .index.tolist()
        )

        ax = sns.barplot(
            data=df_plot,
            x="cluster_name",
            y="review_score",
            hue="cluster_name", # Fix FutureWarning
            palette=palette,
            order=order_plot, # ordem dinâmica pela média da nota
            capsize=0.1,
            legend=False # Fix FutureWarning
        )

        # 4. Adicionar Linha de Média Global
        global_mean = df_plot["review_score"].mean()
        plt.axhline(y=global_mean, color="black", linestyle="--", label=f"Média Global ({global_mean:.2f})")
        plt.legend()

        # 1. Adicionar Rótulos de Valor (Data Labels)
        # Como estamos usando barplot com agregação, precisamos pegar os valores das barras
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3, fontsize=10, fontweight="bold")

        plt.title("Média de Nota de Avaliação por Cluster (com IC 95%)", fontsize=14)
        plt.xlabel("Cluster", fontsize=12)
        plt.ylabel("Nota de Avaliação", fontsize=12)
        plt.ylim(1, 5.5) # Espaço extra para os labels
        plt.tight_layout()

        plot_file = PLOTS_DIR / "review_score_by_cluster.png"
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
    
    # Tradução dos títulos
    feature_titles = {
        "total_items_price": "Preço Total (R$)",
        "total_freight_value": "Frete Total (R$)",
        "payment_value": "Valor Pago (R$)",
        "n_items": "Qtd. Itens",
        "delivery_time_days": "Tempo de Entrega (dias)",
        "delivery_delay_days": "Atraso (dias)",
    }

    features_for_plot = [c for c in features_for_plot if c in df.columns]

    # Reutilizar df_plot que já tem cluster_name (criado no passo anterior)
    # Se por acaso não tiver rodado o passo 5 (review_score), recriar aqui:
    if "cluster_name" not in df_plot.columns:
         df_plot = df.dropna(subset=["cluster"] + features_for_plot).copy()
         df_plot["cluster_name"] = df_plot["cluster"].map(cluster_names)
    
    # Fazer um boxplot por feature, todos no mesmo figure usando subplots
    n_features = len(features_for_plot)
    if n_features > 0:
        n_cols = 2
        n_rows = int(np.ceil(n_features / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten()

        order_plot = (
            df_plot.groupby("cluster_name")["review_score"].mean().sort_values(ascending=False).index.tolist()
            if "review_score" in df_plot.columns else list(cluster_names.values())
        )

        for i, feature in enumerate(features_for_plot):
            ax = axes[i]
            sns.boxplot(
                data=df_plot,
                x="cluster_name",
                y=feature,
                hue="cluster_name",
                palette=palette,
                order=order_plot,
                showfliers=False, # Remover outliers visualmente
                ax=ax,
                dodge=False,
                legend=False,
            )
            ax.set_title(feature_titles.get(feature, feature), fontsize=12)
            ax.set_xlabel("Cluster")
            ax.set_ylabel("")
            # Legenda removida para evitar warnings; cores seguem o mapping.

        # Esconder eixos sobrando
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        boxplot_file = PLOTS_DIR / "boxplots_features_by_cluster.png"
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
