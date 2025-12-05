#!/usr/bin/env python
"""
05_regression_review_score.py

Script para modelar a nota de review (review_score) como variável contínua,
usando Regressão Linear e Ridge.

Fluxo:
- Lê data/processed/olist_model_dataset.csv
- Usa review_score como target contínuo
- Usa as mesmas features numéricas/categóricas da parte supervisionada
- Divide em treino/teste
- Treina:
    - Regressão Linear
    - Regressão Ridge
- Avalia com:
    - RMSE
    - R²
- Salva:
    - results/regression/regression_results.json
    - results/regression/parity_plot_linear.png
    - results/regression/parity_plot_ridge.png
    - results/regression/residuals_linear.png
    - results/regression/residuals_ridge.png
"""

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from src.utils import set_global_seed
from src.utils import save_model

from src.feature_engineering import (
    get_supervised_feature_lists,
    build_preprocessor,
)
from src.models import (
    split_train_test,
    train_linear_regression,
    evaluate_regression,
)

# Estilo dos gráficos
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (8, 4)


def prepare_xy_for_regression(df: pd.DataFrame):
    """
    Prepara X e y para regressão contínua da review_score.

    - Remove linhas com review_score nulo
    - Usa as listas de features numéricas/categóricas definidas em get_supervised_feature_lists()
    """
    df = df.copy()
    df = df[~df["review_score"].isna()].copy()

    numeric_features, categorical_features = get_supervised_feature_lists()

    # Garantir que as colunas existem
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    X = df[numeric_features + categorical_features]
    y = df["review_score"].astype(float)

    return X, y, numeric_features, categorical_features


def parity_plot(y_true, y_pred, title: str, out_path: Path):
    """
    Plota gráfico de paridade (y_true vs y_pred) com melhorias de legibilidade:
    - densidade por hexbin para evitar overplotting
    - linha de identidade destacada
    - métricas no próprio gráfico
    - distribuição das previsões por classe real (1-5)
    """
    # Limitar visualização ao intervalo válido (1 a 5)
    y_true_plot = np.clip(y_true, 1, 5)
    y_pred_plot = np.clip(y_pred, 1, 5)
    df_plot = pd.DataFrame({"y_true": y_true_plot, "y_pred": y_pred_plot})
    # Usar buckets inteiros 1-5 para o eixo x do violin, evitando eixos contínuos
    df_plot["y_true_bucket"] = (
        df_plot["y_true"].round().clip(1, 5).astype(int)
    )

    # Métricas para anotar
    mae = mean_absolute_error(df_plot["y_true"], df_plot["y_pred"])
    # Calcula RMSE manualmente para compatibilidade com versões mais antigas do scikit-learn
    rmse = np.sqrt(mean_squared_error(df_plot["y_true"], df_plot["y_pred"]))
    r2 = r2_score(df_plot["y_true"], df_plot["y_pred"])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [3, 2]},
        constrained_layout=False,
    )

    min_val, max_val = 1, 5

    # --- Painel de paridade com densidade ---
    ax = axes[0]
    hb = ax.hexbin(
        df_plot["y_true"],
        df_plot["y_pred"],
        gridsize=30,
        cmap="magma",
        mincnt=1,
        linewidths=0.2,
    )
    # Jitter leve para evidenciar dispersão em cima do hexbin
    rng = np.random.default_rng(42)
    jitter = rng.normal(scale=0.03, size=(len(df_plot), 2))
    ax.scatter(
        df_plot["y_true"] + jitter[:, 0],
        df_plot["y_pred"] + jitter[:, 1],
        s=8,
        color="white",
        alpha=0.08,
        linewidth=0,
    )
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="gray",
        linewidth=2,
        linestyle="-",
        alpha=0.9,
    )
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Valor real (review_score)")
    ax.set_ylabel("Valor previsto")
    ax.set_title("Paridade com densidade")
    ax.grid(True, color="0.9", linestyle="--", linewidth=0.7)

    cbar = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Densidade de pontos")

    ax.text(
        0.02,
        0.98,
        f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="lightgray",
            alpha=0.9,
        ),
    )

    # --- Distribuição das previsões por classe real ---
    ax_dist = axes[1]
    sns.violinplot(
        data=df_plot,
        x="y_true_bucket",
        y="y_pred",
        hue="y_true_bucket",
        legend=False,
        ax=ax_dist,
        palette="Blues",
        cut=0,
        inner="quartile",
        linewidth=1,
        saturation=0.85,
    )
    ax_dist.set_xlabel("Valor real (review_score)")
    ax_dist.set_ylabel("Distribuição das previsões")
    ax_dist.set_ylim(min_val, max_val)
    ax_dist.set_yticks([1, 2, 3, 4, 5])
    ax_dist.set_title("Preditos agrupados pelo valor real")
    ax_dist.grid(axis="y", color="0.9", linestyle="--", linewidth=0.7)

    fig.suptitle(title, fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Gráfico de paridade salvo em: {out_path.resolve()}")


def residuals_plot(y_true, y_pred, title: str, out_path: Path):
    """
    Plota gráficos de resíduos com foco em interpretação:
    - Linha da média dos resíduos por bins de valor previsto (+/- 1 desvio padrão)
    - Boxplot dos resíduos por nota real (1-5)
    """
    min_val, max_val = 1, 5
    bin_width = 0.25

    y_pred_plot = np.clip(y_pred, min_val, max_val)
    residuals = y_true - y_pred_plot

    df_res = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred_plot,
            "residual": residuals,
        }
    )
    df_res["y_true_bucket"] = df_res["y_true"].round().clip(min_val, max_val).astype(int)

    # Bins de previsão
    bins = np.arange(min_val, max_val + bin_width, bin_width)
    df_res["pred_bin"] = pd.cut(df_res["y_pred"], bins=bins, include_lowest=True)
    agg = (
        df_res.groupby("pred_bin", observed=True)
        .agg(mean_res=("residual", "mean"), std_res=("residual", "std"))
        .dropna()
    )
    bin_centers = agg.index.map(lambda interval: (interval.left + interval.right) / 2)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 2]}, constrained_layout=False
    )

    # --- Linha média de resíduos por bin de predição ---
    ax0 = axes[0]
    hb = ax0.hexbin(
        df_res["y_pred"],
        df_res["residual"],
        gridsize=35,
        cmap="magma",
        mincnt=1,
        linewidths=0.2,
        alpha=0.45,
    )
    ax0.errorbar(
        bin_centers,
        agg["mean_res"],
        yerr=agg["std_res"],
        fmt="-o",
        color="dodgerblue",
        ecolor="lightblue",
        elinewidth=1.2,
        capsize=3,
        linewidth=1.5,
        markersize=4,
        label="Resíduo médio ±1 desvio",
    )
    ax0.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax0.set_xlim(min_val, max_val)
    ax0.set_xticks([1, 2, 3, 4, 5])
    ax0.set_xlabel("Valor previsto (clamp 1-5)")
    ax0.set_ylabel("Resíduo (y_true - y_pred)")
    ax0.set_title("Resíduo médio por faixa de previsão")
    ax0.grid(axis="y", color="0.9", linestyle="--", linewidth=0.7)
    cbar = fig.colorbar(hb, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Densidade de pontos")
    ax0.legend(loc="upper right")

    # --- Boxplot de resíduos por nota real ---
    ax1 = axes[1]
    sns.boxplot(
        data=df_res,
        x="y_true_bucket",
        y="residual",
        hue="y_true_bucket",
        legend=False,
        ax=ax1,
        palette="Blues",
        linewidth=1,
        showfliers=False,
    )
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("Valor real (review_score)")
    ax1.set_ylabel("Resíduo (y_true - y_pred)")
    ax1.set_title("Resíduos por valor real")
    ax1.grid(axis="y", color="0.9", linestyle="--", linewidth=0.7)

    fig.suptitle(title, fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Gráfico de resíduos salvo em: {out_path.resolve()}")


def main() -> None:
    set_global_seed(42)
    DATA_FILE = Path("data/processed/olist_model_dataset.csv")
    RESULTS_DIR = Path("results/regression")
    METRICS_DIR = RESULTS_DIR / "metrics"
    PLOTS_DIR = RESULTS_DIR / "plots"
    MODELS_DIR = RESULTS_DIR / "models"

    if not DATA_FILE.exists():
        raise SystemExit(
            f"❌ ERRO: Arquivo de dados não encontrado: {DATA_FILE.resolve()}.\n"
            f"Execute antes o 02_preprocessing.py."
        )

    print("📄 Lendo base de modelagem em:", DATA_FILE.resolve())
    df = pd.read_csv(DATA_FILE)

    # ============================================================
    # 1) Preparar X, y para regressão
    # ============================================================
    print("\n📌 Preparando features e target (review_score contínuo)...")
    X, y, numeric_features, categorical_features = prepare_xy_for_regression(df)

    print(f"Formato X: {X.shape}")
    print(f"Formato y: {y.shape}")
    print("Estatísticas de review_score:")
    print(y.describe())

    # Preprocessador (StandardScaler + OneHotEncoder)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Divisão treino/teste (sem estratificar, porque é regressão)
    tt = split_train_test(X, y, test_size=0.2, random_state=42, stratify=False)
    print("\nDivisão treino/teste realizada.")
    print(f"Treino: {tt.X_train.shape[0]} linhas")
    print(f"Teste : {tt.X_test.shape[0]} linhas")

    # ============================================================
    # 2) Treinar modelos de regressão
    # ============================================================
    print("\n🚀 Treinando modelos de regressão...")

    print("\n➡ Treinando Regressão Linear...")
    linear_model = train_linear_regression(
        preprocessor, tt.X_train, tt.y_train, ridge=False
    )

    print("\n➡ Treinando Regressão Ridge (alpha = 1.0)...")
    ridge_model = train_linear_regression(
        preprocessor, tt.X_train, tt.y_train, ridge=True, alpha=1.0
    )

    # ============================================================
    # 3) Avaliar modelos
    # ============================================================
    print("\n📊 Avaliando modelos...")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # Linear
    metrics_linear = evaluate_regression(linear_model, tt.X_test, tt.y_test)
    print("\n=== Regressão Linear ===")
    print(f"RMSE: {metrics_linear['rmse']:.4f}")
    print(f"R²  : {metrics_linear['r2']:.4f}")
    results["linear_regression"] = metrics_linear
    best_model = ("linear_regression", metrics_linear, linear_model)

    # Ridge
    metrics_ridge = evaluate_regression(ridge_model, tt.X_test, tt.y_test)
    print("\n=== Regressão Ridge ===")
    print(f"RMSE: {metrics_ridge['rmse']:.4f}")
    print(f"R²  : {metrics_ridge['r2']:.4f}")
    results["ridge_regression"] = metrics_ridge
    if metrics_ridge["rmse"] < best_model[1]["rmse"]:
        best_model = ("ridge_regression", metrics_ridge, ridge_model)

    # Salvar melhor modelo para uso em inferência/Streamlit
    best_model_name, best_model_metrics, best_model_pipe = best_model
    best_model_file = MODELS_DIR / "best_regression_model.joblib"
    save_model(best_model_pipe, best_model_file)
    with open(MODELS_DIR / "best_regression_model.json", "w", encoding="utf-8") as f:
        json.dump(
            {"best_model": best_model_name, "rmse": best_model_metrics["rmse"], "r2": best_model_metrics["r2"]},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"💾 Melhor modelo de regressão salvo em: {best_model_file.resolve()} ({best_model_name})")

    # ============================================================
    # 4) Gráficos (paridade e resíduos)
    # ============================================================
    print("\n📈 Gerando gráficos de paridade e resíduos...")

    # Para pegar y_pred, vamos reaproveitar evaluate_regression de forma simples:
    # como já temos os modelos treinados, chamamos .predict diretamente.
    y_pred_linear = linear_model.predict(tt.X_test)
    y_pred_ridge = ridge_model.predict(tt.X_test)

    parity_plot(
        tt.y_test,
        y_pred_linear,
        title="Regressão Linear - Paridade (review_score)",
    out_path=PLOTS_DIR / "parity_plot_linear.png",
    )
    parity_plot(
        tt.y_test,
        y_pred_ridge,
        title="Regressão Ridge - Paridade (review_score)",
    out_path=PLOTS_DIR / "parity_plot_ridge.png",
    )

    residuals_plot(
        tt.y_test,
        y_pred_linear,
        title="Regressão Linear - Resíduos",
    out_path=PLOTS_DIR / "residuals_linear.png",
    )
    residuals_plot(
        tt.y_test,
        y_pred_ridge,
        title="Regressão Ridge - Resíduos",
    out_path=PLOTS_DIR / "residuals_ridge.png",
    )

    # ============================================================
    # 5) Salvar métricas em JSON
    # ============================================================
    results_file = METRICS_DIR / "regression_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n💾 Resultados de regressão salvos em:", results_file.resolve())
    print("\n✅ Regressão da review_score concluída com sucesso.")


if __name__ == "__main__":
    main()
