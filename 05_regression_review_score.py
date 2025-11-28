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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils import set_global_seed

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
    Plota gráfico de paridade (y_true vs y_pred) e salva em arquivo.
    """
    # Limitar visualização ao intervalo válido (1 a 5)
    y_true_plot = y_true.clip(1, 5)
    y_pred_plot = np.clip(y_pred, 1, 5)

    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=y_true_plot, y=y_pred_plot, alpha=0.3)
    min_val, max_val = 1, 5
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Valor real (review_score)")
    plt.ylabel("Valor previsto")
    plt.title(title)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xticks([1, 2, 3, 4, 5])
    plt.yticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Gráfico de paridade salvo em: {out_path.resolve()}")


def residuals_plot(y_true, y_pred, title: str, out_path: Path):
    """
    Plota gráfico de resíduos (y_true vs erro) e salva em arquivo.
    """
    y_pred_plot = np.clip(y_pred, 1, 5)
    residuals = y_true - y_pred_plot
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=y_pred_plot, y=residuals, alpha=0.3)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlim(1, 5)
    plt.xticks([1, 2, 3, 4, 5])
    plt.xlabel("Valor previsto (clamp 1-5)")
    plt.ylabel("Resíduo (y_true - y_pred)")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Gráfico de resíduos salvo em: {out_path.resolve()}")


def main() -> None:
    set_global_seed(42)
    DATA_FILE = Path("data/processed/olist_model_dataset.csv")
    RESULTS_DIR = Path("results/regression")

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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    # Linear
    metrics_linear = evaluate_regression(linear_model, tt.X_test, tt.y_test)
    print("\n=== Regressão Linear ===")
    print(f"RMSE: {metrics_linear['rmse']:.4f}")
    print(f"R²  : {metrics_linear['r2']:.4f}")
    results["linear_regression"] = metrics_linear

    # Ridge
    metrics_ridge = evaluate_regression(ridge_model, tt.X_test, tt.y_test)
    print("\n=== Regressão Ridge ===")
    print(f"RMSE: {metrics_ridge['rmse']:.4f}")
    print(f"R²  : {metrics_ridge['r2']:.4f}")
    results["ridge_regression"] = metrics_ridge

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
        out_path=RESULTS_DIR / "parity_plot_linear.png",
    )
    parity_plot(
        tt.y_test,
        y_pred_ridge,
        title="Regressão Ridge - Paridade (review_score)",
        out_path=RESULTS_DIR / "parity_plot_ridge.png",
    )

    residuals_plot(
        tt.y_test,
        y_pred_linear,
        title="Regressão Linear - Resíduos",
        out_path=RESULTS_DIR / "residuals_linear.png",
    )
    residuals_plot(
        tt.y_test,
        y_pred_ridge,
        title="Regressão Ridge - Resíduos",
        out_path=RESULTS_DIR / "residuals_ridge.png",
    )

    # ============================================================
    # 5) Salvar métricas em JSON
    # ============================================================
    results_file = RESULTS_DIR / "regression_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n💾 Resultados de regressão salvos em:", results_file.resolve())
    print("\n✅ Regressão da review_score concluída com sucesso.")


if __name__ == "__main__":
    main()
