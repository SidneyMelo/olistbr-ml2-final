#!/usr/bin/env python
"""
03_supervised_models.py

Script para treinar e comparar modelos de classificação supervisionada
no dataset Olist já pré-processado.

Fluxo:
- Lê data/processed/olist_model_dataset.csv
- Usa target: review_binary (0 = ruim, 1 = bom, neutros (3) são ignorados)
- Prepara X, y com as features numéricas e categóricas definidas em feature_engineering.py
- Divide em treino/teste
- Treina os modelos:
    - Naive Bayes
    - Regressão Logística
    - SVM
    - Random Forest
    - XGBoost (se disponível)
- Avalia:
    - Accuracy
    - classification_report
    - matriz de confusão
- Salva resultados em results/supervised_results.json
- Salva gráfico de barras com acurácia em results/accuracy_barplot.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone

from src.feature_engineering import (
    prepare_supervised_xy,
    build_preprocessor,
)
from src.utils import set_global_seed
from src.models import (
    split_train_test,
    train_naive_bayes_classifier,
    train_logistic_regression_classifier,
    train_svm_classifier,
    train_random_forest_classifier,
    train_xgboost_classifier,
    evaluate_classifier,
)

# Estilo dos gráficos
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (8, 4)


def main() -> None:
    set_global_seed(42)
    DATA_FILE = Path("data/processed/olist_model_dataset.csv")
    RESULTS_DIR = Path("results")
    # Para validar o fluxo rapidamente, limitar amostra usada no SVM
    SVM_MAX_TRAIN_SAMPLES = 30000

    if not DATA_FILE.exists():
        raise SystemExit(f"❌ ERRO: Arquivo de dados não encontrado: {DATA_FILE.resolve()}.\n"
                         f"Execute antes o 02_preprocessing.py.")

    print("📄 Lendo base de modelagem em:", DATA_FILE.resolve())
    df = pd.read_csv(DATA_FILE)

    # ============================================================
    # 1) Preparar X, y para classificação (target = review_binary)
    # ============================================================
    print("\n📌 Preparando features e target (review_binary)...")

    X, y, numeric_features, categorical_features = prepare_supervised_xy(
        df,
        target_col="review_binary",
    )

    print(f"Formato X: {X.shape}")
    print(f"Formato y: {y.shape}")
    print("Classes em y:", np.unique(y, return_counts=True))

    # Construir preprocessor (StandardScaler + OneHotEncoder)
    preprocessor_template = build_preprocessor(numeric_features, categorical_features)

    # Dividir em treino/teste
    tt = split_train_test(X, y, test_size=0.2, random_state=42, stratify=True)
    print("\nDivisão treino/teste realizada.")
    print(f"Treino: {tt.X_train.shape[0]} linhas")
    print(f"Teste : {tt.X_test.shape[0]} linhas")

    # ============================================================
    # 2) Treinar modelos
    # ============================================================
    print("\n🚀 Treinando modelos supervisionados...")

    models = {}

    print("\n➡ Treinando Naive Bayes...")
    models["naive_bayes"] = train_naive_bayes_classifier(
        clone(preprocessor_template), tt.X_train, tt.y_train
    )

    print("\n➡ Treinando Regressão Logística...")
    models["logistic_regression"] = train_logistic_regression_classifier(
        clone(preprocessor_template), tt.X_train, tt.y_train
    )

    print("\n➡ Treinando SVM (kernel Linear)...") #print("\n➡ Treinando SVM (kernel RBF)...")
    # Amostrar para validar o fluxo mais rápido
    if len(tt.X_train) > SVM_MAX_TRAIN_SAMPLES:
        tt_svm = tt.X_train.sample(n=SVM_MAX_TRAIN_SAMPLES, random_state=42)
        y_svm = tt.y_train.loc[tt_svm.index]
        print(f"   > Usando amostra de {len(tt_svm)} linhas para o SVM (limite: {SVM_MAX_TRAIN_SAMPLES})")
    else:
        tt_svm = tt.X_train
        y_svm = tt.y_train

    models["svm_linear"] = train_svm_classifier(
        clone(preprocessor_template), tt_svm, y_svm, kernel="linear", C=1.0, gamma="scale" # kernel="rbf"
    )

    print("\n➡ Treinando Random Forest...")
    models["random_forest"] = train_random_forest_classifier(
        clone(preprocessor_template), tt.X_train, tt.y_train, n_estimators=200, random_state=42
    )

    print("\n➡ Tentando treinar XGBoost...")
    xgb_model = train_xgboost_classifier(
        clone(preprocessor_template), tt.X_train, tt.y_train, random_state=42
    )
    if xgb_model is not None:
        models["xgboost"] = xgb_model
    else:
        print("⏭ XGBoost não disponível, ignorando esse modelo.")

    # ============================================================
    # 3) Avaliar modelos
    # ============================================================
    print("\n📊 Avaliando modelos...")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    accuracies = []

    for name, model in models.items():
        print(f"\n=== Modelo: {name} ===")
        metrics = evaluate_classifier(model, tt.X_test, tt.y_test)

        acc = metrics["accuracy"]
        print(f"Accuracy: {acc:.4f}")

        # classification_report como texto legível
        report_dict = metrics["classification_report"]
        # confusion_matrix é um numpy array
        cm = metrics["confusion_matrix"]

        # guardar num dicionário serializável
        results[name] = {
            "accuracy": float(acc),
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
        }

        accuracies.append((name, acc))

    # ============================================================
    # 4) Salvar resultados em JSON
    # ============================================================
    results_file = RESULTS_DIR / "supervised_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n💾 Resultados salvos em:", results_file.resolve())

    # ============================================================
    # 5) Gráfico de comparação de acurácia
    # ============================================================
    model_names = [x[0] for x in accuracies]
    acc_values = [x[1] for x in accuracies]

    plt.figure()
    sns.barplot(
        x=model_names,
        y=acc_values,
    )
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy (teste)")
    plt.xlabel("Modelo")
    plt.title("Comparação de acurácia entre modelos")
    plt.xticks(rotation=30)
    plt.tight_layout()

    acc_plot_file = RESULTS_DIR / "accuracy_barplot.png"
    plt.savefig(acc_plot_file, dpi=120)
    plt.close()

    print("📈 Gráfico de acurácia salvo em:", acc_plot_file.resolve())

    # ============================================================
    # 6) Melhor modelo: matriz de confusão e precision/recall por classe
    # ============================================================
    best_name, best_acc = max(accuracies, key=lambda x: x[1])
    best_metrics = results[best_name]

    cm = np.array(best_metrics["confusion_matrix"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title(f"Matriz de confusão - {best_name} (acc={best_acc:.3f})")
    cm_plot_file = RESULTS_DIR / f"confusion_matrix_{best_name}.png"
    plt.tight_layout()
    plt.savefig(cm_plot_file, dpi=120)
    plt.close()
    print("📈 Matriz de confusão do melhor modelo salva em:", cm_plot_file.resolve())

    report_df = pd.DataFrame(best_metrics["classification_report"]).T
    per_class = report_df.loc[["0", "1"], ["precision", "recall", "f1-score", "support"]]
    per_class_file = RESULTS_DIR / f"class_report_{best_name}.csv"
    per_class.to_csv(per_class_file, index=True)
    print("📈 Precision/recall por classe salvo em CSV:", per_class_file.resolve())

    melted = per_class[["precision", "recall"]].reset_index().melt(
        id_vars="index", value_vars=["precision", "recall"]
    )
    plt.figure(figsize=(6, 4))
    sns.barplot(data=melted, x="value", y="index", hue="variable")
    plt.xlabel("Valor")
    plt.ylabel("Classe")
    plt.title(f"Precision/Recall por classe - {best_name}")
    plt.xlim(0, 1)
    plt.tight_layout()
    pr_plot_file = RESULTS_DIR / f"precision_recall_{best_name}.png"
    plt.savefig(pr_plot_file, dpi=120)
    plt.close()
    print("📈 Gráfico precision/recall do melhor modelo salvo em:", pr_plot_file.resolve())

    print("\n✅ Treino e avaliação dos modelos concluídos com sucesso.")


if __name__ == "__main__":
    main()
