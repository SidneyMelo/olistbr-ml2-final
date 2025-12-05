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
 - Salva heatmap de métricas em results/accuracy_barplot.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

from src.feature_engineering import (
    prepare_supervised_xy,
    build_preprocessor,
)
from src.utils import set_global_seed
from src.utils import save_model
from src.models import (
    split_train_test,
    train_naive_bayes_classifier,
    train_logistic_regression_classifier,
    train_svm_classifier,
    train_random_forest_classifier,
    train_xgboost_classifier,
    evaluate_classifier,
)
try:
    from xgboost import XGBClassifier
    HAS_XGB_IMPORT = True
except ImportError:
    HAS_XGB_IMPORT = False

# Estilo dos gráficos
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (8, 4)


def main() -> None:
    set_global_seed(42)
    DATA_FILE = Path("data/processed/olist_model_dataset.csv")
    RESULTS_DIR = Path("results/classification")
    METRICS_DIR = RESULTS_DIR / "metrics"
    PLOTS_DIR = RESULTS_DIR / "plots"
    MODELS_DIR = RESULTS_DIR / "models"
    # Para validar o fluxo rapidamente, limitar amostra usada no SVM
    SVM_MAX_TRAIN_SAMPLES = 100000
    HPARAM_SEARCH = True
    N_ITER_RF = 10
    N_ITER_XGB = 12

    if not DATA_FILE.exists():
        raise SystemExit(f"❌ ERRO: Arquivo de dados não encontrado: {DATA_FILE.resolve()}.\n"
                         f"Execute antes o 02_preprocessing.py.")

    print("📄 Lendo base de modelagem em:", DATA_FILE.resolve())
    df = pd.read_csv(DATA_FILE)
    before = len(df)
    df = df.dropna()
    if len(df) != before:
        print(f"⚠️ Linhas removidas por NaN: {before - len(df)} (após dropna)")

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

    # ============================================================
    # 1.1) Validação cruzada estratificada para comparação inicial
    # ============================================================
    print("\n🔁 Rodando validação cruzada estratificada (3 folds)...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def run_cross_validation(
        name: str,
        estimator,
        X_data: pd.DataFrame,
        y_data: pd.Series,
        add_stabilizer: bool = False,
        variance_threshold: float | None = None,
        sample_limit: int | None = None,
        n_jobs_cv: int = -1,
    ) -> dict:
        if sample_limit is not None and len(X_data) > sample_limit:
            X_data = X_data.sample(n=sample_limit, random_state=42)
            y_data = y_data.loc[X_data.index]
            print(f"   > {name}: usando amostra de {len(X_data)} linhas para CV (limite {sample_limit})")

        steps = [("preprocessor", clone(preprocessor_template))]
        if add_stabilizer:
            steps.append(
                (
                    "stabilizer",
                    FunctionTransformer(
                        lambda X: np.clip(
                            np.nan_to_num(X, copy=False, posinf=0, neginf=0),
                            -20,
                            20,
                        ),
                        feature_names_out="one-to-one",
                    ),
                )
            )
        if variance_threshold is not None:
            steps.append(
                ("var_thresh", VarianceThreshold(threshold=variance_threshold))
            )
        steps.append(("clf", estimator))

        pipe = Pipeline(steps)
        scores = cross_validate(
            pipe,
            X_data,
            y_data,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs_cv,
            return_train_score=False,
        )
        summary = {
            metric: {
                "mean": float(scores[f"test_{metric}"].mean()),
                "std": float(scores[f"test_{metric}"].std()),
            }
            for metric in scoring.keys()
        }
        print(
            f"   {name}: acc={summary['accuracy']['mean']:.4f} (+/- {summary['accuracy']['std']:.4f}) "
            f"f1_w={summary['f1_weighted']['mean']:.4f} "
            f"prec_w={summary['precision_weighted']['mean']:.4f} "
            f"rec_w={summary['recall_weighted']['mean']:.4f}"
        )
        return summary

    def run_random_search(
        name: str,
        base_estimator,
        param_distributions: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_iter: int = 10,
        scoring: str = "recall_weighted",
    ):
        """
        Executa RandomizedSearchCV com pipeline (preprocessador + estimador) focado em recall weighted.
        """
        pipe = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor_template)),
                ("clf", base_estimator),
            ]
        )
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        search.fit(X_train, y_train)
        print(
            f"   {name} (search): melhor {scoring} CV = {search.best_score_:.4f}\n"
            f"   Params: {search.best_params_}"
        )
        return search.best_estimator_, search.best_params_, search.best_score_

    cv_results = {}
    cv_results["naive_bayes"] = run_cross_validation(
        "Naive Bayes",
        GaussianNB(var_smoothing=1e-4),
        X,
        y,
        add_stabilizer=True,
        variance_threshold=1e-4,
        sample_limit=100000, # limitar amostra para NB
        n_jobs_cv=1,  # serial para evitar explosão numérica/memória no NB
    )
    cv_results["logistic_regression"] = run_cross_validation(
        "Regressão Logística",
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.001,
            max_iter=2000,
            tol=1e-3,
            learning_rate="adaptive",
            eta0=0.01,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        X,
        y,
    )
    cv_results["random_forest"] = run_cross_validation(
        "Random Forest",
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        X,
        y,
    )

    # Para o SVM, amostrar se a base for grande para manter tempo de execução razoável
    X_svm_cv = X
    y_svm_cv = y
    if len(X) > SVM_MAX_TRAIN_SAMPLES:
        X_svm_cv = X.sample(n=SVM_MAX_TRAIN_SAMPLES, random_state=42)
        y_svm_cv = y.loc[X_svm_cv.index]
        print(f"   > Usando amostra de {len(X_svm_cv)} linhas para SVM na CV (limite: {SVM_MAX_TRAIN_SAMPLES})")

    cv_results["svm_linear"] = run_cross_validation(
        "SVM Linear",
        SVC(kernel="linear", C=1.0, gamma="scale", class_weight="balanced"),
        X_svm_cv,
        y_svm_cv,
    )

    cv_results_file = METRICS_DIR / "supervised_cv_results.json"
    with open(cv_results_file, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2, ensure_ascii=False)
    print("💾 Resultados de CV salvos em:", cv_results_file.resolve())

    # Heatmap de CV (médias com anotação de desvio)
    cv_means = []
    cv_stds = []
    for model_name, metrics in cv_results.items():
        row_mean = {}
        row_std = {}
        for metric_name, values in metrics.items():
            row_mean[metric_name] = values.get("mean", np.nan)
            row_std[metric_name] = values.get("std", np.nan)
        cv_means.append(pd.Series(row_mean, name=model_name))
        cv_stds.append(pd.Series(row_std, name=model_name))

    df_cv_mean = pd.DataFrame(cv_means)
    df_cv_std = pd.DataFrame(cv_stds).reindex(df_cv_mean.index)
    rename_map = {
        "accuracy": "Accuracy",
        "precision_weighted": "Precision (weighted)",
        "recall_weighted": "Recall (weighted)",
        "f1_weighted": "F1 (weighted)",
    }
    df_cv_mean = df_cv_mean.rename(columns=rename_map)
    df_cv_std = df_cv_std.rename(columns=rename_map)
    if not df_cv_mean.empty:
        # Ordenar por Accuracy média (melhor para cima)
        df_cv_mean = df_cv_mean.sort_values("Accuracy", ascending=False)
        df_cv_std = df_cv_std.loc[df_cv_mean.index]
        desired_cols = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"]
        existing_cols = [c for c in desired_cols if c in df_cv_mean.columns]
        df_cv_mean = df_cv_mean[existing_cols]
        df_cv_std = df_cv_std[existing_cols]

        annot = pd.DataFrame(index=df_cv_mean.index, columns=df_cv_mean.columns, dtype=object)
        for r in df_cv_mean.index:
            for c in df_cv_mean.columns:
                m = df_cv_mean.loc[r, c]
                s = df_cv_std.loc[r, c]
                annot.loc[r, c] = f"{m:.3f}\n±{s:.3f}"

        fig, ax = plt.subplots(figsize=(8, 0.8 * len(df_cv_mean) + 2))
        sns.heatmap(
            df_cv_mean,
            annot=annot,
            fmt="",
            cmap="YlGnBu",
            linewidths=0.5,
            cbar_kws={"label": "Média (CV)"},
            ax=ax,
        )
        ax.set_title("Validação cruzada (3 folds) — médias e desvios")
        ax.set_xlabel("Métricas")
        ax.set_ylabel("Modelos")
        plt.xticks(rotation=25, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        cv_plot_file = PLOTS_DIR / "cv_metrics_heatmap.png"
        plt.savefig(cv_plot_file, dpi=120)
        plt.close()
        print("📈 Heatmap de CV salvo em:", cv_plot_file.resolve())

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
    best_params_log = {}

    print("\n➡ Treinando Naive Bayes...")
    models["naive_bayes"] = train_naive_bayes_classifier(
        clone(preprocessor_template), tt.X_train, tt.y_train
    )

    print("\n➡ Treinando Regressão Logística...")
    models["logistic_regression"] = train_logistic_regression_classifier(
        clone(preprocessor_template), tt.X_train, tt.y_train, class_weight="balanced"
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
        clone(preprocessor_template), tt_svm, y_svm, kernel="linear", C=1.0, gamma="scale", class_weight="balanced" # kernel="rbf"
    )

    print("\n➡ Treinando Random Forest...")
    if HPARAM_SEARCH:
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        rf_params = {
            "clf__n_estimators": [200, 300, 400],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_leaf": [1, 2, 5],
            "clf__max_features": ["sqrt", "log2", 0.5],
            "clf__class_weight": [None, "balanced", {0: 2, 1: 1}, {0: 3, 1: 1}],
        }
        rf_model, rf_best_params, rf_cv_score = run_random_search(
            "Random Forest",
            rf_base,
            rf_params,
            tt.X_train,
            tt.y_train,
            n_iter=N_ITER_RF,
            scoring="recall_weighted",
        )
        models["random_forest"] = rf_model
        best_params_log["random_forest"] = {
            "best_params": rf_best_params,
            "cv_recall_weighted": rf_cv_score,
        }
    else:
        models["random_forest"] = train_random_forest_classifier(
            clone(preprocessor_template), tt.X_train, tt.y_train, n_estimators=200, random_state=42
        )

    print("\n➡ Tentando treinar XGBoost...")
    xgb_model = None
    if HAS_XGB_IMPORT:
        if HPARAM_SEARCH:
            xgb_base = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=42,
                verbosity=0,
            )
            xgb_params = {
                "clf__n_estimators": [200, 400, 600],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__max_depth": [3, 5, 7],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.7, 1.0],
                "clf__min_child_weight": [1, 3, 5],
                "clf__reg_lambda": [1, 2, 5],
                "clf__scale_pos_weight": [1, 3, 5],
            }
            xgb_model, xgb_best_params, xgb_cv_score = run_random_search(
                "XGBoost",
                xgb_base,
                xgb_params,
                tt.X_train,
                tt.y_train,
                n_iter=N_ITER_XGB,
                scoring="recall_weighted",
            )
            best_params_log["xgboost"] = {
                "best_params": xgb_best_params,
                "cv_recall_weighted": xgb_cv_score,
            }
        else:
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

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    accuracies = []
    best_model_name = None
    best_model_pipe = None
    best_recall_name = None
    best_recall_pipe = None
    best_recall_value = -np.inf
    best_acc = -np.inf

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
        if name in best_params_log:
            results[name]["best_params"] = best_params_log[name]

        accuracies.append((name, acc))
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model_pipe = model
        # Melhor recall da classe 0 (ruim)
        recall_0 = report_dict.get("0", {}).get("recall", -np.inf)
        if recall_0 > best_recall_value:
            best_recall_value = recall_0
            best_recall_name = name
            best_recall_pipe = model

    # ============================================================
    # 4) Salvar resultados em JSON
    # ============================================================
    results_file = METRICS_DIR / "supervised_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n💾 Resultados salvos em:", results_file.resolve())

    # ============================================================
    # 5) Heatmap de métricas no teste
    # ============================================================
    heatmap_rows = []
    for name, metrics in results.items():
        report = metrics["classification_report"]
        weighted = report.get("weighted avg", {})
        heatmap_rows.append(
            {
                "Modelo": name,
                "Accuracy": metrics["accuracy"],
                "Precision (weighted)": weighted.get("precision", np.nan),
                "Recall (weighted)": weighted.get("recall", np.nan),
                "F1 (weighted)": weighted.get("f1-score", np.nan),
            }
        )

    df_heatmap = pd.DataFrame(heatmap_rows).set_index("Modelo")
    df_heatmap = df_heatmap.sort_values("Accuracy", ascending=False)

    # Altura dinâmica conforme quantidade de modelos
    height = max(4.0, 0.6 * len(df_heatmap) + 2)
    fig, ax = plt.subplots(figsize=(9, height))

    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Valor da métrica"},
        ax=ax,
    )

    ax.set_title("Métricas por modelo (teste holdout)")
    ax.set_xlabel("Métricas")
    ax.set_ylabel("Modelos")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    acc_plot_file = PLOTS_DIR / "accuracy_barplot.png"
    plt.savefig(acc_plot_file, dpi=120)
    plt.close()
    print("📈 Heatmap de métricas salvo em:", acc_plot_file.resolve())

    # ============================================================
    # 6) Importância de atributos do melhor modelo
    # ============================================================
    def _get_feature_names(preprocessor):
        try:
            return preprocessor.get_feature_names_out()
        except Exception:
            names = []
            for _, transformer, cols in preprocessor.transformers_:
                if transformer == "drop":
                    continue
                try:
                    if hasattr(transformer, "get_feature_names_out"):
                        part = transformer.get_feature_names_out(cols)
                    elif hasattr(transformer, "named_steps"):
                        # tenta pegar último step com get_feature_names_out
                        last = None
                        for step in reversed(transformer.named_steps.values()):
                            if hasattr(step, "get_feature_names_out"):
                                last = step
                                break
                        if last is not None:
                            part = last.get_feature_names_out(cols)
                        else:
                            part = cols
                    else:
                        part = cols
                except Exception:
                    part = cols
                names.extend([str(c) for c in part])
            return np.array(names)

    def _short_feature_name(name: str) -> str:
        mapping = {
            "product_category_name_english": "cat",
            "product_category_name": "cat",
            "payment_type": "pay",
            "customer_state": "state",
            "delivery_delay_days": "delay_days",
            "estimated_delivery_days": "eta_days",
            "delivery_time_days": "deliv_days",
            "total_items_price": "items_price",
            "total_freight_value": "freight",
            "payment_installments": "installments",
            "payment_value": "pay_value",
        }
        short = name
        for k, v in mapping.items():
            short = short.replace(k, v)
        short = short.replace("cat__", "cat_")
        short = short.replace("num__", "")
        short = short.replace(" ", "_")
        return short

    def plot_feature_importance(model_pipe, model_name: str, out_path: Path, top_n: int = 10) -> None:
        if model_pipe is None:
            print("⚠️ Nenhum modelo disponível para importância de atributos.")
            return
        pre = model_pipe.named_steps.get("preprocessor")
        clf = model_pipe.named_steps.get("clf")
        if pre is None or clf is None:
            print("⚠️ Pipeline sem preprocessor ou clf; não é possível extrair importâncias.")
            return

        feature_names = _get_feature_names(pre)
        importances = None
        if hasattr(clf, "feature_importances_"):
            importances = np.array(clf.feature_importances_)
        elif hasattr(clf, "coef_"):
            coefs = np.array(clf.coef_)
            if coefs.ndim == 1:
                importances = np.abs(coefs)
            else:
                importances = np.mean(np.abs(coefs), axis=0)
        else:
            print(f"⚠️ Modelo {model_name} não suporta importâncias (feature_importances_/coef_).")
            return

        # Ajustar tamanho se houver mismatch
        m = min(len(feature_names), len(importances))
        feature_names = feature_names[:m]
        importances = importances[:m]

        df_imp = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )
        df_imp["feature_short"] = df_imp["feature"].map(_short_feature_name)
        df_imp = df_imp.reset_index(drop=True)

        df_rest = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .iloc[top_n:]
        )
        if not df_rest.empty:
            rest_table = METRICS_DIR / (out_path.stem + "_rest.csv")
            df_rest.to_csv(rest_table, index=False)
            print(f"📄 Importâncias restantes salvas em: {rest_table.resolve()}")

        # Cores contínuas, destacando top 5 com mais opacidade
        cmap = plt.cm.viridis
        norm = plt.Normalize(df_imp["importance"].min(), df_imp["importance"].max())
        base_colors = cmap(norm(df_imp["importance"]))
        colors = []
        for pos, color in enumerate(base_colors):
            rgba = list(color)
            rgba[3] = 0.95 if pos < 5 else 0.75  # alpha
            colors.append(rgba)

        height = max(5, 0.55 * len(df_imp) + 1)
        fig, ax = plt.subplots(figsize=(10, height))
        y_pos = np.arange(len(df_imp))
        bars = ax.barh(
            y=y_pos,
            width=df_imp["importance"],
            color=colors,
            edgecolor="black",
            linewidth=[1.2 if pos < 5 else 0.8 for pos in y_pos],
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_imp["feature_short"])
        ax.invert_yaxis()
        ax.set_xlabel("Importância")
        ax.set_ylabel("Feature")
        ax.set_title(f"Top {top_n} — Importância de atributos ({model_name})")

        max_imp = df_imp["importance"].max()
        pad = 0.02 * max_imp
        ax.set_xlim(0, max_imp * 1.15)

        # Anotar valores ao final das barras
        for bar, value in zip(bars, df_imp["importance"]):
            x = bar.get_width()
            ax.text(
                x + pad,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=9,
                path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
            )

        # Barra de cor contínua coerente
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Importância (escala contínua)")

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"📊 Importância de atributos salva em: {out_path.resolve()}")

    print("\n🎯 Gerando importâncias de atributos do melhor modelo (por acurácia)...")
    imp_path = PLOTS_DIR / "feature_importance_best_model.png"
    plot_feature_importance(best_model_pipe, best_model_name, imp_path)

    # ============================================================
    # 7) Salvar melhores modelos e metadados
    # ============================================================
    if best_model_pipe is not None:
        best_model_file = MODELS_DIR / "best_classification_model.joblib"
        save_model(best_model_pipe, best_model_file)
        meta_file = MODELS_DIR / "best_classification_model.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"best_model": best_model_name, "accuracy": best_acc}, f, indent=2, ensure_ascii=False)
        print(f"💾 Melhor modelo salvo em: {best_model_file.resolve()}")
    else:
        print("⚠️ Nenhum modelo salvo (best_model_pipe=None).")

    if best_recall_pipe is not None:
        best_recall_file = MODELS_DIR / "best_recall_model.joblib"
        save_model(best_recall_pipe, best_recall_file)
        meta_recall_file = MODELS_DIR / "best_recall_model.json"
        with open(meta_recall_file, "w", encoding="utf-8") as f:
            json.dump({"best_model_recall": best_recall_name, "recall_class_0": best_recall_value}, f, indent=2, ensure_ascii=False)
        print(f"💾 Melhor modelo (recall classe 0) salvo em: {best_recall_file.resolve()}")
    else:
        print("⚠️ Nenhum modelo salvo (best_recall_pipe=None).")

    # ============================================================
    # 8) Gerar gráficos para TODOS os modelos
    # ============================================================
    print("\n📊 Gerando gráficos para todos os modelos...")

    for model_name, metrics in results.items():
        acc = metrics["accuracy"]
        
        report_df = pd.DataFrame(metrics["classification_report"]).T
        per_class = report_df.loc[["0", "1"], ["precision", "recall", "f1-score", "support"]]

        cm = np.array(metrics["confusion_matrix"])
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        labels = np.array([["TN", "FP"], ["FN", "TP"]])
        annot_matrix = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot_matrix[i, j] = (
                    f"{labels[i, j]}\n{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)"
                )

        fig, ax = plt.subplots(figsize=(7.5, 5))
        max_count = cm.max() if cm.max() > 0 else 1
        cmap_correct = plt.get_cmap("Greens")
        cmap_error = plt.get_cmap("Reds")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value_norm = cm[i, j] / max_count
                # Evitar cores saturadas: limitar mínimo e máximo
                value_norm = 0.2 + 0.8 * value_norm
                cell_color = cmap_correct(value_norm) if i == j else cmap_error(value_norm)

                ax.add_patch(
                    plt.Rectangle(
                        (j, i),
                        1,
                        1,
                        facecolor=cell_color,
                        edgecolor="white",
                        linewidth=1.2,
                        zorder=1,
                    )
                )

                brightness = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
                text_color = "#000000" if brightness > 0.6 else "#ffffff"
                outline_color = "#000000" if text_color == "#ffffff" else "#ffffff"

                ax.text(
                    j + 0.5,
                    i + 0.5,
                    annot_matrix[i, j],
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                    fontweight="semibold",
                    zorder=3,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground=outline_color)],
                )

        ax.set_xticks(np.arange(cm.shape[1]))
        ax.set_yticks(np.arange(cm.shape[0]))
        ax.set_xticklabels(["Ruim", "Bom"])
        ax.set_yticklabels(["Ruim", "Bom"], rotation=0)
        ax.set_xlim(0, cm.shape[1])
        ax.set_ylim(cm.shape[0], 0)  # manter origem no canto superior esquerdo
        ax.grid(False)

        ax.set_xlabel("Predito")
        ax.set_ylabel("Verdadeiro")
        ax.set_title(
            f"Matriz de confusão - {model_name} (acc={acc:.3f})\n"
            "Valores absolutos e % por linha"
        )
        ax.set_xticklabels(["Ruim", "Bom"])
        ax.set_yticklabels(["Ruim", "Bom"], rotation=0)
        ax.grid(False)

        metrics_text = (
            f"Ruim: P={per_class.loc['0', 'precision']:.3f} "
            f"R={per_class.loc['0', 'recall']:.3f} "
            f"F1={per_class.loc['0', 'f1-score']:.3f} "
            f"Sup={int(per_class.loc['0', 'support'])}\n"
            f"Bom: P={per_class.loc['1', 'precision']:.3f} "
            f"R={per_class.loc['1', 'recall']:.3f} "
            f"F1={per_class.loc['1', 'f1-score']:.3f} "
            f"Sup={int(per_class.loc['1', 'support'])}"
        )
        fig.text(
            1.02,
            0.5,
            metrics_text,
            ha="left",
            va="center",
            fontsize=10,
        )

        cm_plot_file = PLOTS_DIR / f"confusion_matrix_{model_name}.png"
        plt.tight_layout(rect=(0, 0, 0.82, 1))
        plt.savefig(cm_plot_file, dpi=120)
        plt.close()
        print(f"📈 Matriz de confusão salva: {cm_plot_file.name}")

        per_class_file = METRICS_DIR / f"class_report_{model_name}.csv"
        per_class.to_csv(per_class_file, index=True)

        # Renomear índice para o gráfico
        per_class_plot = per_class.copy()
        per_class_plot.index = ["Ruim", "Bom"]
        melted = per_class_plot[["precision", "recall"]].reset_index().melt(
            id_vars="index", value_vars=["precision", "recall"], var_name="metric"
        )
        class_order = sorted(melted["index"].unique())

        fig, ax = plt.subplots(figsize=(8, 3.5))
        bars = sns.barplot(
            data=melted,
            x="value",
            y="index",
            hue="metric",
            order=class_order,
            palette="Set2",
            dodge=True,
            ax=ax,
        )

        ax.set_xlabel("Valor")
        ax.set_ylabel("Classe")
        ax.set_title(f"Precision vs Recall por classe - {model_name}")
        ax.set_xlim(0, 1)

        # Rótulos nas barras
        for patch in bars.patches:
            width = patch.get_width()
            # Suprimir rótulos muito pequenos (evita 0.000 poluindo o gráfico)
            if width >= 0.01:
                ax.text(
                    width + 0.01,
                    patch.get_y() + patch.get_height() / 2,
                    f"{width:.3f}",
                    va="center",
                    ha="left",
                    fontsize=9,
                )

        # Grid leve apenas no eixo x
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        ax.grid(axis="y", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            title="Métrica",
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
        )
        plt.tight_layout()
        pr_plot_file = RESULTS_DIR / f"precision_recall_{model_name}.png"
        plt.savefig(pr_plot_file, dpi=120)
        plt.close()
        print(f"📈 Gráfico P/R salvo: {pr_plot_file.name}")

    print("\n✅ Treino e avaliação dos modelos concluídos com sucesso.")


if __name__ == "__main__":
    main()
