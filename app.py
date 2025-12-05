#!/usr/bin/env python
"""
Streamlit app para apresentação dos resultados e realização de predições.

Requisitos:
- Rodar previamente os scripts de treino para gerar resultados e salvar modelos:
  * python 03_supervised_models.py (salva best_classification_model.joblib/.json)
  * python 05_regression_review_score.py (salva best_regression_model.joblib/.json)
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

from src.feature_engineering import get_supervised_feature_lists
from src.utils import load_model


PROJECT_TITLE = "Determinantes de Satisfação no E-Commerce Brasileiro: Uma Análise sobre o Dataset Olist"

st.set_page_config(
    page_title=PROJECT_TITLE,
    layout="wide",
)

RESULTS_DIR = Path("results/classification")
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"
REG_RESULTS_DIR = Path("results/regression")
REG_METRICS_DIR = REG_RESULTS_DIR / "metrics"
REG_PLOTS_DIR = REG_RESULTS_DIR / "plots"
REG_MODELS_DIR = REG_RESULTS_DIR / "models"
DATA_FILE = Path("data/processed/olist_model_dataset.csv")
BEST_CLS_MODEL = MODELS_DIR / "best_classification_model.joblib"
BEST_CLS_META = MODELS_DIR / "best_classification_model.json"
BEST_RECALL_MODEL = MODELS_DIR / "best_recall_model.joblib"
BEST_RECALL_META = MODELS_DIR / "best_recall_model.json"
BEST_REG_MODEL = REG_MODELS_DIR / "best_regression_model.joblib"
BEST_REG_META = REG_MODELS_DIR / "best_regression_model.json"


@st.cache_data
def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_dataset(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_joblib(path: Path):
    if not path.exists():
        return None
    return load_model(path)


def get_feature_defaults(df: pd.DataFrame, numeric_features, categorical_features):
    defaults = {}
    q1 = df[numeric_features].quantile(0.01)
    q99 = df[numeric_features].quantile(0.99)
    med = df[numeric_features].median()
    for col in numeric_features:
        defaults[col] = {
            "min": float(q1.get(col, 0.0)),
            "max": float(q99.get(col, med.get(col, 0.0) + 1)),
            "value": float(med.get(col, 0.0)),
        }
    for col in categorical_features:
        defaults[col] = list(df[col].dropna().unique())[:50]
    return defaults


def classification_section():
    st.header("Classificação — review_binary")
    cls_results = load_json(METRICS_DIR / "supervised_results.json")
    cv_results = load_json(METRICS_DIR / "supervised_cv_results.json")
    if not cls_results:
        st.warning("Resultados de classificação não encontrados. Rode `python 03_supervised_models.py`.")
        return

    cols = st.columns(3)
    # melhor modelo por accuracy holdout
    best_name, best_metrics = max(cls_results.items(), key=lambda kv: kv[1]["accuracy"])
    cols[0].metric("Melhor modelo (acc)", best_name, f"{best_metrics['accuracy']:.3f}")
    cols[1].metric("Acc (std CV RF)" if cv_results else "Acc melhor",
                   f"{cv_results.get('random_forest', {}).get('accuracy', {}).get('mean', np.nan):.3f}" if cv_results else f"{best_metrics['accuracy']:.3f}")
    cols[2].metric("F1 weighted (melhor)", f"{best_metrics['classification_report']['weighted avg']['f1-score']:.3f}")

    # melhor modelo para recall da classe 0
    recall_vals = {}
    for name, metrics in cls_results.items():
        recall_vals[name] = metrics["classification_report"].get("0", {}).get("recall", np.nan)
    best_recall_name = max(recall_vals.items(), key=lambda kv: kv[1])[0]
    best_recall_val = recall_vals[best_recall_name]
    st.info(f"Melhor recall (classe 0 - avaliações ruins): {best_recall_name} com recall={best_recall_val:.3f}")

    with st.expander("Validação cruzada (3 folds)", expanded=False):
        cv_heatmap = PLOTS_DIR / "cv_metrics_heatmap.png"
        if cv_heatmap.exists():
            st.image(str(cv_heatmap), caption="CV (média ± desvio) para accuracy, f1, roc_auc")
        else:
            st.info("Heatmap de CV não encontrado. Rode `python 03_supervised_models.py` para gerá-lo.")

    with st.expander("Heatmap de métricas (teste holdout)", expanded=True):
        heatmap_path = PLOTS_DIR / "accuracy_barplot.png"
        if heatmap_path.exists():
            st.image(str(heatmap_path), caption="Accuracy / Precision / Recall / F1 (weighted)")
        else:
            st.info("Arquivo accuracy_barplot.png não encontrado.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Matriz de confusão")
        model_choices = list(cls_results.keys())
        chosen = st.selectbox("Modelo", model_choices, index=model_choices.index(best_name))
        cm_path = PLOTS_DIR / f"confusion_matrix_{chosen}.png"
        if cm_path.exists():
            st.image(str(cm_path), caption=f"Matriz de confusão — {chosen}")
        else:
            st.info(f"Arquivo {cm_path.name} não encontrado.")
    with c2:
        st.subheader("Importância de atributos (melhor modelo)")
        imp_path = PLOTS_DIR / "feature_importance_best_model.png"
        if imp_path.exists():
            st.image(str(imp_path))
            rest_path = METRICS_DIR / "feature_importance_best_model_rest.csv"
            if rest_path.exists():
                st.download_button("Baixar importâncias restantes (CSV)", rest_path.read_bytes(), file_name=rest_path.name)
        else:
            st.info("Arquivo feature_importance_best_model.png não encontrado.")


def regression_section():
    st.header("Regressão — review_score contínuo (1–5)")
    reg_results = load_json(REG_METRICS_DIR / "regression_results.json")
    if not reg_results:
        st.warning("Resultados de regressão não encontrados. Rode `python 05_regression_review_score.py`.")
        return
    best_name, best_metrics = min(reg_results.items(), key=lambda kv: kv[1]["rmse"])
    cols = st.columns(4)
    cols[0].metric("Melhor modelo (RMSE)", best_name, f"{best_metrics['rmse']:.3f}")
    cols[1].metric("MAE (melhor)", f"{best_metrics.get('mae', float('nan')):.3f}")
    cols[2].metric("R² (melhor)", f"{best_metrics['r2']:.3f}")
    cols[3].metric("Escala", "review_score 1–5")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Paridade y_true vs y_pred")
        for name in ["parity_plot_linear.png", "parity_plot_ridge.png"]:
            path = REG_PLOTS_DIR / name
            if path.exists():
                st.image(str(path), caption=name)
    with c2:
        st.subheader("Resíduos")
        for name in ["residuals_linear.png", "residuals_ridge.png"]:
            path = REG_PLOTS_DIR / name
            if path.exists():
                st.image(str(path), caption=name)


def prediction_section():
    st.header("Predição interativa")
    numeric_features, categorical_features = get_supervised_feature_lists()

    df_full = load_dataset(DATA_FILE)
    defaults = None
    if df_full is not None:
        defaults = get_feature_defaults(df_full, numeric_features, categorical_features)

    cls_model_acc = load_joblib(BEST_CLS_MODEL)
    cls_model_recall = load_joblib(BEST_RECALL_MODEL)
    reg_model = load_joblib(BEST_REG_MODEL)

    task = st.radio("Tarefa", ["Classificação (review_binary)", "Regressão (review_score)"], key="task_choice")

    if task.startswith("Classificação") and cls_model_acc is None:
        st.warning("Modelo de classificação não encontrado. Rode `python 03_supervised_models.py` para salvar o best_classification_model.joblib.")
        return
    if task.startswith("Regressão") and reg_model is None:
        st.warning("Modelo de regressão não encontrado. Rode `python 05_regression_review_score.py` para salvar o best_regression_model.joblib.")
        return

    model_options = []
    if cls_model_acc is not None:
        model_options.append("Melhor acurácia")
    if cls_model_recall is not None:
        model_options.append("Melhor recall (classe 0)")
    selected_model = None
    if task.startswith("Classificação"):
        selected_model = st.radio(
            "Modelo para predição",
            options=model_options,
            index=0,
            key="model_choice_class",
        )

    st.subheader("Preencha as features")
    form_cols = st.columns(2)
    input_data = {}
    for col_name in numeric_features:
        params = defaults.get(col_name, {}) if defaults else {}
        input_data[col_name] = form_cols[0 if len(input_data) % 2 == 0 else 1].number_input(
            col_name,
            value=params.get("value", 0.0),
            min_value=params.get("min", 0.0),
            max_value=params.get("max", params.get("value", 0.0) * 5 + 1),
        )
    for col_name in categorical_features:
        options = defaults.get(col_name, []) if defaults else []
        input_data[col_name] = form_cols[0 if len(input_data) % 2 == 0 else 1].selectbox(
            col_name,
            options if options else ["desconhecido"],
            index=0,
        )

    if st.button("Prever"):
        df_input = pd.DataFrame([input_data])
        if task.startswith("Classificação"):
            model_choice = selected_model or "Melhor acurácia"
            model_to_use = cls_model_acc if model_choice == "Melhor acurácia" else cls_model_recall
            y_pred = model_to_use.predict(df_input)[0]
            proba = None
            if hasattr(model_to_use, "predict_proba"):
                proba = model_to_use.predict_proba(df_input)[0]
            st.session_state["last_pred"] = {
                "type": "class",
                "model_choice": model_choice,
                "y_pred": int(y_pred),
                "proba": proba.tolist() if proba is not None else None,
            }
        else:
            y_pred = reg_model.predict(df_input)[0]
            y_pred_clamped = float(np.clip(y_pred, 1, 5))
            st.session_state["last_pred"] = {
                "type": "reg",
                "y_pred": float(y_pred),
                "y_pred_clamped": y_pred_clamped,
            }

    # Exibir última predição, mesmo após rerun
    if "last_pred" in st.session_state:
        lp = st.session_state["last_pred"]
        if lp["type"] == "class":
            label = "Bom (1)" if lp["y_pred"] == 1 else "Ruim (0)"
            proba = lp.get("proba")
            cols = st.columns([1.2, 1])
            with cols[0]:
                icon = "👍" if lp["y_pred"] == 1 else "⚠️"
                color = "#16a34a" if lp["y_pred"] == 1 else "#b91c1c"
                subtext = "Alta probabilidade de satisfação" if lp["y_pred"] == 1 else "Risco de insatisfação"
                st.markdown(
                    f"""
                    <div style="border:1px solid {color};border-radius:10px;padding:14px;background:rgba(15,23,42,0.6);">
                      <div style="font-size:24px;">{icon} Predição ({lp['model_choice']})</div>
                      <div style="font-size:20px;font-weight:700;color:{color};margin-top:4px;">{label}</div>
                      <div style="color:#e2e8f0;margin-top:2px;">{subtext}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if proba is not None:
                    st.markdown(
                        f"""
                        <div style="margin-top:10px;padding:10px;border-radius:8px;border:1px solid #2b8a3e;">
                            <div style="font-weight:600;margin-bottom:6px;color:#e2e8f0;">Probabilidades</div>
                            <div style="display:flex;gap:10px;flex-wrap:wrap;">
                                <div style="background:#0f172a;color:#e9ecef;padding:8px 12px;border-radius:6px;">P(classe 0): <b>{proba[0]:.3f}</b></div>
                                <div style="background:#0f172a;color:#e9ecef;padding:8px 12px;border-radius:6px;">P(classe 1): <b>{proba[1]:.3f}</b></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            with cols[1]:
                st.caption("Modelo de melhor acurácia ou melhor recall de classe 0, conforme seleção.")
        else:
            rounded = int(round(lp["y_pred_clamped"]))
            stars_full = rounded
            stars = "★" * stars_full + "☆" * (5 - stars_full)
            show_clamp = abs(lp["y_pred"] - lp["y_pred_clamped"]) > 1e-6
            clamp_text = (
                f"<div style=\"color:#94a3b8;font-size:12px;margin-top:4px;\">Clamp 1–5 aplicado: {lp['y_pred_clamped']:.2f}</div>"
                if show_clamp
                else ""
            )
            st.markdown(
                f"""
                <div style="border:1px solid #0ea5e9;border-radius:10px;padding:14px;background:rgba(14,165,233,0.08);">
                  <div style="font-size:18px;font-weight:700;color:#0ea5e9;margin-bottom:6px;">Predição (review_score contínuo)</div>
                  <div style="font-size:22px;color:#e2e8f0;">{rounded:.2f} / 5.00</div>
                  <div style="font-size:20px;color:#22c55e;margin-top:4px;">{stars}</div>
                  {clamp_text}
                </div>
                """,
                unsafe_allow_html=True,
            )


def main():
    st.title(PROJECT_TITLE)
    page = st.sidebar.radio(
        "Seções",
        ["Visão geral", "Perguntas-chave", "Classificação", "Regressão", "Predição"],
    )

    if page == "Visão geral":
        st.write("Apresentação dos resultados de modelagem para satisfação do cliente (Olist).")
        st.write("- Execute `03_supervised_models.py` e `05_regression_review_score.py` para gerar resultados e modelos.")
        st.write("- Use as abas para navegar entre classificação, regressão e predição interativa.")
    elif page == "Perguntas-chave":
        st.header("Perguntas-chave respondidas")
        st.markdown(
            "- **Características ligadas a notas altas/baixas**: ver importâncias (Classificação) e paridade/resíduos (Regressão).\n"
            "- **Tempo de entrega afeta a nota?** `figures/delivery_delay_by_review_score.png` e `figures/analysis/review_score_by_delay.png`.\n"
            "- **Categorias com maior insatisfação?** `figures/analysis/categories_most_negative.png` (+ CSV top).\n"
            "- **Regiões mais negativas/positivas?** `figures/analysis/states_most_negative.png` (+ CSV top)."
        )
        cols = st.columns(2)
        with cols[0]:
            st.subheader("Atraso vs Nota")
            for path in ["figures/delivery_delay_by_review_score.png", "figures/analysis/review_score_by_delay.png"]:
                p = Path(path)
                if p.exists():
                    st.image(str(p), caption=p.name)
        with cols[1]:
            st.subheader("Categorias e Estados com mais notas ruins")
            for path in ["figures/analysis/categories_most_negative.png", "figures/analysis/states_most_negative.png"]:
                p = Path(path)
                if p.exists():
                    st.image(str(p), caption=p.name)
    elif page == "Classificação":
        classification_section()
    elif page == "Regressão":
        regression_section()
    else:
        prediction_section()


if __name__ == "__main__":
    main()
