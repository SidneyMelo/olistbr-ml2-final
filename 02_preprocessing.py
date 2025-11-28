#!/usr/bin/env python
"""
02_preprocessing.py

Script para pré-processar o dataset Olist e gerar uma base consolidada
para modelagem (classificação, regressão, SVM, clustering, etc.).

- Lê sempre da pasta:       data/
- Salva sempre em:          data/processed/olist_model_dataset.csv
- Remove pedidos cancelados
"""

from pathlib import Path
import pandas as pd

from src.preprocessing import preprocess_base
from src.feature_engineering import add_targets, get_supervised_feature_lists
from src.utils import set_global_seed


# =====================================================================
#   Funções auxiliares
# =====================================================================

def summarize_model_dataset(df: pd.DataFrame) -> None:
    """Imprime um resumo rápido da base de modelagem."""
    print("\n" + "=" * 80)
    print("RESUMO DA BASE DE MODELAGEM")
    print("=" * 80)
    print(f"Formato final: {df.shape[0]} linhas x {df.shape[1]} colunas\n")

    print("Distribuição das notas originais:")
    print(df["review_score"].value_counts(dropna=False).sort_index())
    print()

    if "review_positive" in df.columns:
        print("Distribuição review_positive (1 = nota >= 4):")
        print(df["review_positive"].value_counts(dropna=False))
        print()

    if "review_negative" in df.columns:
        print("Distribuição review_negative (1 = nota <= 2):")
        print(df["review_negative"].value_counts(dropna=False))
        print()

    if "review_binary" in df.columns:
        print("Distribuição review_binary (0 ruim / 1 bom / NaN neutro):")
        print(df["review_binary"].value_counts(dropna=False))
        print()

    cols_preview = [
        "order_id",
        "review_score",
        "review_binary",
        "delivery_time_days",
        "delivery_delay_days",
        "total_items_price",
        "total_freight_value",
        "payment_type",
        "payment_value",
        "product_category_name",
        "customer_state",
        "customer_city",
    ]
    cols_preview = [c for c in cols_preview if c in df.columns]

    print("\nExemplo de linhas:")
    print(df[cols_preview].head())


# =====================================================================
#                          SCRIPT PRINCIPAL
# =====================================================================

def main() -> None:
    set_global_seed(42)
    # Pastas fixas
    DATA_DIR = Path("data")
    OUT_FILE = Path("data/processed/olist_model_dataset.csv")

    # Verificações
    if not DATA_DIR.exists():
        raise SystemExit(f"❌ ERRO: Pasta de dados não encontrada: {DATA_DIR.resolve()}")

    print("📁 Lendo dados da pasta:", DATA_DIR.resolve())
    print("📄 Saída será salva em :", OUT_FILE.resolve())
    print("📌 Removendo pedidos cancelados...")

    # 1) Pré-processar base consolidada
    df_base = preprocess_base(
        base_path=DATA_DIR,
        drop_canceled=True,   # sempre remove cancelados
    )

    print("\n✔ Base consolidada carregada.")
    print(f"Formato após preprocess_base: {df_base.shape[0]} linhas x {df_base.shape[1]} colunas")

    # 2) Adicionar targets
    df_model = add_targets(df_base)

    # 3) Verificar se há features faltantes (informativo)
    numeric_features, categorical_features = get_supervised_feature_lists()

    missing_numeric = [c for c in numeric_features if c not in df_model.columns]
    missing_categorical = [c for c in categorical_features if c not in df_model.columns]

    if missing_numeric or missing_categorical:
        print("\n⚠ AVISO: Algumas features esperadas não constam na base final:")
        if missing_numeric:
            print(" - Numéricas faltando:", missing_numeric)
        if missing_categorical:
            print(" - Categóricas faltando:", missing_categorical)
        print("Isso não impede os modelos, mas você pode ajustar as listas em feature_engineering.py.\n")

    # 4) Resumo no terminal
    summarize_model_dataset(df_model)

    # 5) Salvar dataset final
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_model.to_csv(OUT_FILE, index=False)

    print("\n💾 Dataset de modelagem salvo com sucesso em:")
    print("   ", OUT_FILE.resolve())
    print("\n🎉 Pré-processamento concluído com sucesso!")


if __name__ == "__main__":
    main()
