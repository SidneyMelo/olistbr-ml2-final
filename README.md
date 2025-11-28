# Olist - Modelagem de Reviews

Projeto de ciência de dados aplicado ao dataset público da Olist (Kaggle). O foco é explicar a satisfação do cliente (review_score) por meio de:
- Análise exploratória com gráficos.
- Pipeline de pré-processamento e criação de targets.
- Modelos supervisionados para classificar reviews boas vs ruins.
- Experimento de clustering em pedidos/clientes.
- Regressão para prever a nota contínua.

## Estrutura do repositório
- `data/`: CSVs brutos do Kaggle Olist e saídas processadas em `data/processed/`.
- `figures/`: gráficos da análise exploratória e cortes por categoria/estado.
- `results/`: artefatos dos modelos (JSON, CSV e PNG).
- `src/`: módulos reutilizáveis (pré-processamento, feature engineering, modelos, clustering, utils).
- `01_data_overview.py ... 05_regression_review_score.py`: scripts de ponta a ponta.

## Requisitos
- Python >= 3.10
- Instale dependências: `pip install -r requirements.txt`
- Todos os CSVs do dataset Olist devem estar em `data/` com os nomes originais.

## Como rodar
1) Análise exploratória  
`python 01_data_overview.py`  
Saídas em `figures/` (distribuição de review_score, pedidos por mês, atraso vs nota) e tabelas auxiliares em `figures/analysis/`.

2) Pré-processamento e base de modelagem  
`python 02_preprocessing.py`  
Gera `data/processed/olist_model_dataset.csv` (remove cancelados, adiciona features de tempo, agregações por pedido e targets: review_binary, review_positive, review_negative).

3) Classificação supervisionada (target = review_binary: 1 >= 4, 0 <= 2, neutros dropados)  
`python 03_supervised_models.py`  
Treina Naive Bayes, Regressão Logística, SVM linear, Random Forest e XGBoost (se instalado). Produz:
- `results/supervised_results.json`
- `results/accuracy_barplot.png`
- `results/confusion_matrix_<melhor>.png`, `class_report_<melhor>.csv`, `precision_recall_<melhor>.png`  
Resultado atual: acc ~0.89 (Random Forest / XGBoost), com melhor recall para a classe positiva.

4) Clustering (K-Means, n_clusters=3)  
`python 04_clustering.py`  
Usa features numéricas (preço, frete, itens, tempo/atraso de entrega). Saídas:
- `data/processed/olist_model_dataset_with_clusters.csv` (coluna cluster)
- `results/clustering/cluster_summary.csv`, `review_score_by_cluster.png`, `boxplots_features_by_cluster.png`

5) Regressão da nota contínua  
`python 05_regression_review_score.py`  
Treina Regressão Linear e Ridge usando as mesmas features supervisionadas. Saídas:
- `results/regression/regression_results.json` (RMSE ~1.20, R² ~0.19)
- Gráficos de paridade e resíduos em `results/regression/*.png`

## Insights e artefatos prontos
- Categorias com maior percentual de notas <= 2 (`figures/analysis/categories_most_negative_top.csv`): destaque para fashion_roupa_masculina, moveis_escritorio, telefonia_fixa, audio.
- Estados com médias mais baixas (`figures/analysis/states_most_negative_top.csv`): AL, MA, SE, PA aparecem com maior % de reviews ruins.
- Impacto de atraso na entrega: gráficos em `figures/analysis/review_score_by_delay.png` e `figures/delivery_delay_by_review_score.png`.
- Volume histórico de pedidos em `figures/orders_per_month.png`.

## Notas
- Listas de features numéricas/categóricas ficam em `src/feature_engineering.py`; ajuste aqui se quiser experimentar variáveis novas.
- Clustering usa padronização interna (StandardScaler) antes do K-Means.
- XGBoost é opcional; se a lib não estiver instalada o script segue sem ele.
