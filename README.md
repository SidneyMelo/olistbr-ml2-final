# Olist - Modelagem de Reviews

Projeto de ciencia de dados aplicado ao dataset publico da Olist (Kaggle). O foco e explicar a satisfacao do cliente (review_score) por meio de:
- Analise exploratoria com graficos.
- Pipeline de pre-processamento e criacao de targets.
- Modelos supervisionados para classificar reviews boas vs ruins.
- Experimento de clustering em pedidos/clientes.
- Regressao para prever a nota continua.

## Dataset
- Nome: Olist Brazilian E-Commerce Public Dataset
- Origem: [Kaggle - Brazilian E-Commerce Public Dataset](https://www.kaggle.com/code/thuandao/brazilian-e-commerce-analysis)

## Problema e perguntas de pesquisa
Objetivo: analisar os fatores que impactam a satisfacao do cliente nas compras realizadas em e-commerce, utilizando as avaliacoes registradas e os dados dos pedidos do dataset Olist.

Perguntas que o projeto busca responder:
- Quais caracteristicas dos pedidos estao relacionadas a avaliacoes altas ou baixas (review_score)?
- O tempo entre compra e entrega (order_purchase_timestamp, order_delivered_customer_date) afeta a nota do review?
- Existem categorias de produtos (product_category_name) com maior insatisfacao dos clientes?
- Clientes de determinadas regioes (customer_state, customer_city) avaliam de maneira mais positiva ou negativa?

## Estrutura do repositorio
- `data/`: CSVs brutos do Kaggle Olist e saidas processadas em `data/processed/`.
- `figures/`: graficos da analise exploratoria e cortes por categoria/estado.
- `results/`: artefatos dos modelos (JSON, CSV e PNG).
- `src/`: modulos reutilizaveis (pre-processamento, feature engineering, modelos, clustering, utils).
- `01_data_overview.py ... 05_regression_review_score.py`: scripts de ponta a ponta.

## Requisitos
- Python >= 3.10
- Instale dependencias: `pip install -r requirements.txt`
- Todos os CSVs do dataset Olist devem estar em `data/` com os nomes originais.

## Como rodar
1) Analise exploratoria  
`python 01_data_overview.py`  
Saidas em `figures/` (distribuicao de review_score, pedidos por mes, atraso vs nota) e tabelas auxiliares em `figures/analysis/`.

2) Pre-processamento e base de modelagem  
`python 02_preprocessing.py`  
Gera `data/processed/olist_model_dataset.csv` (remove cancelados, adiciona features de tempo, agregacoes por pedido e targets: review_binary, review_positive, review_negative).

3) Classificacao supervisionada (target = review_binary: 1 >= 4, 0 <= 2, neutros dropados)  
`python 03_supervised_models.py`  
Treina Naive Bayes, Regressao Logistica, SVM linear, Random Forest e XGBoost (se instalado). Produz:
- `results/supervised_results.json`
- `results/accuracy_barplot.png`
- `results/confusion_matrix_<melhor>.png`, `class_report_<melhor>.csv`, `precision_recall_<melhor>.png`  
Resultado atual: acc ~0.89 (Random Forest / XGBoost), com melhor recall para a classe positiva.

4) Clustering (K-Means, n_clusters=3)  
`python 04_clustering.py`  
Usa features numericas (preco, frete, itens, tempo/atraso de entrega). Saidas:
- `data/processed/olist_model_dataset_with_clusters.csv` (coluna cluster)
- `results/clustering/cluster_summary.csv`, `review_score_by_cluster.png`, `boxplots_features_by_cluster.png`

5) Regressao da nota continua  
`python 05_regression_review_score.py`  
Treina Regressao Linear e Ridge usando as mesmas features supervisionadas. Saidas:
- `results/regression/regression_results.json` (RMSE ~1.20, R2 ~0.19)
- Graficos de paridade e residuos em `results/regression/*.png`

## Insights e artefatos prontos
- Categorias com maior percentual de notas <= 2 (`figures/analysis/categories_most_negative_top.csv`): destaque para fashion_roupa_masculina, moveis_escritorio, telefonia_fixa, audio.
- Estados com medias mais baixas (`figures/analysis/states_most_negative_top.csv`): AL, MA, SE, PA aparecem com maior % de reviews ruins.
- Impacto de atraso na entrega: graficos em `figures/analysis/review_score_by_delay.png` e `figures/delivery_delay_by_review_score.png`.
- Volume historico de pedidos em `figures/orders_per_month.png`.

## Notas
- Listas de features numericas/categoricas ficam em `src/feature_engineering.py`; ajuste aqui se quiser experimentar variaveis novas.
- Clustering usa padronizacao interna (StandardScaler) antes do K-Means.
- XGBoost e opcional; se a lib nao estiver instalada o script segue sem ele.
