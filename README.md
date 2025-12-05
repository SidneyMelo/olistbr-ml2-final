# 🛒 Análise de Satisfação no E-Commerce Brasileiro

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Projeto de ciência de dados que investiga os determinantes da satisfação do cliente no e-commerce brasileiro, utilizando o dataset público da Olist. O objetivo é identificar padrões e construir modelos preditivos para entender e antecipar a experiência do cliente.

## 📊 Sobre o Dataset

**Dataset:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/code/thuandao/brazilian-e-commerce-analysis)

O dataset contém informações reais de pedidos realizados entre 2016 e 2018 na Olist, marketplace brasileiro que conecta pequenos e médios negócios a diversos canais de venda. Inclui dados sobre:
- 🛍️ Pedidos e seus status
- ⭐ Avaliações dos clientes (review_score)
- 📦 Produtos e categorias
- 🚚 Informações de entrega
- 💰 Valores e formas de pagamento
- 📍 Localização de clientes e vendedores

---

## 🎯 Objetivos e Perguntas de Pesquisa

### Objetivo Principal
Analisar os fatores que impactam a satisfação do cliente em compras online, utilizando técnicas de machine learning e análise exploratória de dados.

### Perguntas de Pesquisa
1. **Quais características** dos pedidos estão mais relacionadas a avaliações altas ou baixas?
2. **O tempo de entrega** (prazo estimado vs. real) afeta significativamente a nota do review?
3. **Existem categorias de produtos** com maior tendência a gerar insatisfação?
4. **A localização geográfica** do cliente influencia na avaliação dos pedidos?
5. **É possível prever** a satisfação do cliente antes da avaliação ser registrada?

---

## 🏗️ Estrutura do Projeto

```
olist-analysis/
├── data/
│   ├── *.csv                          # CSVs brutos do Kaggle
│   └── processed/                     # Dados processados
├── figures/
│   └── analysis/                      # Gráficos e tabelas da EDA
├── results/
│   ├── classification/                # Resultados dos modelos de classificação
│   ├── clustering/                    # Análise de clusters
│   └── regression/                    # Modelos de regressão
├── src/
│   ├── preprocessing.py               # Pipeline de pré-processamento
│   ├── feature_engineering.py         # Criação de features
│   ├── models.py                      # Definição dos modelos
│   ├── clustering.py                  # Algoritmos de clustering
│   └── utils.py                       # Funções auxiliares
├── 01_data_overview.py                # Script de análise exploratória
├── 02_preprocessing.py                # Script de preparação dos dados
├── 03_supervised_models.py            # Treinamento de modelos supervisionados
├── 04_clustering.py                   # Análise de agrupamentos
├── 05_regression_review_score.py      # Modelos de regressão
├── app.py                             # Dashboard Streamlit
└── requirements.txt                   # Dependências do projeto
```

---

## 🚀 Como Usar

### Pré-requisitos

- Python 3.10 ou superior
- Todos os arquivos CSV do dataset Olist baixados do Kaggle

### Instalação

```bash
# Clone o repositório
git clone <url-do-repositorio>
cd olist-analysis

# Crie um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt
```

### Pipeline de Execução

#### 1️⃣ Análise Exploratória de Dados (EDA)
```bash
python 01_data_overview.py
```
**Saídas:**
- Distribuição de review_score
- Evolução temporal de pedidos
- Relação entre atraso na entrega e avaliação
- Análise por categoria e estado

📁 Resultados em: `figures/` e `figures/analysis/`

---

#### 2️⃣ Pré-processamento
```bash
python 02_preprocessing.py
```
**Processamento:**
- Remove pedidos cancelados
- Calcula features temporais (tempo de entrega, atrasos)
- Cria agregações por pedido
- Gera variáveis target:
  - `review_binary`: Bom (≥4) vs. Ruim (≤2)
  - `review_positive`: Review ≥4
  - `review_negative`: Review ≤2

📁 Saída: `data/processed/olist_model_dataset.csv`

---

#### 3️⃣ Modelos de Classificação
```bash
python 03_supervised_models.py
```
**Modelos Treinados:**
- Naive Bayes
- Regressão Logística
- SVM Linear
- Random Forest
- XGBoost (opcional)

**Avaliação:**
- Métricas holdout e cross-validation (5-fold)
- Matriz de confusão
- Importância de features
- Comparação de performance

📁 Resultados em: `results/classification/`

**Performance Atual:**
- ✅ Acurácia: ~89% (Random Forest/XGBoost)
- ✅ Recall classe negativa: Melhor em SVM/LogReg

---

#### 4️⃣ Análise de Clusters
```bash
python 04_clustering.py
```
**Método:** K-Means (k=3) com StandardScaler

**Features utilizadas:**
- Preço total
- Valor do frete
- Número de itens
- Tempo de entrega
- Dias de atraso

**Clusters Identificados:**
1. **Baixo Valor** - Entregas rápidas, 1 produto
2. **Alto Valor** - 1 produto, entregas adiantadas
3. **Multi-produtos** - Volume médio

📁 Saída: `data/processed/olist_model_dataset_with_clusters.csv`

---

#### 5️⃣ Regressão para Previsão de Nota
```bash
python 05_regression_review_score.py
```
**Modelos:**
- Regressão Linear
- Ridge Regression

**Performance:**
- RMSE: ~1.23
- R²: ~0.15
- MAE: ~0.95

📁 Resultados em: `results/regression/`

---

#### 6️⃣ Dashboard Interativo
```bash
streamlit run app.py
```

**Funcionalidades:**
- 📊 Visualização de métricas dos modelos
- 🎯 Predição interativa de satisfação
- 📈 Análise exploratória interativa
- 🔍 Comparação entre modelos
- 🎨 Gráficos de importância de features

Acesse em: `http://localhost:8501`

---

## 💡 Principais Insights

### 📉 Categorias com Maior Insatisfação
- Fashion (roupa masculina)
- Móveis de escritório
- Telefonia fixa
- Equipamentos de áudio

📁 Detalhes: `figures/analysis/categories_most_negative_top.csv`

### 🗺️ Estados com Menores Avaliações
- Alagoas (AL)
- Maranhão (MA)
- Sergipe (SE)
- Pará (PA)

📁 Detalhes: `figures/analysis/states_most_negative_top.csv`

### ⏰ Impacto do Atraso na Entrega
Pedidos atrasados apresentam correlação forte com avaliações negativas. Visualizações demonstram queda acentuada na satisfação conforme aumenta o atraso.

📁 Gráficos: `figures/analysis/review_score_by_delay.png`

---

## 🛠️ Customização

### Adicionar Novas Features
Edite `src/feature_engineering.py` para incluir novas variáveis numéricas ou categóricas.

### Ajustar Modelos
Modifique hiperparâmetros em `src/models.py` ou adicione novos algoritmos.

### Configurar Clustering
Altere o número de clusters ou método em `04_clustering.py`.

---

## 📋 Requisitos Técnicos

### Dependências Principais
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- streamlit >= 1.20.0
- xgboost >= 1.7.0 (opcional)

📄 Lista completa: `requirements.txt`

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir novas features
- Melhorar a documentação
- Submeter pull requests

---

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

---

## 📧 Contato

Para dúvidas ou sugestões, abra uma issue no repositório.

---

## 🙏 Agradecimentos

- **Olist** por disponibilizar o dataset público
- **Kaggle** por hospedar e facilitar o acesso aos dados
- Comunidade open-source pelas ferramentas utilizadas

---