# Ensaio de Machine Learning - Classificação, Regressão e Clusterização

![status](https://img.shields.io/badge/status-conclu%C3%ADdo-brightgreen)

## 📓 Notebooks do Projeto

| Notebook | Algoritmos | Tipo de Problema |
|---|---|---|
| `proj_classificacao_ensaio.ipynb` | KNN, Decision Tree Classifier, Random Forest, Logistic Regression | Classificação ||
| `proj_regression_ensaio.ipynb` | Decision Tree Regressor, Polinomial Regression, Linear Regression, Random Forest Regressor, Lasso, Ridge, Elastic Net (Linear e Polinomial) | Regressão |
| `proj_clusterizacao_ensaio.ipynb` | K-Means, Affinity Propagation | Clusterização |

---

## 1. Visão Geral do Projeto

Este é um projeto de **ensaio de Machine Learning**, cujo objetivo é comparar de forma sistemática o desempenho de diferentes algoritmos de **Classificação**, **Regressão** e **Clusterização** sobre conjuntos de dados distintos. A solução foi desenvolvida utilizando **Python, scikit-learn, Pandas e Matplotlib**, em ambiente Jupyter Notebook.

O ensaio foi estruturado em três frentes principais:

1. Ensaio de Classificação
2. Ensaio de Regressão
3. Ensaio de Clusterização

O objetivo é consolidar, para cada algoritmo, as métricas de desempenho sobre os conjuntos de **treino**, **validação** e **teste**, permitindo identificar qual técnica se ajusta melhor a cada tipo de problema.

---

## 2. Contexto e Objetivo do Ensaio

Ao construir um modelo de Machine Learning, uma das principais dúvidas é: **qual algoritmo utilizar?**

Não existe um algoritmo universalmente melhor — o desempenho depende da natureza dos dados, da quantidade de variáveis, da presença de ruído e da relação entre as features e a variável resposta.

Diante desse cenário, o projeto propõe a construção de um ensaio comparativo capaz de responder perguntas como:

* Qual algoritmo apresenta a melhor acurácia em um problema de classificação?
* Como o ajuste de hiperparâmetros impacta o desempenho de cada modelo?
* Modelos mais simples (ex: Regressão Linear) performam melhor ou pior que modelos mais complexos (ex: Random Forest)?
* A regularização (Lasso, Ridge, Elastic Net) melhora a generalização dos modelos de regressão?
* É possível encontrar agrupamentos naturais nos dados sem a necessidade de rótulos?

O resultado é um conjunto de notebooks e tabelas comparativas que documentam o comportamento de cada algoritmo frente aos mesmos dados.

---

## 3. Origem e Tratamento dos Dados

Foram utilizados três conjuntos de dados distintos, um para cada tipo de ensaio, todos previamente divididos em **treino**, **validação** e **teste**:

* **Dataset de Classificação:** dados de satisfação de clientes de companhia aérea, já normalizados e codificados (`X_training.csv`, `X_validation.csv`, `X_test.csv` e respectivos `y`).
* **Dataset de Regressão:** dados de popularidade de músicas, com variáveis de áudio já padronizadas (`X_training (1).csv`, `X_validation (1).csv`, `X_test (1).csv` e respectivos `y`).
* **Dataset de Clusterização:** dados de composição química de vinhos, já normalizados (`X_dataset.csv`), sem variável resposta (aprendizado não supervisionado).

### Principais Tratamentos

* Dados já divididos previamente em treino, validação e teste.
* Variáveis numéricas já normalizadas/padronizadas.
* Variáveis categóricas já codificadas (one-hot encoding).
* Ausência de valores nulos nos conjuntos utilizados.

---

## 4. Estrutura do Projeto

O ensaio foi desenvolvido em notebooks Jupyter independentes, organizados por tipo de problema e por algoritmo, seguindo sempre a mesma metodologia de avaliação.

### Módulos Disponíveis

* Ensaio de Classificação
* Ensaio de Regressão
* Ensaio de Clusterização

---

## 5. Ensaio de Classificação

O ensaio de classificação avalia o desempenho de diferentes algoritmos na tarefa de prever a satisfação de clientes.

### 5.1 Algoritmos Avaliados

* KNN
* Decision Tree Classifier
* Random Forest Classifier
* Logistic Regression

### 5.2 Metodologia

Para cada algoritmo, o ensaio segue as seguintes etapas:

1. Treinamento e avaliação sobre os dados de **treino**.
2. Busca do melhor hiperparâmetro (ex: `n_neighbors`, `max_depth`, `n_estimators`, `C`) utilizando os dados de **validação**.
3. Treinamento do modelo final com o melhor hiperparâmetro, unindo os dados de **treino + validação**.
4. Avaliação final sobre os dados de **teste**, utilizando as métricas Accuracy, Precision, Recall e F1-Score.

### 5.3 Resultados sobre os dados de teste

| Algoritmo | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| KNN | 0.688 | 0.648 | 0.635 | 0.642 |
| Decision Tree | 0.956 | 0.957 | 0.943 | 0.950 |
| Random Forest | 0.965 | 0.974 | 0.945 | 0.959 |
| Logistic Regression | 0.871 | 0.866 | 0.834 | 0.850 |

O **Random Forest** apresentou o melhor desempenho geral, seguido de perto pela **Decision Tree**. O **KNN** foi o algoritmo com menor desempenho entre os avaliados.

---

## 6. Ensaio de Regressão

O ensaio de regressão avalia o desempenho de diferentes algoritmos na tarefa de prever a popularidade de músicas.

### 6.1 Algoritmos Avaliados

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Polinomial Regression
* Linear Regression (Lasso, Ridge, Elastic Net)
* Polinomial Regression (Lasso, Ridge, Elastic Net)

### 6.2 Metodologia

A metodologia segue o mesmo padrão do ensaio de classificação, adaptada para métricas de regressão:

1. Treinamento e avaliação sobre os dados de **treino**.
2. Busca do melhor hiperparâmetro (ex: `max_depth`, `n_estimators`, `alpha`, grau do polinômio) utilizando os dados de **validação**.
3. Treinamento do modelo final com o melhor hiperparâmetro, unindo os dados de **treino + validação**.
4. Avaliação final sobre os dados de **teste**, utilizando as métricas R², MSE, RMSE, MAE e MAPE.

### 6.3 Resultados sobre os dados de teste

| Algoritmo | R² | MSE | RMSE | MAE | MAPE |
|---|---|---|---|---|---|
| Linear Regression | 0.051 | 461.99 | 21.49 | 17.14 | 8.531 |
| Decision Tree Regressor | 0.065 | 455.23 | 21.34 | 17.03 | 8.129 |
| Random Forest Regressor | 0.404 | 290.20 | 17.04 | 12.23 | 6.338 |
| Polinomial Regression | 0.091 | 442.64 | 21.04 | 16.74 | 8.277 |
| Linear Regression Lasso | 0.008 | 483.10 | 21.98 | 17.47 | 8.753 |
| Linear Regression Ridge | 0.051 | 461.99 | 21.49 | 17.14 | 8.532 |
| Linear Regression Elastic Net | 0.008 | 483.06 | 21.98 | 17.47 | 8.746 |
| Polinomial Regression Lasso | 0.009 | 482.62 | 21.97 | 17.46 | 8.756 |
| Polinomial Regression Ridge | 0.090 | 442.97 | 21.05 | 16.74 | 8.309 |
| Polinomial Regression Elastic Net | 0.011 | 481.54 | 21.94 | 17.43 | 8.754 |

O **Random Forest Regressor** foi, de longe, o algoritmo com melhor desempenho, reduzindo significativamente o erro em relação aos demais modelos. Os modelos lineares regularizados (Lasso, Ridge e Elastic Net) apresentaram desempenho muito próximo ao modelo linear sem regularização, indicando pouco ganho de generalização nesse conjunto de dados.

---

## 7. Ensaio de Clusterização

O ensaio de clusterização avalia a capacidade de diferentes algoritmos de agrupar amostras de vinhos com base em suas características químicas, sem o uso de rótulos.

### 7.1 Algoritmos Avaliados

* K-Means
* Affinity Propagation

### 7.2 Metodologia

1. Para o **K-Means**, foi realizada uma busca pelo melhor número de clusters (`k`), testando valores de 2 a 20 e avaliando o **Average Silhouette Score** para cada configuração.
2. Para o **Affinity Propagation**, foi realizada uma busca pelo melhor parâmetro `damping`, avaliando o impacto no número de clusters formados e no Average Silhouette Score.
3. Em ambos os casos, foi selecionado o modelo com o maior Average Silhouette Score.

### 7.3 Resultados

| Algoritmo | Número de Clusters | Average Silhouette Score |
|---|---|---|
| K-Means | 3 | 0.233 |
| Affinity Propagation | 18 | 0.171 |

O **K-Means**, com apenas 3 clusters, apresentou uma separação mais coesa entre os grupos do que o **Affinity Propagation**, que tende a formar um número maior de clusters e, consequentemente, uma separação menos definida nesse conjunto de dados.

---

## 8. Tecnologias Utilizadas

### Linguagem

* Python

### Machine Learning

* scikit-learn

### Manipulação de Dados

* Pandas
* NumPy

### Visualização de Dados

* Matplotlib

### Ambiente

* Jupyter Notebook
* Git
* GitHub

---

## 9. Considerações Finais

O Ensaio de Machine Learning demonstra, de forma prática, como diferentes algoritmos de Classificação, Regressão e Clusterização se comportam frente aos mesmos princípios de avaliação (treino, validação e teste), permitindo uma comparação justa e replicável entre técnicas.

Os resultados reforçam conceitos importantes da área, como a superioridade de modelos baseados em ensemble (Random Forest) em cenários com relações não-lineares entre as variáveis, e a importância da busca de hiperparâmetros para a generalização dos modelos.

Como evoluções futuras, podem ser adicionadas:

* Novos algoritmos de Classificação, Regressão e Clusterização (ex: SVM, Gradient Boosting, DBSCAN).
* Validação cruzada (cross-validation) em substituição à divisão fixa treino/validação/teste.
* Engenharia de features para melhoria dos modelos de regressão.
* Otimização de hiperparâmetros via Grid Search ou Random Search.
* Redução de dimensionalidade (PCA) previamente ao ensaio de clusterização.
* Documentação comparativa entre todos os ensaios em um dashboard interativo.
