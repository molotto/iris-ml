# Classificação de Plantas com o Dataset Iris e Análise Estatística

Este projeto avalia e compara o desempenho de quatro algoritmos de classificação supervisionada no clássico dataset **Iris**, utilizando métricas de desempenho, visualizações gráficas e testes estatísticos formais.

## Objetivo

Avaliar os modelos:

* **K-Nearest Neighbors (KNN)**
* **Decision Tree**
* **Random Forest**
* **Support Vector Machine (SVM)**

...com base em:

* Acurácia
* Recall
* F1-Score

Além disso, aplicamos **ANOVA** e, quando necessário, o **teste de Tukey** para avaliar diferenças estatisticamente significativas entre os modelos.

---

## Etapas do Código

### 1. **Carregamento e Preprocessamento**

* Dataset `iris` do `scikit-learn`.
* Adição de ruído aleatório de 5% aos atributos para simular variabilidade real.

### 2. **Modelos Utilizados**

* KNN e SVM com `StandardScaler`
* Decision Tree
* Random Forest

### 3. **Validação Cruzada**

* `StratifiedKFold` com 5 dobras.
* Coleta de métricas por **fold** para permitir teste estatístico real.
* Geração de:

  * Acurácia Média e Desvio
  * Recall Médio
  * F1-Score Médio
* Salvos:

  * `metricas_iris_modelos.csv`
  * `metricas_folds.csv`

### 4. **Visualizações**

* **Matriz de confusão** por modelo (`matriz_confusao_<modelo>.png`)
* **Gráfico de barras** de acurácia com erro padrão
* **Boxplots** de acurácia, recall e f1-score por modelo

### 5. **Testes Estatísticos**

* **ANOVA** para `acuracia`, `recall` e `f1`
* Se `p < 0.05`: aplica **Tukey HSD**
* Resultados ajudam a concluir se há diferença significativa entre os modelos

---

## Estrutura de Saída

```
resultados/
├── grafico_acuracia_modelos.png
├── matriz_confusao_KNN.png
├── matriz_confusao_SVM.png
├── ...
├── metricas_cross_val/
│   ├── metricas_iris_modelos.csv
│   └── metricas_folds.csv
└── analise_estatistica/
    ├── boxplot_acuracia.png
    ├── boxplot_recall.png
    └── boxplot_f1.png
```

---

## Conclusão

Este pipeline oferece uma abordagem didática e estatisticamente rigorosa para comparar modelos clássicos de classificação. Ele pode ser facilmente adaptado para outros conjuntos de dados e algoritmos, e oferece uma base robusta para trabalhos acadêmicos, projetos de pesquisa e aprendizado de técnicas de avaliação em machine learning.

---

## Requisitos e Instalação

### Pré-requisitos

- Python 3.13 ou superior
- [UV Package Manager](https://github.com/astral-sh/uv) - Um gerenciador de pacotes Python moderno e rápido

### Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/molotto/iris-ml.git
   cd iris-ml
   ```

2. Crie o ambiente virtual e instale as dependências:
   ```bash
   uv sync
   ```

3. Ative o ambiente virtual:

   No Windows:
   ```bash
   .venv\Scripts\activate
   ```

   No Unix/macOS:
   ```bash
   source .venv/bin/activate
   ```

4. Execute o programa:
   ```bash
   python -m main
   ```

