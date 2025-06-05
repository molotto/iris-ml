from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Garantir que a pasta 'resultados' existe
os.makedirs("resultados/metricas_cross_val", exist_ok=True)
os.makedirs("resultados/analise_estatistica", exist_ok=True)

# 1. Carregar os dados
iris = load_iris()
X = iris.data
y = iris.target

# 2. Adicionar ruído aleatório (5% de desvio)
np.random.seed(42)
noise = np.random.normal(loc=0.0, scale=0.05, size=X.shape)
X_noisy = X + noise

# 3. Modelos com padronização (onde necessário)
modelos = {
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier()),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': make_pipeline(StandardScaler(), SVC())
}

# 4. Estratégia de validação cruzada (5 dobras)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 5. Avaliar cada modelo
resultados = {
    'Modelo': [],
    'Acurácia Média': [],
    'Acurácia Desvio': [],
    'Recall Média': [],
    'F1-Score Média': []
}

folds_expandido = []

for nome, modelo in modelos.items():
    y_pred = cross_val_predict(modelo, X_noisy, y, cv=kf)
    acc_scores = cross_val_score(modelo, X_noisy, y, cv=kf, scoring='accuracy')
    recall_scores = cross_val_score(modelo, X_noisy, y, cv=kf, scoring='recall_macro')
    f1_scores = cross_val_score(modelo, X_noisy, y, cv=kf, scoring='f1_macro')

    resultados['Modelo'].append(nome)
    resultados['Acurácia Média'].append(np.mean(acc_scores))
    resultados['Acurácia Desvio'].append(np.std(acc_scores))
    resultados['Recall Média'].append(np.mean(recall_scores))
    resultados['F1-Score Média'].append(np.mean(f1_scores))

    # Salvar resultados por fold para análise estatística
    for i in range(5):
        folds_expandido.append({'Modelo': nome, 'Fold': i+1, 'Metrica': 'acuracia', 'Valor': acc_scores[i]})
        folds_expandido.append({'Modelo': nome, 'Fold': i+1, 'Metrica': 'recall', 'Valor': recall_scores[i]})
        folds_expandido.append({'Modelo': nome, 'Fold': i+1, 'Metrica': 'f1', 'Valor': f1_scores[i]})

    print(f"\n=== {nome} ===")
    print(f"Acurácia média: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"Recall média: {np.mean(recall_scores):.4f}")
    print(f"F1-score média: {np.mean(f1_scores):.4f}")

    # Matriz de confusão
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(cmap='Blues')
    plt.title(f"Matriz de Confusão - {nome}")
    plt.tight_layout()
    plt.savefig(f"resultados/matriz_confusao_{nome}.png")
    plt.show()

    # Relatório por classe
    print("Relatório de Classificação por Classe:")
    print(classification_report(y, y_pred, target_names=iris.target_names))

# 6. Criar DataFrame e exibir
df_resultados = pd.DataFrame(resultados)
print("\nResumo das métricas:")
print(df_resultados)

# 7. Corrigir nomes de colunas
df_resultados.columns = [col.strip() for col in df_resultados.columns]

# 8. Gráfico com barras de erro da acurácia + rótulo com valores
plt.figure(figsize=(10, 6))
barras = plt.bar(df_resultados["Modelo"], df_resultados["Acurácia Média"], yerr=df_resultados["Acurácia Desvio"], capsize=6)
plt.title("Validação Cruzada com Ruído - Dataset Iris")
plt.ylabel("Acurácia")
plt.ylim(0, 1.1)
plt.grid(axis="y")

for bar, media, desvio in zip(barras, df_resultados["Acurácia Média"], df_resultados["Acurácia Desvio"]):
    altura = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, altura + 0.02, f"{media:.2f}\n±{desvio:.2f}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("resultados/grafico_acuracia_modelos.png")
plt.show()

# 9. Exportar métricas gerais e por fold
df_resultados.to_csv("resultados/metricas_cross_val/metricas_iris_modelos.csv", index=False)
df_folds = pd.DataFrame(folds_expandido)
df_folds.to_csv("resultados/metricas_cross_val/metricas_folds.csv", index=False)

# 10. Testes estatísticos (ANOVA e Tukey) para cada métrica
print("\n\n=== Testes Estatísticos ===")
for metrica in ['acuracia', 'recall', 'f1']:
    print(f"\n--- ANOVA para {metrica.upper()} ---")
    df_metrica = df_folds[df_folds['Metrica'] == metrica]
    modelos = df_metrica['Modelo'].unique()
    grupos = [df_metrica[df_metrica['Modelo'] == m]['Valor'] for m in modelos]
    f_stat, p_valor = f_oneway(*grupos)
    print(f"Estatística F: {f_stat:.4f}")
    print(f"p-valor: {p_valor:.4f}")

    if p_valor < 0.05:
        print(f"\n--- Teste de Tukey para {metrica.upper()} ---")
        tukey = pairwise_tukeyhsd(endog=df_metrica['Valor'], groups=df_metrica['Modelo'], alpha=0.05)
        print(tukey)

    # Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_metrica, x='Modelo', y='Valor')
    plt.title(f"Boxplot - {metrica.upper()} por Modelo")
    plt.ylabel(metrica.capitalize())
    plt.xlabel("Modelo")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"resultados/analise_estatistica/boxplot_{metrica}.png")
    plt.show()
