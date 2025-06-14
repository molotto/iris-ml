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

def analisar_modelos_iris(nivel_ruido=0.00):
    # Garantir que a pasta 'resultados' existe
    os.makedirs("resultados/metricas_cross_val", exist_ok=True)
    os.makedirs("resultados/analise_estatistica", exist_ok=True)
    os.makedirs("resultados/comparacao_ruido", exist_ok=True)

    # 1. Carregar os dados
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2. Adicionar ruído aleatório
    np.random.seed(42)
    noise = np.random.normal(loc=0.0, scale=nivel_ruido, size=X.shape)
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
        plt.title(f"Matriz de Confusão - {nome} (Ruído: {nivel_ruido})")
        plt.tight_layout()
        plt.savefig(f"resultados/matriz_confusao_{nome}_ruido_{nivel_ruido}.png")
        plt.close()

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
    plt.title(f"Validação Cruzada com Ruído {nivel_ruido} - Dataset Iris")
    plt.ylabel("Acurácia")
    plt.ylim(0, 1.1)
    plt.grid(axis="y")

    for bar, media, desvio in zip(barras, df_resultados["Acurácia Média"], df_resultados["Acurácia Desvio"]):
        altura = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, altura + 0.02, f"{media:.2f}\n±{desvio:.2f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"resultados/grafico_acuracia_modelos_ruido_{nivel_ruido}.png")
    plt.close()

    # 9. Exportar métricas gerais e por fold
    df_resultados.to_csv(f"resultados/metricas_cross_val/metricas_iris_modelos_ruido_{nivel_ruido}.csv", index=False)
    df_folds = pd.DataFrame(folds_expandido)
    df_folds.to_csv(f"resultados/metricas_cross_val/metricas_folds_ruido_{nivel_ruido}.csv", index=False)

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
        plt.title(f"Boxplot - {metrica.upper()} por Modelo (Ruído: {nivel_ruido})")
        plt.ylabel(metrica.capitalize())
        plt.xlabel("Modelo")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f"resultados/analise_estatistica/boxplot_{metrica}_ruido_{nivel_ruido}.png")
        plt.close()

    return df_resultados

def plot_comparacao_ruido(resultados_por_ruido):
    """
    Plota gráficos comparativos de como cada modelo se comporta com diferentes níveis de ruído
    """
    # Preparar dados para o gráfico
    modelos = list(resultados_por_ruido[0]['Modelo'])
    ruidos = [f"{r:.2f}" for r in resultados_por_ruido.keys()]
    
    # Criar gráfico para acurácia
    plt.figure(figsize=(12, 6))
    for modelo in modelos:
        acuracias = [resultados_por_ruido[ruido][resultados_por_ruido[ruido]['Modelo'] == modelo]['Acurácia Média'].values[0] 
                    for ruido in resultados_por_ruido.keys()]
        plt.plot(ruidos, acuracias, marker='o', label=modelo)
    
    plt.title('Comparação da Acurácia dos Modelos em Diferentes Níveis de Ruído')
    plt.xlabel('Nível de Ruído')
    plt.ylabel('Acurácia Média')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('resultados/comparacao_ruido/comparacao_acuracia_ruido.png')
    plt.close()

    # Criar gráfico para F1-Score
    plt.figure(figsize=(12, 6))
    for modelo in modelos:
        f1_scores = [resultados_por_ruido[ruido][resultados_por_ruido[ruido]['Modelo'] == modelo]['F1-Score Média'].values[0] 
                    for ruido in resultados_por_ruido.keys()]
        plt.plot(ruidos, f1_scores, marker='o', label=modelo)
    
    plt.title('Comparação do F1-Score dos Modelos em Diferentes Níveis de Ruído')
    plt.xlabel('Nível de Ruído')
    plt.ylabel('F1-Score Média')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('resultados/comparacao_ruido/comparacao_f1_ruido.png')
    plt.close()

def analise_estatistica_ruido(resultados_por_ruido):
    """
    Realiza análise estatística comparativa entre os modelos em diferentes níveis de ruído
    """
    print("\n\n=== Análise Estatística Comparativa ===")
    
    # Criar DataFrame com todos os resultados dos folds
    dados_completos = []
    for nivel_ruido, df in resultados_por_ruido.items():
        # Ler o arquivo de folds correspondente
        df_folds = pd.read_csv(f"resultados/metricas_cross_val/metricas_folds_ruido_{nivel_ruido}.csv")
        
        # Adicionar nível de ruído
        df_folds['Nivel_Ruido'] = nivel_ruido
        dados_completos.append(df_folds)
    
    df_completo = pd.concat(dados_completos, ignore_index=True)
    
    # Análise para cada nível de ruído
    for nivel in resultados_por_ruido.keys():
        print(f"\n--- Análise para nível de ruído {nivel:.2f} ---")
        df_nivel = df_completo[df_completo['Nivel_Ruido'] == nivel]
        
        # Análise para cada métrica
        for metrica in ['acuracia', 'recall', 'f1']:
            print(f"\nANOVA para {metrica.upper()}:")
            df_metrica = df_nivel[df_nivel['Metrica'] == metrica]
            
            # ANOVA
            grupos = [df_metrica[df_metrica['Modelo'] == modelo]['Valor'] 
                     for modelo in df_metrica['Modelo'].unique()]
            f_stat, p_valor = f_oneway(*grupos)
            print(f"Estatística F: {f_stat:.4f}")
            print(f"p-valor: {p_valor:.4f}")
            
            if p_valor < 0.05:
                print(f"\nTeste de Tukey para {metrica.upper()}:")
                tukey = pairwise_tukeyhsd(endog=df_metrica['Valor'], 
                                        groups=df_metrica['Modelo'], 
                                        alpha=0.05)
                print(tukey)
    
    # Análise da degradação com o aumento do ruído
    print("\n--- Análise da Degradação com Aumento do Ruído ---")
    for modelo in df_completo['Modelo'].unique():
        print(f"\nModelo: {modelo}")
        
        for metrica in ['acuracia', 'recall', 'f1']:
            df_modelo = df_completo[(df_completo['Modelo'] == modelo) & 
                                  (df_completo['Metrica'] == metrica)]
            
            # Calcular média por nível de ruído
            medias = df_modelo.groupby('Nivel_Ruido')['Valor'].mean()
            
            # Correlação
            correlacao = np.corrcoef(medias.index, medias.values)[0,1]
            print(f"\n{metrica.upper()}:")
            print(f"Correlação com Ruído: {correlacao:.4f}")
            
            # Taxa de degradação
            degradacao = (medias.iloc[0] - medias.iloc[-1]) / medias.iloc[0] * 100
            print(f"Degradação: {degradacao:.2f}%")
            
            # Análise de tendência
            if correlacao < -0.7:
                print("Forte tendência negativa com aumento do ruído")
            elif correlacao < -0.3:
                print("Tendência negativa moderada com aumento do ruído")
            elif correlacao > 0.7:
                print("Forte tendência positiva com aumento do ruído")
            elif correlacao > 0.3:
                print("Tendência positiva moderada com aumento do ruído")
            else:
                print("Pouca ou nenhuma tendência com aumento do ruído")

# Exemplo de uso com diferentes níveis de ruído
if __name__ == "__main__":
    # Você pode testar diferentes níveis de ruído
    niveis_ruido = [0.00, 0.10, 0.20, 0.30, 0.40]
    
    # Dicionário para armazenar resultados de cada nível de ruído
    resultados_por_ruido = {}
    
    for nivel in niveis_ruido:
        print(f"\n\n=== Testando com nível de ruído: {nivel} ===")
        resultados_por_ruido[nivel] = analisar_modelos_iris(nivel_ruido=nivel)
    
    # Gerar gráficos comparativos
    plot_comparacao_ruido(resultados_por_ruido)
    
    # Realizar análise estatística
    analise_estatistica_ruido(resultados_por_ruido)
    
    # Análise final
    print("\n\n=== Análise Final ===")
    print("Esta análise simula diferentes níveis de qualidade de imagem que podem ser encontrados em fotos de celular.")
    print("Os níveis de ruído representam:")
    print("0.00 - Imagem perfeita (referência)")
    print("0.10 - Imagem de boa qualidade")
    print("0.20 - Imagem de qualidade média")
    print("0.30 - Imagem de qualidade baixa")
    print("0.40 - Imagem extremamente ruidosa")
