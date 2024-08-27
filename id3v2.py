import os
import pandas as pd
import numpy as np
from math import log2

classes = [1, 2, 3, 4]

# Função para calcular a entropia total
def calcular_entropia_total(data_set):
    total_row = len(data_set)
    total_entropia = 0
    for c in classes: 
        total_classe_count = len(data_set[data_set[5] == c])
        if total_classe_count != 0:
            probabilidade_classe = total_classe_count / total_row
            total_classe_entropia = -probabilidade_classe * log2(probabilidade_classe)
            total_entropia += total_classe_entropia
    return total_entropia

# Função para calcular o ganho de informação
def calcular_ganho(df, coluna_index, bins, labels):
    df['binned'] = pd.cut(df.iloc[:, coluna_index], bins=bins, labels=labels)
    total = len(df)
    ganho_total = 0
    for label in labels:
        subset = df[df['binned'] == label]
        probabilidade_subset = len(subset) / total
        subset_entropy = calcular_entropia_total(subset)
        ganho_total += probabilidade_subset * subset_entropy
    return calcular_entropia_total(df) - ganho_total

# Função para encontrar o melhor atributo
def encontrar_melhor_atributo(df, atributos):
    max_gain = -1
    melhor_atributo = None
    for atributo, params in atributos.items():
        ganho = calcular_ganho(df, params['index'], params['bins'], params['labels'])
        if ganho > max_gain:
            max_gain = ganho
            melhor_atributo = atributo
    return melhor_atributo, max_gain

# Função recursiva para construir a árvore ID3
def id3(df, atributos, profundidade=0):
    classes_unicas = df.iloc[:, -1].unique()
    
    # Caso base: se todas as instâncias tiverem a mesma classe, retornar essa classe
    if len(classes_unicas) == 1:
        return classes_unicas[0]
    
    # Caso base: se não houver mais atributos para dividir, retornar a classe mais comum
    if len(atributos) == 0:
        return df.iloc[:, -1].mode()[0]
    
    # Encontrar o melhor atributo para dividir
    melhor_atributo, ganho = encontrar_melhor_atributo(df, atributos)
    
    # Criar o nó da árvore para o melhor atributo
    tree = {melhor_atributo: {}}
    
    # Discretizar o melhor atributo e remover ele da lista de atributos
    params = atributos.pop(melhor_atributo)
    df['binned'] = pd.cut(df.iloc[:, params['index']], bins=params['bins'], labels=params['labels'])
    
    # Criar subárvores para cada valor possível do melhor atributo
    for label in params['labels']:
        subset = df[df['binned'] == label]
        if len(subset) == 0:
            tree[melhor_atributo][label] = df.iloc[:, -1].mode()[0]
        else:
            tree[melhor_atributo][label] = id3(subset, atributos.copy(), profundidade + 1)
    
    return tree

# Função principal
def main():
    current_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(current_path + '/DataSet.csv', header=None)

    # Definir os intervalos e labels para cada atributo
    atributos = {
        'Pressão': {'index': 1, 'bins': [-10, -3, 3, 10], 'labels': ['baixa', 'normal', 'alto']},
        'Pulso': {'index': 2, 'bins': [0, 60, 110, 200], 'labels': ['baixa', 'normal', 'alto']},
        'Respiração': {'index': 3, 'bins': [0, 7, 14, 22], 'labels': ['baixa', 'normal', 'alto']}
    }

    # Calcular a entropia total do dataset
    entropia_total = calcular_entropia_total(df)
    print(f"Entropia Total do Dataset: {entropia_total}")

    # Construir a árvore de decisão usando ID3
    arvore = id3(df, atributos)
    print("Árvore de Decisão ID3:")
    print(arvore)

if __name__ == "__main__":
    main()
