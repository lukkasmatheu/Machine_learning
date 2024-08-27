import os
import pandas as pd
import numpy as np
from math import log2
import matplotlib.pyplot as plt
import networkx as nx
import json
i=0
classes = [1, 2, 3, 4]
class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

def calcular_entropia_total(data_set):
    total_row = len(data_set)
    total_entropia = 0
    for c in classes: 
        total_classe_count = len(data_set[data_set[5] == c])
        if total_classe_count != 0:
            probabilidade_classe = total_classe_count / total_row
            if(probabilidade_classe == 1):
                print(f"[{c}]-{probabilidade_classe} ")
            total_classe_entropia = -probabilidade_classe * log2(probabilidade_classe)
            total_entropia += total_classe_entropia
    return total_entropia

def calcular_ganho(df, coluna_index, bins, labels):
    df['binned'] = pd.cut(df.iloc[:, coluna_index], bins=bins, labels=labels, include_lowest=True, ordered=False)
    total = len(df)
    ganho_total = 0
    for label in labels:
        subset = df[df['binned'] == label]
        probabilidade_subset = len(subset) / total
        if probabilidade_subset > 0:  # Evita divisão por zero
            subset_entropy = calcular_entropia_total(subset)
            ganho_total += probabilidade_subset * subset_entropy
    return calcular_entropia_total(df) - ganho_total

def encontrar_melhor_atributo(df, atributos):
    max_gain = -1
    melhor_atributo = None
    for atributo, params in atributos.items():
        ganho = calcular_ganho(df, params['index'], params['bins'], params['labels'])
        if ganho > max_gain:
            max_gain = ganho
            melhor_atributo = atributo
    return melhor_atributo, max_gain

def id3(df, atributos):
    classes_unicas = df.iloc[:, -1].unique()
    
    if len(classes_unicas) == 1:
        return classes_unicas[0]
    
    if len(atributos) == 0:
        return df.iloc[:, -1].mode()[0]
    
    melhor_atributo, ganho = encontrar_melhor_atributo(df, atributos)
    tree = {melhor_atributo: {}}
    
    params = atributos[melhor_atributo]  # Manter os parâmetros para o atributo
    df['binned'] = pd.cut(df.iloc[:, params['index']], bins=params['bins'], labels=params['labels'], include_lowest=True,ordered=False)
    
    for label in params['labels']:
        subset = df[df['binned'] == label].drop(columns=['binned'])  # Remover a coluna 'binned' ao criar subárvores
        if len(subset) == 0:
            # Se não houver dados, retorna a classe mais comum
            classe_comum = df.iloc[:, -1].mode()[0]
            tree[melhor_atributo][label] = classe_comum
        else:
            # Recursivamente construir subárvores
            tree[melhor_atributo][label] = id3(subset, {k: v for k, v in atributos.items() if k != melhor_atributo})
    
    return tree

def build_graph(data, graph=None, parent=None, level=0):
    global i
    if graph is None:
        graph = nx.DiGraph()

    for key, value in data.items():
        # Create an internal identifier but keep the key as the display label
        node_id = f"{key}_{i}"  # Internal identifier
        graph.add_node(node_id, label=key, subset=level)  # Store the display label in the node's data
        if parent:
            graph.add_edge(parent, node_id)
        i += 1
        if isinstance(value, dict):
            build_graph(value, graph, node_id, level + 1)
        else:
            translation = {
                1: 'Critico',
                2: 'Instavel',
                3: 'P-Estavel',
                4: 'Estavel'
            }
            leaf_key = f'{key} ({translation[value]})'  # Create a unique identifier for leaf nodes
            leaf_id = f"{leaf_key}_{i}"  # Internal identifier for the leaf node
            graph.add_node(leaf_id, label=leaf_key, subset=level + 1)
            graph.add_edge(node_id, leaf_id)

    return graph

def main():

    current_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(current_path + '/DataSet.csv', header=None)

    # Certifique-se de que os bins refletem corretamente a distribuição dos seus dados
    atributos = {
        'qPa': {'index': 1, 'bins': [-10,-7, -3, 3, 7, 10], 'labels': ['extremo','baixa', 'normal', 'alta','extremo']},
        'Pulso': {'index': 2, 'bins': [0, 40, 60, 110, 160, 200], 'labels': ['extremo','baixa', 'normal', 'alta','extremo']},
        'Respiração': {'index': 3, 'bins': [0, 7, 12, 19, 22], 'labels': ['extremo','baixa', 'normal', 'alta']}
    }

    entropia_total = calcular_entropia_total(df)
    print(f"Entropia Total do Dataset: {entropia_total}")

    arvore = id3(df, atributos)
    print("Árvore de Decisão ID3:")
    print(arvore)
    

    with open('arvore_decisao.json', 'w') as json_file:
        json.dump(arvore, json_file, indent=4,cls=NpEncoder)

    # Construindo o grafo
    G = build_graph(arvore)

    # Desenhando o grafo com layout personalizado
 # Desenhando o grafo com layout multipartite
    pos = nx.multipartite_layout(G, subset_key="subset",align='horizontal')  # Usando o layout multipartite
    plt.figure(figsize=(14, 10))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=200, edge_color='k', font_size=6, font_weight='bold', arrows=False)
    plt.title("Grafo Gerado a Partir do JSON")
    plt.show()

if __name__ == "__main__":
    main()