import os
import pandas as pd
import numpy as np
from math import log2
import matplotlib.pyplot as plt
import networkx as nx
import json
import plotly.graph_objs as go
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

def binarizar_dados(df, atributos):
    for atributo, params in atributos.items():
        df[atributo] = pd.cut(df.iloc[:, params['index']], bins=params['bins'], labels=params['labels'], ordered=False)
    return df

def separar_treino_teste(df, tamanho_teste):
    # Seleciona aleatoriamente 'tamanho_teste' amostras do dataset
    df_teste = df.sample(n=tamanho_teste, random_state=42)

    # Remove essas amostras do dataset original para criar o conjunto de treinamento
    df_treino = df.drop(df_teste.index)

    return df_treino, df_teste


# Função para realizar uma previsão usando a árvore de decisão
def prever(arvore, row):
    if not isinstance(arvore, dict):
        return arvore
    
    atributo = next(iter(arvore))
    valor = row[atributo]
    if valor in arvore[atributo]:
        return prever(arvore[atributo][valor], row)
    else:
        return None

# Função para calcular a acurácia
def calcular_acuracia(arvore, df_teste, atributos):
    df_teste_binado = binarizar_dados(df_teste.copy(), atributos)
    total = len(df_teste_binado)
    acertos = 0

    for _, row in df_teste_binado.iterrows():
        classe_real = row.iloc[5]
        previsao = prever(arvore, row)
        if previsao == classe_real:
            acertos += 1

    acuracia = acertos / total
    return acuracia

# Função para calcular a precisão por classe
# Função para calcular a precisão por classe
def calcular_precisao(arvore, df_teste, atributos):
    df_teste_binado = binarizar_dados(df_teste.copy(), atributos)
    precisao_por_classe = {}

    for classe in classes:
        verdadeiros_positivos = 0
        falsos_positivos = 0

        for _, row in df_teste_binado.iterrows():
            classe_real = row.iloc[5]
            previsao = prever(arvore, row)
            
            if previsao == classe:
                if previsao == classe_real:
                    verdadeiros_positivos += 1
                else:
                    falsos_positivos += 1

        if verdadeiros_positivos + falsos_positivos > 0:
            precisao = verdadeiros_positivos / (verdadeiros_positivos + falsos_positivos)
        else:
            precisao = 0.0
        
        precisao_por_classe[classe] = precisao

    return precisao_por_classe

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
            leaf_key = f'({translation[value]})'  
            # leaf_key = f'{key} ({translation[value]}) {i}'
            graph.add_node(leaf_key, label =  f'({translation[value]})', subset=level + 1)
            graph.add_edge(node_id, leaf_key)

    return graph

def draw_interactive_graph(G):
    pos = nx.multipartite_layout(G, subset_key="subset")
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=(x0, x1), y=(y0, y1),
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))

    node_trace = go.Scatter(
        x=[], y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color='skyblue',
            size=30,
            line_width=2))

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([G.nodes[node]['label']])

    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False),
                        clickmode='event+select'))

    fig.update_traces(marker=dict(size=50, line=dict(width=2, color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.update_layout(dragmode='lasso')

    fig.show()

def main():
    global i
    translation = {
                1: 'Critico',
                2: 'Instavel',
                3: 'P-Estavel',
                4: 'Estavel'
            }
    df_treino = df_teste = ''
    current_path = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(current_path + '/DataSet.csv', header=None)
    
    if os.path.exists(current_path + "/Treino.csv") and os.path.exists(current_path + "/Teste.csv"):
        df_treino = pd.read_csv(current_path + '/Treino.csv', header=None)
        df_teste = pd.read_csv(current_path + '/Teste.csv', header=None)
    else:
        # Separar 20% das linhas para o conjunto de teste
        tamanho_teste = int(0.2 * len(df))
        #Salva arquivo de treino e de teste
        df_treino, df_teste = separar_treino_teste(df, tamanho_teste)
        df_treino.to_csv(current_path + '/Treino.csv', index=False, header=False)
        df_teste.to_csv(current_path + '/Teste.csv', index=False, header=False)
    # atributos = {
    #     'qPa': {'index': 1, 'bins': [-10,-7, -3, 3, 7, 10], 'labels': ['extremo','baixa', 'normal', 'alta','extremo']},
    #     'Pulso': {'index': 2, 'bins': [0, 60, 140, 200], 'labels': ['baixa', 'normal', 'alta']},
    #     'Respiração': {'index': 3, 'bins': [0, 12, 22], 'labels': ['baixa', 'normal']}
    # } 
    atributos = {
        'qPa': {'index': 1, 'bins': [-10,-7, -3, 3, 7, 10], 'labels': ['extremo','baixa', 'normal', 'alta','extremo']},
        'Pulso': {'index': 2, 'bins': [0, 40, 60, 110, 160, 200], 'labels': ['extremo','baixa', 'normal', 'alta','extremo']},
        'Respiração': {'index': 3, 'bins': [0, 7, 12, 19, 22], 'labels': ['extremo','baixa', 'normal', 'alta']}
    } 
    # atributos = {
    #     'Pressão': {'index': 1, 'bins': [-10, -3, 3, 10], 'labels': ['baixa', 'normal', 'alto']},
    #     'Pulso': {'index': 2, 'bins': [0, 60, 110, 200], 'labels': ['baixa', 'normal', 'alto']},
    #     'Respiração': {'index': 3, 'bins': [0, 7, 14, 22], 'labels': ['baixa', 'normal', 'alto']}
    # }

    entropia_total = calcular_entropia_total(df)
    print(f"Entropia Total do Dataset: {entropia_total}")

    arvore = id3(df, atributos)
    print("Árvore de Decisão ID3:")
    print(arvore)
    

    with open(current_path + '/arvore_decisao.json', 'w') as json_file:
        json.dump(arvore, json_file, indent=4,cls=NpEncoder)
    
    # Calcular acurácia
    acuracia = calcular_acuracia(arvore, df_teste,atributos)
    print(f"Acurácia: {acuracia:.2f}")

    precisao_por_classe = calcular_precisao(arvore, df_teste, atributos)
    
    for classe, precisao in precisao_por_classe.items():
        print(f"Precisão para a classe {translation[classe]}: {precisao:.2f}")
        
    # Construindo o grafo
    G = build_graph(arvore)
    print(f"Quantidade de nós da arvore {i}")
    draw_interactive_graph(G)
    
    #Outro modelo de construir o grafo
    #     labels = nx.get_node_attributes(G, 'label')
    #     # Desenhando o grafo com layout personalizado
    #  # Desenhando o grafo com layout multipartite
    #     pos = nx.multipartite_layout(G, subset_key="subset",align='horizontal',scale=5)  # Usando o layout multipartite
    #     plt.figure(figsize=(14, 10))
    #     nx.draw(G, pos , labels=labels, with_labels=True, node_color='skyblue', node_size=2000, edge_color='k', font_size=11, font_weight='bold', arrows=False)
    #     plt.title("Grafo Gerado a Partir do JSON")
    #     plt.show()

if __name__ == "__main__":
    main()