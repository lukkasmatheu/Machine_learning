import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('treino_sinais_vitais_com_label.txt', sep=',', header=None)

df.columns = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']

df = df.drop(['id','pSist', 'pDiast', 'gravidade'], axis=1)

y = df['classe']  
X = df.drop('classe', axis=1)  # Features

print(X.head())
print(y.head())


# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Inicializar o modelo de Random Forest
clf = RandomForestClassifier(n_estimators=200, max_depth=16, max_features="sqrt")

# Treinar o modelo
clf.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = clf.predict(X_test)

# Avaliar o desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy * 100:.2f}%')

# Relatório de classificação
print(classification_report(y_test, y_pred))
