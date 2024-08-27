import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar os dados
df = pd.read_csv('treino_sinais_vitais_com_label.txt', sep=',', header=None)
df.columns = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']

# Remover colunas desnecessárias
df = df.drop(['id', 'pSist', 'pDiast', 'gravidade'], axis=1)

# Separar features e rótulos
y = df['classe']
X = df.drop('classe', axis=1)

# Reclassificar os rótulos para que estejam no intervalo [0, num_classes-1]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("Classes:", label_encoder.classes_)  # Verificar as classes reclassificadas

# Dividir os dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar os dados

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar o modelo
model = Sequential([
    tf.keras.layers.Dense(18, input_shape=(X_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(18, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 neurônios na camada de saída
])

# Compilar o modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.2), metrics=['accuracy'])

# Treinar o modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.1, shuffle=True)


# Fazer previsões
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred_classes)
print(f'Acurácia: {accuracy * 100:.2f}%')

print(classification_report(y_test, y_pred_classes))

tf.keras.utils.plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

y_pred_classes = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.show()
