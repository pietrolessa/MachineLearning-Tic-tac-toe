import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Função para carregar e dividir os dados
def split_train_val_test(file_path, test_size=0.2, val_size=0.25):
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Converter rótulos para numéricos
    label_map = {'x': 1, 'o': 2, 'tie': 3, 'ongoing': 4}
    y = np.array([label_map[label] for label in y])
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Função para normalizar os dados
def normalize_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)
    X_test_normalized = scaler.transform(X_test)
    
    return X_train_normalized, X_val_normalized, X_test_normalized, scaler

# Função para ajustar os hiperparâmetros do KNN
def tune_knn_hyperparameters(X_train, y_train):
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 20, 30, 40, 50]
    }
    
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Carregar e preparar os dados
file_path = r"C:\Users\pietro.lessa\Documents\Pietro Uni 2024-1\IA\T1\MachineLearning-Tic-tac-toe\T1-IA\dataset\balanced_tic_tac_toe_dataset_2000_entries.data"
X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(file_path)
X_train_normalized, X_val_normalized, X_test_normalized, scaler = normalize_data(X_train, X_val, X_test)

# Verificar a distribuição das classes no conjunto de treinamento
unique, counts = np.unique(y_train, return_counts=True)
print("Distribuição das classes no conjunto de treinamento:", dict(zip(unique, counts)))

# Treinar o classificador k-NN com os melhores hiperparâmetros
classifier = tune_knn_hyperparameters(X_train_normalized, y_train)
classifier.fit(X_train_normalized, y_train)

# Função para testar diferentes estados do tabuleiro
def test_boards(classifier, scaler):
    label_map = {1: 'x', 2: 'o', 3: 'tie', 4: 'ongoing'}
    boards = [
        ([1, 1, 1, 2, 2, 0, 0, 0, 0], 'x'),  # X ganha
        ([2, 2, 2, 1, 1, 0, 0, 0, 0], 'o'),  # O ganha
        ([1, 2, 1, 2, 1, 2, 2, 1, 1], 'tie'),  # Empate
        ([1, 0, 0, 0, 2, 0, 0, 0, 1], 'ongoing'),  # Jogo em andamento
        ([0, 0, 0, 0, 0, 0, 0, 0, 0], 'ongoing')  # Jogo em andamento (tabuleiro vazio)
    ]

    for board, expected in boards:
        normalized_board = scaler.transform([board])
        result = classifier.predict(normalized_board)[0]
        print(f"Tabuleiro: {board}, Esperado: {expected}, Resultado: {label_map[result]}")

# Verificação de escalonamento dos dados normalizados
def verify_normalization():
    print("Verificação de Normalização dos Dados:")
    print("Primeiros 5 exemplos normalizados do conjunto de treino:")
    print(X_train_normalized[:5])
    print("Primeiros 5 exemplos normalizados do conjunto de validação:")
    print(X_val_normalized[:5])
    print("Primeiros 5 exemplos normalizados do conjunto de teste:")
    print(X_test_normalized[:5])

verify_normalization()
test_boards(classifier, scaler)
