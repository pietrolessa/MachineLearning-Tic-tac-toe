import numpy as np
from datasetload import split_train_val_test, normalize_data
from mlp import classifier as mlp
from knn import tune_knn_hyperparameters, classifier as knn
from decisionTree import tune_decision_tree
from sklearn.metrics import confusion_matrix, classification_report

# Função para treinar o modelo até atingir a acurácia desejada ou o número máximo de iterações
def retrain_until_accuracy(classifier, x_train, y_train, x_val, y_val, threshold=0.90, max_iterations=100):
    accuracy = 0  # Inicializa a acurácia
    iterations = 0  # Inicializa o contador de iterações
    while accuracy < threshold and iterations < max_iterations:
        classifier.fit(x_train, y_train)  # Treina o classificador com o conjunto de treino
        new_accuracy = classifier.score(x_val, y_val)  # Avalia o classificador no conjunto de validação
        if new_accuracy == accuracy:  # Se a acurácia não mudar, interrompe o treinamento
            break
        accuracy = new_accuracy
        iterations += 1
        print(f"Iteration {iterations}: Accuracy {accuracy:.2f}")
    return accuracy

# Caminho do arquivo de dados
file_path = r"C:\Users\pietro.lessa\Documents\Pietro Uni 2024-1\IA\T1\MachineLearning-Tic-tac-toe\T1-IA\dataset\balanced_tic_tac_toe_dataset_5000_entries.data"

# Dividir o dataset em treino, validação e teste
x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(file_path)
x_train_normalized, x_val_normalized, x_test_normalized, scaler = normalize_data(x_train, x_val, x_test)

# Função para escolher o classificador
def choose_classifier():
    print("Escolha o classificador:")
    print("1. KNN")
    print("2. MLP")
    print("3. Decision Tree")
    choice = input("Digite o número do classificador desejado: ")

    if choice == '1':
        print("Tuning KNN hyperparameters...")
        classifier = tune_knn_hyperparameters(x_train_normalized, y_train)
        retrain_until_accuracy(classifier, x_train_normalized, y_train, x_val_normalized, y_val)
    elif choice == '2':
        print("Training MLP...")
        mlp.set_params(max_iter=2000)
        classifier = mlp
        retrain_until_accuracy(classifier, x_train_normalized, y_train, x_val_normalized, y_val)
    elif choice == '3':
        print("Tuning Decision Tree hyperparameters...")
        classifier = tune_decision_tree(x_train_normalized, y_train)
        retrain_until_accuracy(classifier, x_train_normalized, y_train, x_val_normalized, y_val)
    else:
        print("Escolha inválida. Usando KNN como padrão.")
        classifier = tune_knn_hyperparameters(x_train_normalized, y_train)
        retrain_until_accuracy(classifier, x_train_normalized, y_train, x_val_normalized, y_val)

    return classifier

classifier = choose_classifier()
classifier.fit(x_train_normalized, y_train)

board = ["b", "b", "b",
         "b", "b", "b",
         "b", "b", "b"]

game_on = True
current_player = "X"

def display_board():
    print(board[0] + " | " + board[1] + " | " + board[2] + "      " + "1 | 2 | 3")
    print(board[3] + " | " + board[4] + " | " + board[5] + "      " + "4 | 5 | 6")
    print(board[6] + " | " + board[7] + " | " + board[8] + "      " + "7 | 8 | 9")
    print("\n")

def player_position():
    global current_player
    print("Vez de: " + current_player)
    position = input("Escolha uma posição entre 1 e 9: ")

    valid = False
    while not valid:
        while position not in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            position = input("Escolha uma posição entre 1 e 9: ")
        position = int(position) - 1

        if board[position] == "b":
            valid = True
        else:
            print("Posição já ocupada, escolha outra!")
    board[position] = current_player
    display_board()

def transform_board():
    return [0 if s == 'b' else 1 if s == 'X' else 2 for s in board]

def check_status_from_ai():
    global game_on
    changed_board = transform_board()
    print(f"Tabuleiro transformado: {changed_board}")  # Adicionado para debug
    changed_board_normalized = scaler.transform([changed_board])  # Normalizar a entrada
    #label_map = {1: "winX", 2: "wincircle", 3: "tie", 4: "game"}
    result = classifier.predict(changed_board_normalized)[0]  
    print(f"Resultado da IA: {result}") 
    if result == 'game':
        print('Ainda há jogo!')
    elif result == 'winX':
        print('X venceu!')
        game_on = False
    elif result == 'wincircle':
        print('O venceu!')
        game_on = False
    elif result == 'tie':
        print('Empate!')
        game_on = False

def flip_player():
    global current_player
    if current_player == "X":
        current_player = "O"
    else:
        current_player = "X"

def play_game():
    print("My Tic Tac Toe Game\n")
    display_board()

    while game_on:
        player_position()
        check_status_from_ai()
        if game_on:  # Flip player only if game is still ongoing
            flip_player()

play_game()
