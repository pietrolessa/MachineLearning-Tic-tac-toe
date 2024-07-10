from datasetload import split_train_val_test, normalize_data
from knn import tune_knn_hyperparameters
from sklearn.preprocessing import StandardScaler

# Caminho do arquivo de dados
file_path = r"C:\Users\pietro.lessa\Documents\Pietro Uni 2024-1\IA\T1\MachineLearning-Tic-tac-toe\T1-IA\dataset\balanced_tic_tac_toe_dataset_2000_entries.data"

# Carregar e preparar os dados
x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(file_path)
x_train_normalized, x_val_normalized, x_test_normalized = normalize_data(x_train, x_val, x_test)

# Treinar o classificador k-NN com os melhores hiperparâmetros
classifier = tune_knn_hyperparameters(x_train_normalized, y_train)
classifier.fit(x_train_normalized, y_train)

# Verificação de treinamento
print(f"Acurácia no conjunto de validação: {classifier.score(x_val_normalized, y_val):.2f}")
print(f"Acurácia no conjunto de teste: {classifier.score(x_test_normalized, y_test):.2f}")

# Crie e ajuste o scaler no conjunto de treinamento normalizado
scaler = StandardScaler().fit(x_train_normalized)

# Teste manual do classificador
known_board = [1, 1, 1, 2, 2, 0, 0, 0, 0]  # Exemplo de tabuleiro conhecido
known_board_normalized = scaler.transform([known_board])  # Normalizar o exemplo
result = classifier.predict(known_board_normalized)[0]
label_map = {1: 'x', 2: 'o', 3: 'tie', 4: 'ongoing'}
print(f"Resultado esperado: x, Resultado obtido: {label_map.get(result, 'Resultado inesperado')}")

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
    transformed = [0 if s == 'b' else 1 if s == 'X' else 2 for s in board]
    print(f"Tabuleiro transformado: {transformed}")  # Adicionado para debug
    return transformed

def check_status_from_ai():
    global game_on
    changed_board = transform_board()
    changed_board_normalized = scaler.transform([changed_board])  # Normalizar a entrada
    result = classifier.predict(changed_board_normalized)[0]
    label_map = {1: 'x', 2: 'o', 3: 'tie', 4: 'ongoing'}
    result_label = label_map.get(result, 'Resultado inesperado')
    print(f"Resultado da IA: {result_label}")  # Adicionado para debug
    if result_label == 'ongoing':
        print('Ainda há jogo!')
    elif result_label == 'x':
        print('X venceu!')
        game_on = False
    elif result_label == 'o':
        print('O venceu!')
        game_on = False
    elif result_label == 'tie':
        print('Empate!')
        game_on = False
    else:
        print('Resultado inesperado!')

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
