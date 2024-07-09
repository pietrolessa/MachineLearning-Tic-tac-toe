from datasetload import split_train_test
from knn import classifier

(x_train, y_train, x_test, y_test) = split_train_test()

classifier.fit(x_train, y_train)

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


def play_game():
    print("My Tic Tac Toe Game\n")
    display_board()

    while game_on:
        player_position()

        def check_status_from_ai():
            global game_on
            changed_board = transform_board()
            print(f"Tabuleiro transformado: {changed_board}")  # Adicionado para debug
            [result] = classifier.predict([changed_board])
            print(f"Resultado da IA: {result}")  # Adicionado para debug
            if result == 'game':
                print('Ainda há jogo!')
            elif result == 'win':
                print('X venceu!')
                game_on = False
            elif result == 'lost':
                print('O venceu!')
                game_on = False
            elif result == 'tie':
                print('Empate!')
                game_on = False
                exit()

        def flip_player():
            global current_player
            if current_player == "X":
                current_player = "O"
            else:
                current_player = "X"

        flip_player()
        check_status_from_ai()


play_game()
