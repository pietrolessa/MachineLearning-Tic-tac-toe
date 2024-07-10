import random

def check_winner(board):
    # Define as combinações vencedoras
    winning_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # linhas
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # colunas
        [0, 4, 8], [2, 4, 6]              # diagonais
    ]
    for combo in winning_combinations:
        if board[combo[0]] == board[combo[1]] == board[combo[2]] and board[combo[0]] != 'b':
            return board[combo[0]]
    if 'b' not in board:
        return 'tie'
    return 'ongoing'

def generate_random_game():
    board = ['b'] * 9
    moves = list(range(9))
    random.shuffle(moves)
    current_player = 'x'
    states = []

    for move in moves:
        board[move] = current_player
        state = ''.join(board)
        winner = check_winner(board)
        if winner != 'ongoing':
            break
        current_player = 'o' if current_player == 'x' else 'x'
        states.append((state, 'ongoing'))

    result_label = 'winX' if winner == 'x' else 'wincircle' if winner == 'o' else 'tie'
    states.append((state, result_label))
    return states

def generate_balanced_dataset(target_size_per_class):
    dataset = {'winX': [], 'wincircle': [], 'tie': [], 'ongoing': []}
    
    while min(len(dataset['winX']), len(dataset['wincircle']), len(dataset['tie'])) < target_size_per_class:
        game_states = generate_random_game()
        for state, result in game_states:
            if result in dataset and len(dataset[result]) < target_size_per_class:
                dataset[result].append((state, result))
    
    balanced_dataset = dataset['winX'] + dataset['wincircle'] + dataset['tie'] + dataset['ongoing']
    random.shuffle(balanced_dataset)
    return balanced_dataset

def save_dataset(dataset, file_path):
    with open(file_path, 'w') as file: 
        for state, result in dataset:
            board_state = ','.join(state) + ',' + result
            file.write(board_state + '\n')

if __name__ == "__main__":
    target_size_per_class = 500  # Número desejado de exemplos por classe
    dataset = generate_balanced_dataset(target_size_per_class)
    save_dataset(dataset, 'balanced_tic_tac_toe_dataset_5000_entries.data')
