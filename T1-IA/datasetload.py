from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Open and read the dataset file
with open(r"C:\Users\pietro.lessa\Documents\Pietro Uni 2024-1\IA\T1\T1-IA\T1-IA\dataset\tic-tac-toe.data") as file:
    file_data = file.read().split('\n')

# Function to load and preprocess the dataset
def load_dataset():
    preprocessed_data = []
    for line in file_data:
        if line:  # Add check to avoid processing empty lines
            board_state = [0 if s == 'b' else 1 if s == 'x' else 2 for s in line.split(',')[:-1]]
            outcome = line.split(',')[-1]
            preprocessed_data.append((board_state, outcome))
    attributes = [d[0] for d in preprocessed_data]
    classes = [d[1] for d in preprocessed_data]
    return (attributes, classes)

# Function to load the dataset by class
def load_dataset_by_class():
    win_data = []
    lost_data = []
    tie_data = []
    game_data = []

    for line in file_data:
        if line:  # Add check to avoid processing empty lines
            board_state = [0 if s == 'b' else 1 if s == 'x' else 2 for s in line.split(',')[:-1]]
            outcome = line.split(',')[-1]

            if outcome == 'win':
                win_data.append((board_state, outcome))
            elif outcome == 'lost':
                lost_data.append((board_state, outcome))
            elif outcome == 'tie':
                tie_data.append((board_state, outcome))
            elif outcome == 'game':
                game_data.append((board_state, outcome))

    return (win_data, lost_data, game_data, tie_data)

# Function to split the dataset into training and testing sets
def split_train_test():
    (win_data, lost_data, game_data, tie_data) = load_dataset_by_class()
    x_win = [d[0] for d in win_data]
    x_lost = [d[0] for d in lost_data]
    x_game = [d[0] for d in game_data]
    x_tie = [d[0] for d in tie_data]

    y_win = [d[1] for d in win_data]
    y_lost = [d[1] for d in lost_data]
    y_game = [d[1] for d in game_data]
    y_tie = [d[1] for d in tie_data]

    # Split the data separately since the dataset is imbalanced
    x_train_win, x_test_win, y_train_win, y_test_win = train_test_split(x_win, y_win, test_size=0.3, random_state=42)
    x_train_lost, x_test_lost, y_train_lost, y_test_lost = train_test_split(x_lost, y_lost, test_size=0.3, random_state=42)
    x_train_game, x_test_game, y_train_game, y_test_game = train_test_split(x_game, y_game, test_size=0.3, random_state=42)
    x_train_tie, x_test_tie, y_train_tie, y_test_tie = train_test_split(x_tie, y_tie, test_size=0.3, random_state=42)

    x_train = x_train_win + x_train_lost + x_train_tie + x_train_game
    y_train = y_train_win + y_train_lost + y_train_tie + y_train_game
    x_test = x_test_win + x_test_lost + x_test_tie + x_test_game
    y_test = y_test_win + y_test_lost + y_test_tie + y_test_game

    return (x_train, y_train, x_test, y_test)

# Function to normalize the data
def normalize_data(x_train, x_test):
    scaler = StandardScaler()
    x_train_normalized = scaler.fit_transform(x_train)
    x_test_normalized = scaler.transform(x_test)
    return x_train_normalized, x_test_normalized
