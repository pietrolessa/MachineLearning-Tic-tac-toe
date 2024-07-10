from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Função para carregar e ler o arquivo de dados
def load_data(file_path):
    with open(file_path) as file:
        file_data = file.read().split('\n')  # Lê o conteúdo do arquivo e divide em linhas
    return file_data

# Função para carregar e preprocessar o dataset
def load_dataset(file_data):
    preprocessed_data = []
    for line in file_data:
        if line:  # Verifica se a linha não está vazia para evitar erros
            # Converte o estado do tabuleiro para números: 'b' para 0, 'x' para 1, 'o' para 2
            board_state = [0 if s == 'b' else 1 if s == 'x' else 2 for s in line.split(',')[:-1]]
            outcome = line.split(',')[-1]  # Resultado do jogo: última coluna
            preprocessed_data.append((board_state, outcome))  # Adiciona o estado e o resultado à lista
    attributes = [d[0] for d in preprocessed_data]  # Extrai apenas os estados do tabuleiro
    classes = [d[1] for d in preprocessed_data]  # Extrai apenas os resultados do jogo
    return attributes, classes

# Função para dividir o dataset em conjuntos de treino, validação e teste
def split_train_val_test(file_path):
    file_data = load_data(file_path)  # Carrega os dados do arquivo
    attributes, classes = load_dataset(file_data)  # Preprocessa os dados
    
    # 1. Dividir em 90% treino/validação e 10% teste
    # Usamos train_test_split para separar 10% dos dados para teste
    x_train_val, x_test, y_train_val, y_test = train_test_split(attributes, classes, test_size=0.1, random_state=42)
    
    # 2. Dividir os 90% restantes em 70% treino e 20% validação
    # Usamos train_test_split novamente para dividir os 90% de treino/validação em 70% treino e 20% validação
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=2/9, random_state=42)
    
    return x_train, y_train, x_val, y_val, x_test, y_test  # Retorna os conjuntos divididos

# Função para normalizar os dados
def normalize_data(x_train, x_val, x_test):
    scaler = StandardScaler()  # Cria um objeto StandardScaler para normalização
    x_train_normalized = scaler.fit_transform(x_train)  # Ajusta e transforma os dados de treino
    x_val_normalized = scaler.transform(x_val)  # Transforma os dados de validação
    x_test_normalized = scaler.transform(x_test)  # Transforma os dados de teste
    return x_train_normalized, x_val_normalized, x_test_normalized  # Retorna os dados normalizados

# Função principal atualizada para testes
if __name__ == "__main__":
    file_path = r"C:\Users\pietro.lessa\Documents\Pietro Uni 2024-1\IA\T1\MachineLearning-Tic-tac-toe\T1-IA\dataset\tic_tac_toe_dataset2.data"
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(file_path)  # Divide o dataset
    x_train_normalized, x_val_normalized, x_test_normalized = normalize_data(x_train, x_val, x_test)  # Normaliza os dados
    # Imprime os primeiros 5 exemplos de cada conjunto para verificação
    print(f"x_train_normalized: {x_train_normalized[:5]}")
    print(f"x_val_normalized: {x_val_normalized[:5]}")
    print(f"x_test_normalized: {x_test_normalized[:5]}")
    print(f"y_train: {y_train[:5]}")
    print(f"y_val: {y_val[:5]}")
    print(f"y_test: {y_test[:5]}")
