import numpy as np
from datasetload import split_train_val_test, normalize_data
from mlp import classifier as mlp
from knn import tune_knn_hyperparameters
from decisionTree import tune_decision_tree
from sklearn.metrics import confusion_matrix, classification_report

# Função para treinar o modelo até atingir a acurácia desejada ou o número máximo de iterações
def retrain_until_accuracy(classifier, x_train, y_train, x_val, y_val, threshold=0.90, max_iterations=100):
    # classifier: o classificador que será treinado (pode ser KNN, MLP, Árvore de Decisão, etc.)
    # x_train: os dados de treino (features)
    # y_train: as labels de treino (rótulos/classes)
    # x_val: os dados de validação (features)
    # y_val: as labels de validação (rótulos/classes)
    # threshold: a acurácia desejada para parar o treinamento
    # max_iterations: o número máximo de iterações para o treinamento

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

# Caminho do arquivo de dados ->  MUDE O ARQUIVO DE DATASET DESEJADO PARA TREINO AQUI ! 
file_path = r"C:\Users\pietro.lessa\Documents\Pietro Uni 2024-1\IA\T1\MachineLearning-Tic-tac-toe\T1-IA\dataset\balanced_tic_tac_toe_dataset_500_entries.data" 

# Dividir o dataset em treino, validação e teste
x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(file_path)

# Normalizar os dados
x_train_normalized, x_val_normalized, x_test_normalized = normalize_data(x_train, x_val, x_test)

# Treinar e ajustar hiperparâmetros do KNN
print("Tuning KNN hyperparameters...")
best_knn = tune_knn_hyperparameters(x_train_normalized, y_train)
knn_accuracy = retrain_until_accuracy(best_knn, x_train_normalized, y_train, x_val_normalized, y_val)
print(f"KNN final accuracy: {knn_accuracy:.2f}")

# Treinar o MLP
print("Training MLP...")
mlp.set_params(max_iter=2000)  # Define o número máximo de iterações, ou seja, o número máximo de atualizações de peso que serão feitas durante o treinamento.
mlp_accuracy = retrain_until_accuracy(mlp, x_train_normalized, y_train, x_val_normalized, y_val)
print(f"MLP final accuracy: {mlp_accuracy:.2f}")

# Treinar e ajustar hiperparâmetros da Árvore de Decisão
print("Tuning Decision Tree hyperparameters...")
best_tree = tune_decision_tree(x_train_normalized, y_train)
tree_accuracy = retrain_until_accuracy(best_tree, x_train_normalized, y_train, x_val_normalized, y_val)
print(f"Decision Tree final accuracy: {tree_accuracy:.2f}")

# Função para avaliar o modelo no conjunto de teste
def evaluate_model(classifier, x_test, y_test):
    # classifier: o classificador treinado que será avaliado
    # x_test: os dados de teste (features)
    # y_test: as labels de teste (rótulos/classes)

    y_pred = classifier.predict(x_test)  # Prediz as classes no conjunto de teste
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))  # Imprime a matriz de confusão
    print("Classification Report:")
    print(classification_report(y_test, y_pred))  # Imprime o relatório de classificação

# Avaliar o KNN no conjunto de teste
print("Evaluating KNN on test set...")
evaluate_model(best_knn, x_test_normalized, y_test)

# Avaliar o MLP no conjunto de teste
print("Evaluating MLP on test set...")
evaluate_model(mlp, x_test_normalized, y_test)

# Avaliar a Árvore de Decisão no conjunto de teste
print("Evaluating Decision Tree on test set...")
evaluate_model(best_tree, x_test_normalized, y_test)
