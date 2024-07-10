from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Função para treinar a árvore de decisão
def train_decision_tree(x_train, y_train):
    tree = DecisionTreeClassifier()
    tree.fit(x_train, y_train)
    return tree

# Função para ajustar os hiperparâmetros da árvore de decisão
def tune_decision_tree(x_train, y_train):
    tree = DecisionTreeClassifier()
    param_grid = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10]
    }
    grid_search = GridSearchCV(tree, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    print("Melhores parâmetros encontrados:", grid_search.best_params_)
    return grid_search.best_estimator_

# Função principal para teste
if __name__ == "__main__":
    from datasetload import split_train_val_test, normalize_data
    
    # Use the appropriate file path for your dataset
    file_path = r"C:\Users\pietro.lessa\Documents\Pietro Uni 2024-1\IA\T1\MachineLearning-Tic-tac-toe\T1-IA\dataset\balanced_tic_tac_toe_dataset_2000_entries.data"
    
    x_train, y_train, x_val, y_val, x_test, y_test = split_train_val_test(file_path)
    x_train_normalized, x_val_normalized, x_test_normalized, scaler = normalize_data(x_train, x_val, x_test)
    
    print("Treinando a árvore de decisão...")
    best_tree = tune_decision_tree(x_train_normalized, y_train)
    print(f"Acurácia no conjunto de validação: {best_tree.score(x_val_normalized, y_val):.2f}")
    
    print("Avaliando no conjunto de teste...")
    y_pred = best_tree.predict(x_test_normalized)
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred))
