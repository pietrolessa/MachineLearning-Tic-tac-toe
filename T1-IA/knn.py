from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# Define the initial classifier
classifier = KNeighborsClassifier(n_neighbors=8)

# Function to perform Grid Search for hyperparameter tuning
def tune_knn_hyperparameters(x_train, y_train):
    param_grid = {
        'n_neighbors': np.arange(1, 31, 2),  # Test odd numbers from 1 to 30
        'weights': ['uniform', 'distance'],  # Test different weight functions
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Test different algorithms
        'leaf_size': [10, 20, 30, 40, 50]  # Test different leaf sizes
    }
    
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Example usage of the tune_knn_hyperparameters function
if __name__ == "__main__":
    from datasetload import split_train_test, normalize_data
    x_train, x_test, y_train, y_test = split_train_test()
    x_train_normalized, x_test_normalized = normalize_data(x_train, x_test)
    
    best_knn = tune_knn_hyperparameters(x_train_normalized, y_train)
    classifier = best_knn

    # Evaluate the best KNN
    accuracy = best_knn.score(x_test_normalized, y_test)
    print(f"KNN Accuracy: {accuracy:.2f}")
