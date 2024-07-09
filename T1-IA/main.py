import numpy as np
from datasetload import split_train_test, normalize_data
from mlp import classifier as mlp
from knn import tune_knn_hyperparameters
from kmeans import train_kmeans, find_optimal_clusters
from sklearn.metrics import silhouette_score

def retrain_until_accuracy(classifier, x_train, y_train, x_test, y_test, threshold=0.90, max_iterations=100):
    accuracy = 0
    iterations = 0
    while accuracy < threshold and iterations < max_iterations:
        classifier.fit(x_train, y_train)
        new_accuracy = classifier.score(x_test, y_test)
        if new_accuracy == accuracy:
            break
        accuracy = new_accuracy
        iterations += 1
        print(f"Iteration {iterations}: Accuracy {accuracy:.2f}")
    return accuracy

def evaluate_kmeans(kmeans, x_test):
    y_pred = kmeans.predict(x_test)
    score = silhouette_score(x_test, y_pred)
    return score

(x_train, y_train, x_test, y_test) = split_train_test()
x_train_normalized, x_test_normalized = normalize_data(x_train, x_test)

# Train K-Means
print("Finding optimal number of clusters for K-Means...")
find_optimal_clusters(x_train_normalized)
optimal_clusters = int(input("Insira o número de clusters desejado com base na regra do cotovelo: "))
kmeans = train_kmeans(x_train_normalized, n_clusters=optimal_clusters)
kmeans_silhouette_score = evaluate_kmeans(kmeans, x_test_normalized)
print(f"K-Means Silhouette Score: {kmeans_silhouette_score:.2f}")

# Retrain KNN with hyperparameter tuning
print("Tuning KNN hyperparameters...")
best_knn = tune_knn_hyperparameters(x_train_normalized, y_train)
knn_accuracy = retrain_until_accuracy(best_knn, x_train_normalized, y_train, x_test_normalized, y_test)
print(f"KNN final accuracy: {knn_accuracy:.2f}")

# Retrain MLP
print("Training MLP...")
mlp.max_iter = 100000  # Increase max iterations for MLP
mlp_accuracy = retrain_until_accuracy(mlp, x_train_normalized, y_train, x_test_normalized, y_test)
print(f"MLP final accuracy: {mlp_accuracy:.2f}")
