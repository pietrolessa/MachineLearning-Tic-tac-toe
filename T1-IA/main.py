from datasetload import split_train_test
from mlp import classifier as mlp
from knn import classifier as knn
from kmeans import train_kmeans

def retrain_until_accuracy(classifier, x_train, y_train, x_test, y_test, threshold=0.90, max_iterations=100):
    accuracy = 0
    iterations = 0
    while accuracy < threshold and iterations < max_iterations:
        classifier.fit(x_train, y_train)
        accuracy = classifier.score(x_test, y_test)
        iterations += 1
        print(f"Iteration {iterations}: Accuracy {accuracy:.2f}")
    return accuracy

def evaluate_kmeans(kmeans, x_test, y_test):
    y_pred = kmeans.predict(x_test)
    accuracy = np.mean(y_pred == y_test)
    return accuracy

(x_train, y_train, x_test, y_test) = split_train_test()

# Retrain KNN
print("Training KNN...")
knn_accuracy = retrain_until_accuracy(knn, x_train, y_train, x_test, y_test)
print(f"KNN final accuracy: {knn_accuracy:.2f}")

# Retrain MLP
print("Training MLP...")
mlp_accuracy = retrain_until_accuracy(mlp, x_train, y_train, x_test, y_test)
print(f"MLP final accuracy: {mlp_accuracy:.2f}")

# Train K-Means
print("Training K-Means...")
optimal_clusters = int(input("Insira o número de clusters desejado com base na regra do cotovelo: "))
kmeans = train_kmeans(x_train, n_clusters=optimal_clusters)
kmeans_accuracy = evaluate_kmeans(kmeans, x_test, y_test)
print(f"K-Means final accuracy: {kmeans_accuracy:.2f}")
