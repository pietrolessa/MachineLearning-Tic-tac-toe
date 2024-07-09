import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datasetload import split_train_test

def find_optimal_clusters(x_train, max_k=10):
    inertia = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)  # Define n_init explicitamente
        kmeans.fit(x_train)
        inertia.append(kmeans.inertia_)
    
    # Regra do cotovelo: Plotando a inércia em relação ao número de clusters
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k+1), inertia, marker='o')
    plt.title('Regra do Cotovelo')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inércia')
    plt.show()

def train_kmeans(x_train, n_clusters=3, random_state=0):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)  # Define n_init explicitamente
    kmeans.fit(x_train)
    return kmeans

def main():
    x_train, y_train, x_test, y_test = split_train_test()
    
    # Encontrar o número ótimo de clusters usando a regra do cotovelo
    find_optimal_clusters(x_train)
    
    # Treinar o modelo k-means com o número escolhido de clusters (ajustar conforme necessário)
    optimal_clusters = int(input("Insira o número de clusters desejado com base na regra do cotovelo: "))
    kmeans = train_kmeans(x_train, n_clusters=optimal_clusters)
    
    print("Centros dos clusters:", kmeans.cluster_centers_)
    
    y_pred = kmeans.predict(x_test)
    
    print("Rótulos preditos para os dados de teste:", y_pred)

if __name__ == "__main__":
    main()
