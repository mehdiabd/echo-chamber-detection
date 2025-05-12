"""Module providing functions to analyze k-means clustering results."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from community_detection import get_node_embeddings


def plot_elbow_method(embeddings, max_k=10):
    """
    Plot the elbow method to find the optimal number of clusters (k).
    Args:
        embeddings (np.ndarray): Node embeddings.
        max_k (int): Maximum number of clusters to test.
    """
    distortions = []
    K = range(2, max_k + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(K, distortions, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    plt.show()

def plot_silhouette_scores(embeddings, max_k=10):
    """
    Plot silhouette scores for different values of k to find the optimal number of
    clusters.
    Args:
        embeddings (np.ndarray): Node embeddings.
        max_k (int): Maximum number of clusters to test.
    """
    silhouette_scores = []
    K = range(2, max_k + 1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 4))
    plt.plot(K, silhouette_scores, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score For Optimal k')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Load actual embeddings from your community detection code
    embeddings = get_node_embeddings()
    
    # Analyze the k-means threshold with your actual data
    plot_elbow_method(embeddings, max_k=10)
    plot_silhouette_scores(embeddings, max_k=10)