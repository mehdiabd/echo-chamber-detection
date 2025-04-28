"""Community Detection on Sample Social Graph via NetworkX and community-louvain"""

import json
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from node2vec import Node2Vec


# CLASSIC Approach: Louvain Detector
def detect_communities_louvain(g):
    """
    Run Louvain algorithm on graph G and return partition.
    G: networkx.Graph
    Returns: dict mapping node → community_id
    """
    partition = community_louvain.best_partition(g)
    return partition


# Plotting Function
def draw_partitioned_graph(
    g, partition, title="Louvain Community Detection", method="Classic"
):
    """
    Visualize the graph with node colors based on community partition.
    Includes a legend indicating the community and detection method.
    """
    # Ensure the function operates within the current figure context
    plt.gca().clear()  # Clear the current axes to avoid overlapping plots

    pos = nx.spring_layout(g, seed=42)
    cmap = cm.get_cmap('Set3', max(partition.values()) + 1)

    # Draw nodes with community colors
    nx.draw_networkx_nodes(g, pos, node_size=300,
                           node_color=list(partition.values()),
                           cmap=cmap)
    nx.draw_networkx_edges(g, pos, alpha=0.4)
    nx.draw_networkx_labels(g, pos, font_size=10)

    # Add legend for communities
    unique_communities = set(partition.values())
    for community in unique_communities:
        plt.scatter([], [], color=cmap(community), label=f"Community {community}\
            ({method})")

    plt.title(title)
    plt.legend(loc="best", title="Communities")
    plt.axis('off')
    plt.tight_layout()


# Main Louvain Example
def main_louvain_example():
    """
    Example of using Louvain algorithm for community detection.
    This example uses the Karate Club graph from NetworkX.
    """

    # Generate test graph (Karate Club)
    g = nx.karate_club_graph()

    # Run Louvain algorithm
    partition = detect_communities_louvain(g)

    # Print community results
    for node, comm in partition.items():
        print(f"Node {node}: Community {comm}")

    # Draw the result
    draw_partitioned_graph(g, partition)


# HYBRID Approach: Node2Vec + Clustering
def build_user_graph(messages):
    """
    Build a user interaction graph from message data.
    """
    g = nx.DiGraph()  # or use nx.Graph() if interactions are symmetric
    for msg in messages:
        sender = msg.get("sender")
        replied = msg.get("reply_to")
        if sender and replied:
            if g.has_edge(sender, replied):
                g[sender][replied]['weight'] += 1
            else:
                g.add_edge(sender, replied, weight=1)
    return g


def get_node_embeddings(g, dimensions=64):
    """
    Generate Node2Vec embeddings for the graph.
    """
    node2vec = Node2Vec(g, dimensions=dimensions, walk_length=20, num_walks=100,
                        workers=2)
    model = node2vec.fit(window=10, min_count=1)
    embeddings = [model.wv[str(node)] for node in g.nodes()]
    return embeddings, list(g.nodes())


def run_kmeans(embeddings, n_clusters=4):
    """
    Apply KMeans clustering on the embeddings.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels


def visualize_embeddings(embeddings, labels, nodes, method="Hybrid"):
    """
    Visualize the embeddings with PCA and clustering labels.
    Includes a legend indicating the community and detection method.
    """
    # Ensure the function operates within the current figure context
    plt.gca().clear()  # Clear the current axes to avoid overlapping plots

    reduced = PCA(n_components=2).fit_transform(embeddings)
    cmap = cm.get_cmap('Set3', len(set(labels)))

    for label in set(labels):
        idx = [i for i, l in enumerate(labels) if l == label]
        x = [reduced[i][0] for i in idx]
        y = [reduced[i][1] for i in idx]
        plt.scatter(x, y, label=f"Community {label} ({method})", s=100,
                    color=cmap(label))

    for i, node in enumerate(nodes):
        plt.text(reduced[i][0], reduced[i][1], str(node), fontsize=8)

    plt.title("Detected Communities (Node2Vec + KMeans)")
    plt.legend(loc="best", title="Communities")
    plt.tight_layout()


# Data Collection Example: Load Tweets Dataset
def load_tweets_dataset(file_path="res.json"):
    """
    Load tweets dataset from a JSON file and extract relevant fields.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        # Skip the "Total hits" line and load the JSON array
        f.readline()  # Skip the first line
        tweets = json.load(f)

    # Extract relevant fields for graph construction
    messages = []
    for tweet in tweets:
        sender = tweet.get("user_id")
        replied_to = tweet.get("entity", {}).get("mention", [])
        for reply in replied_to:
            messages.append({"sender": sender, "reply_to": reply})

    return messages


# Glue It All Together
def main_hybrid_example():
    """
    Example of using Node2Vec and KMeans for community detection.
    This example uses the tweets dataset.
    """

    # Load tweets dataset
    messages = load_tweets_dataset()

    # Build user interaction graph
    g = build_user_graph(messages)

    # Generate embeddings and run clustering
    embeddings, nodes = get_node_embeddings(g)
    labels = run_kmeans(embeddings, n_clusters=2)

    # Visualize the results
    visualize_embeddings(embeddings, labels, nodes)
    print(f"Number of nodes in graph: {len(g.nodes())}")
    print(f"Number of embeddings: {len(embeddings)}")


# Print community results
if __name__ == "__main__":
    print("Running Louvain Example...")
    plt.figure("Louvain Community Detection")
    main_louvain_example()

    print("\nRunning Hybrid Example...")
    plt.figure("Hybrid Community Detection")
    main_hybrid_example()

    print("Community detection completed.")
    plt.show()
