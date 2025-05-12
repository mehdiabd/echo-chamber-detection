"""Community Detection on Sample Social Graph via NetworkX and community-louvain"""
import json
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from node2vec import Node2Vec
from collections import defaultdict


# Visualization for Embeddings (PCA + Clusters)
def visualize_embeddings(embeddings, labels, nodes, ax):
    """
    Visualize node embeddings in 2D using PCA and color them by cluster labels.
    """
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    unique_labels = list(set(labels))
    num_clusters = len(unique_labels)
    cmap = plt.cm.get_cmap("tab10", num_clusters)

    label_to_color = {label: cmap(i / (num_clusters - 1) if num_clusters > 1 else 0)
                      for i, label in enumerate(unique_labels)}
    colors = [label_to_color[label] for label in labels]

    for i, (x, y) in enumerate(reduced_embeddings):
        ax.scatter(x, y, color=colors[i], label=f"Node {nodes[i]}" if nodes else None)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   label=f'Cluster {label}', markersize=8)
        for label, color in label_to_color.items()
    ]
    ax.legend(handles=handles, title="KMeans Clusters", loc="best")
    ax.set_title("Node Embeddings PCA with KMeans Clustering (Hybrid Method)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.grid(True)


# CLASSIC Approach: Louvain Detector
def detect_communities_louvain(g):
    """
    Run Louvain algorithm on graph G and return partition.
    G: networkx.Graph
    Returns: dict mapping node → community_id
    """
    partition = community_louvain.best_partition(g, weight='weight')
    return partition


# Plotting Function
def draw_partitioned_graph(g, partition, title="Louvain Community Detection",
                           method="Classic"):
    """
    Visualize the graph with node colors based on community partition.
    Includes a legend indicating the community and detection method.
    """
    # Ensure the function operates within the current figure context
    plt.gca().clear()  # Clear the current axes to avoid overlapping plots

    pos = nx.spring_layout(g, seed=42, k=0.3, iterations=50)
    # Use ListedColormap to limit colors
    cmap = ListedColormap(colormaps["Set3"].colors[:max(partition.values()) + 1])

    # Draw nodes with community colors
    nx.draw_networkx_nodes(g, pos, node_size=300,
                           node_color=list(partition.values()),
                           cmap=cmap)
    nx.draw_networkx_edges(g, pos, alpha=0.4)

    # Add legend for communities
    unique_communities = set(partition.values())
    for community in unique_communities:
        plt.scatter([], [], color=cmap(community), label=f"Community {community}"
                    f" ({method})")

    plt.title(f"{title} (Louvain)")
    plt.legend(loc="center left", bbox_to_anchor=(-0.25, 0.5),
               title="Communities")
    plt.axis('off')
    plt.tight_layout()


# Main Louvain
def main_louvain(g):
    """
    Using Louvain algorithm for community detection.
    g: networkx.Graph
    """
    if g.number_of_nodes() == 0:
        print("Graph is empty. Louvain cannot proceed.")
        return
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
    g = nx.Graph()
    edge_weights = defaultdict(int)

    for msg in messages:
        sender = msg.get("sender")
        target = msg.get("reply_to")
        if not sender or not target or sender == target:
            continue

        # Assign interaction weights
        weight = 0
        if msg.get("type") == "reply":
            weight = 3
        elif msg.get("type") == "quote":
            weight = 2
        elif msg.get("type") == "mention":
            weight = 1

        edge_weights[(sender, target)] += weight

    # Filter and add strong edges
    for (sender, target), total_weight in edge_weights.items():
        if total_weight >= 5:
            g.add_edge(sender, target, weight=total_weight)
    print(f"Graph has {g.number_of_nodes()} nodes and {g.number_of_edges()} edges.")
    for edge in g.edges(data=True):
        print(f"Edge: {edge}")
    return g


def get_node_embeddings(g, dimensions=64):
    """
    Generate Node2Vec embeddings for the graph.
    """
    node2vec = Node2Vec(g, dimensions=dimensions, walk_length=20, num_walks=100,
                        workers=2)
    model = node2vec.fit(window=10, min_count=1)

    # Ensure embeddings are generated for all nodes
    embeddings = []
    nodes = list(g.nodes())
    for node in nodes:
        try:
            embeddings.append(model.wv[str(node)])
        except KeyError:
            print(f"Warning: No embedding found for node {node}. Using a zero vector.")
            # Use a zero vector for missing embeddings
            embeddings.append([0] * dimensions)

    # Validate embedding lengths
    for i, emb in enumerate(embeddings):
        if len(emb) != dimensions:
            raise ValueError(f"Embedding at index {i} has incorrect dimension "
                             f"{len(emb)}; expected {dimensions}.")

    embeddings = np.array([np.array(emb) for emb in embeddings])
    return embeddings, nodes


# Data Collection: Load Tweets Dataset
def load_tweets_dataset(file_path="res.json"):
    """
    Load tweets dataset from a JSON file and extract relevant fields.
    """
    messages = []
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print("Failed to load JSON file:", e)
            return []

        for tweet in data:
            source = tweet.get("user", {}).get("username")
            if not source:
                continue

            if tweet.get("retweetedTweet") and not tweet.get("quotedTweet") and not tweet.get("mentionedUsers"):
                continue  # skip pure retweets

            # Track interactions and their type
            targets = set()
            # Extract mentioned users
            mentioned_users = tweet.get("mentionedUsers", [])
            for m in mentioned_users:
                if isinstance(m, dict) and "username" in m:
                    targets.add(m["username"])

            # Extract quoted tweet user
            quoted = tweet.get("quotedTweet")
            quoted_user = {}
            if quoted and isinstance(quoted, dict):
                quoted_user = quoted.get("user", {})
                if isinstance(quoted_user, dict) and "username" in quoted_user:
                    targets.add(quoted_user["username"])

            # Extract retweeted tweet user
            retweeted = tweet.get("retweetedTweet")
            retweeted_user = {}
            if retweeted and isinstance(retweeted, dict):
                retweeted_user = retweeted.get("user", {})
                if isinstance(retweeted_user, dict) and "username" in retweeted_user:
                    targets.add(retweeted_user["username"])

            for target in targets:
                interaction_type = "mention"  # default
                if target == quoted_user.get("username", ""):
                    interaction_type = "quote"
                elif target == retweeted_user.get("username", ""):
                    interaction_type = "retweet"
                elif target in [m.get("username", "") for m in mentioned_users]:
                    interaction_type = "mention"
                messages.append({"sender": source, "reply_to": target, "type": interaction_type})
    print(f"Loaded {len(messages)} messages.")
    if len(messages) < 5:
        print("First few messages:", messages[:5])
    return messages


# Glue It All Together
def run_kmeans(embeddings, n_clusters=2):
    """
    Run KMeans clustering on the embeddings and return cluster labels.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels


def main_hybrid(g, ax):
    """
    Using Node2Vec and KMeans for community detection.
    g: networkx.Graph
    ax: matplotlib.axes.Axes
    """

    if g.number_of_nodes() == 0:
        print("Graph is empty. Hybrid method cannot proceed.")
        return

    # Generate embeddings and run clustering
    embeddings, nodes = get_node_embeddings(g)
    print(f"Generated {len(embeddings)} embeddings for {len(nodes)} nodes.")

    # Check if it's safe to cluster
    if len(embeddings) < 2:
        print("Not enough embeddings for clustering.")
        return

    labels = run_kmeans(embeddings, n_clusters=2)

    # Visualize the results
    draw_kmeans_partitioned_graph(g, nodes, labels)
    print(f"Number of nodes in graph: {len(g.nodes())}")
    print(f"Number of embeddings: {len(embeddings)}")


# New function: Draw graph with KMeans cluster coloring
def draw_kmeans_partitioned_graph(g, nodes, labels, title="Node2Vec + KMeans Community Detection"):
    """
    Draw the graph with node colors based on KMeans clustering results.
    """
    # Clear the current axes
    plt.gca().clear()

    # Create a mapping from node to label
    partition = {node: labels[i] for i, node in enumerate(nodes)}

    pos = nx.spring_layout(g, seed=42, k=0.3, iterations=50)
    unique_labels = set(labels)
    cmap = ListedColormap(plt.cm.get_cmap("tab10", len(unique_labels)).colors)

    # Draw nodes with community colors
    nx.draw_networkx_nodes(g, pos, node_size=300,
                           node_color=[partition.get(node, 0) for node in g.nodes()],
                           cmap=cmap)
    nx.draw_networkx_edges(g, pos, alpha=0.4)

    # Add legend for clusters
    for label in unique_labels:
        plt.scatter([], [], color=cmap(label), label=f"Cluster {label} (KMeans)")

    plt.title(title)
    plt.legend(loc="best", title="KMeans Clusters")
    plt.axis('off')
    plt.tight_layout()


# Print community results
if __name__ == "__main__":
    print("Running Louvain and Hybrid community detection...")

    # Set up subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    plt.suptitle("Community Detection Methods: Louvain vs. Hybrid (Node2Vec + KMeans)",
                 fontsize=16)

    # Louvain using real dataset
    real_messages = load_tweets_dataset()
    if not real_messages:
        print("No valid messages found in dataset.")
        exit()
    g_real = build_user_graph(real_messages)
    low_degree_nodes = [n for n, d in g_real.degree() if d < 3]
    g_real.remove_nodes_from(low_degree_nodes)
    print(f"Removed {len(low_degree_nodes)} low-degree nodes.")
    if not nx.is_connected(g_real):
        largest_cc = max(nx.connected_components(g_real), key=len)
        g_real = g_real.subgraph(largest_cc).copy()
    plt.sca(axs[0])
    main_louvain(g_real)

    # Hybrid
    plt.sca(axs[1])
    main_hybrid(g_real, axs[1])

    print("Community detection completed.")
    plt.tight_layout()
    plt.show()
