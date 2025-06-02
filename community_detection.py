"""Community Detection on Sample Social Graph via NetworkX and community-louvain"""
import json
from collections import defaultdict
from pyvis.network import Network
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from node2vec import Node2Vec


# Interactive Graph Visualization with PyVis
def visualize_graph_interactively(g, partition=None, title="Graph",
                                  filename="graph.html"):
    "Visualize a graph interactively using PyVis."
    net = Network(height="800px", width="100%", notebook=False)
    net.from_nx(g)

    # Optional: color nodes by community/cluster
    if partition:
        for node, community_id in partition.items():
            # visually distinct hues
            color = f"hsl({(community_id * 47) % 360}, 70%, 60%)"
            net.get_node(node)['color'] = color
            net.get_node(node)['title'] = f"{node} (Community {community_id})"

    net.force_atlas_2based()
    net.show_buttons(filter_=['physics'])
    net.show(filename)
    print(f"Interactive graph saved to {filename}")


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

    # Add interactive visualization
    visualize_graph_interactively(g, partition, title="Louvain Graph",
                                  filename="louvain_graph.html")


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

        # Treat every interaction type the same: weight = 1
        edge_weights[(sender, target)] += 1

    # Filter and add strong edges
    for (sender, target), total_weight in edge_weights.items():
        if total_weight >= 1:
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
                messages.append({"sender": source, "reply_to": target,
                                 "type": interaction_type})
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

    labels = run_kmeans(embeddings, n_clusters=5)

    print(f"Number of nodes in graph: {len(g.nodes())}")
    print(f"Number of embeddings: {len(embeddings)}")
    # Add interactive visualization
    partition = {node: labels[i] for i, node in enumerate(nodes)}
    visualize_graph_interactively(g, partition, title="Hybrid Graph",
                                  filename="hybrid_graph.html")


# New function to visualize combined dashboard
def visualize_combined_dashboard(g, louvain_partition, hybrid_partition,
                                 filename="combined_graphs.html"):
    """Visualize both Louvain and Hybrid community detection results in a combined
     dashboard."""

    # Louvain graph
    net1 = Network(height="600px", width="100%", notebook=False,
                   heading="Louvain Graph")
    net1.from_nx(g)
    for node, community_id in louvain_partition.items():
        color = f"hsl({(community_id * 47) % 360}, 70%, 60%)"
        net1.get_node(node)['color'] = color
        net1.get_node(node)['title'] = f"{node} (Louvain Community {community_id})"

    # Set fixed layout positions for both graphs using spring_layout
    import networkx as nx
    pos = nx.spring_layout(g, seed=42)

    # Disable physics only once
    net1.set_options('{"physics": {"enabled": false}}')
    for node in g.nodes():
        x, y = pos[node]
        net1.get_node(node)['x'] = x * 1000
        net1.get_node(node)['y'] = y * 1000

    net1.save_graph("louvain_graph.html")

    # Hybrid graph
    net2 = Network(height="600px", width="100%", notebook=False, heading="Hybrid Graph")
    net2.from_nx(g)
    for node, cluster_id in hybrid_partition.items():
        color = f"hsl({(cluster_id * 67) % 360}, 70%, 60%)"
        net2.get_node(node)['color'] = color
        net2.get_node(node)['title'] = f"{node} (Hybrid Cluster {cluster_id})"

    net2.set_options('{"physics": {"enabled": false}}')
    for node in g.nodes():
        x, y = pos[node]
        net2.get_node(node)['x'] = x * 1000
        net2.get_node(node)['y'] = y * 1000

    net2.save_graph("hybrid_graph.html")

    combined_html = """
    <html>
    <head>
        <title>Combined Community Detection Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
            }
            h2 {
                text-align: center;
                margin-top: 20px;
            }
            .container {
                display: flex;
                justify-content: space-around;
                flex-wrap: wrap;
                margin: 20px;
            }
            iframe {
                border: none;
                width: 48%;
                height: 600px;
            }
            @media (max-width: 800px) {
                iframe {
                    width: 100%;
                    height: 400px;
                    margin-bottom: 20px;
                }
            }
        </style>
    </head>
    <body>
        <h2>Louvain vs Hybrid Community Detection</h2>
        <div class="container">
            <iframe src="louvain_graph.html" title="Louvain Graph"></iframe>
            <iframe src="hybrid_graph.html" title="Hybrid Graph"></iframe>
        </div>
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(combined_html)

    print(f"Combined dashboard saved to {filename}")


# Print community results
if __name__ == "__main__":
    print("Running Louvain and Hybrid community detection...")

    # Louvain using real dataset
    real_messages = load_tweets_dataset()
    if not real_messages:
        print("No valid messages found in dataset.")
        exit()
    g_real = build_user_graph(real_messages)
    low_degree_nodes = [n for n, d in g_real.degree() if d < 3]
    g_real.remove_nodes_from(low_degree_nodes)
    print(f"Removed {len(low_degree_nodes)} low-degree nodes.")
    if g_real.number_of_nodes() == 0:
        print("Graph is empty after filtering. Aborting.")
        exit()

    if not nx.is_connected(g_real):
        largest_cc = max(nx.connected_components(g_real), key=len)
        g_real = g_real.subgraph(largest_cc).copy()
    main_louvain(g_real)

    # Hybrid
    main_hybrid(g_real, None)

    # Generate combined dashboard
    partition_louvain = detect_communities_louvain(g_real)
    embeddings, nodes = get_node_embeddings(g_real)
    labels = run_kmeans(embeddings, n_clusters=5)
    partition_hybrid = {node: labels[i] for i, node in enumerate(nodes)}
    visualize_combined_dashboard(g_real, partition_louvain, partition_hybrid)

    print("Community detection completed.")
