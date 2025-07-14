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
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


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

    # Show node labels if label_map is present, regardless of partition
    if hasattr(g, "graph") and "label_map" in g.graph:
        label_map = g.graph["label_map"]
        for node in g.nodes():
            label = label_map.get(node)
            if label:
                title = net.get_node(node).get("title", node)
                net.get_node(node)['title'] = f"{title} ({label})"

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
        target = msg.get("target")
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


# Data Collection: Load Interactions Dataset
def load_interactions(file_path="interactions.json"):
    """
    Load sender-target interactions from a JSON lines file.
    Each line is a dict with keys: sender, target, type
    """
    messages = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                msg = json.loads(line)
                sender = msg.get("sender")
                target = msg.get("target")
                if sender and target and sender != target:
                    messages.append(msg)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(messages)} interactions.")
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


def generate_time_slots(start_date, end_date, slot_type):
    slots = []
    current = start_date
    while current <= end_date:
        if slot_type == "daily":
            next_slot = current + timedelta(days=1)
        elif slot_type == "weekly":
            next_slot = current + timedelta(weeks=1)
        else:  # monthly
            next_slot = current + relativedelta(months=1)

        slot_end = min(next_slot - timedelta(days=1), end_date)
        slots.append((current, slot_end))
        current = next_slot
    return slots


# Print community results
if __name__ == "__main__":
    print("Running Louvain and Hybrid community detection...")

    start_date = datetime.strptime("2025-03-01", "%Y-%m-%d")
    end_date = datetime.strptime("2025-04-28", "%Y-%m-%d")
    delta_days = (end_date - start_date).days
    if delta_days <= 7:
        slot_type = "daily"
    elif delta_days <= 60:
        slot_type = "weekly"
    else:
        slot_type = "monthly"
    print(f"Slot type selected: {slot_type}")
    slots = generate_time_slots(start_date, end_date, slot_type)

    for idx, (slot_start, slot_end) in enumerate(slots):
        print(f"\nProcessing slot {slot_start.date()} to {slot_end.date()}...")

        messages = []
        all_messages = load_interactions()
        for msg in all_messages:
            date_str = msg.get("date")
            if not date_str:
                continue
            try:
                msg_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                if slot_start.date() <= msg_date <= slot_end.date():
                    messages.append(msg)
            except Exception as e:
                print(f"Skipping message with invalid date: {date_str} → {e}")
                continue
        if not messages:
            print("No valid messages found in this slot.")
            continue

        g_slot = build_user_graph(messages)
        low_degree_nodes = [n for n, d in g_slot.degree() if d < 1]
        g_slot.remove_nodes_from(low_degree_nodes)
        if g_slot.number_of_nodes() == 0:
            print("Graph is empty after filtering. Skipping.")
            continue

        if not nx.is_connected(g_slot):
            largest_cc = max(nx.connected_components(g_slot), key=len)
            g_slot = g_slot.subgraph(largest_cc).copy()

        partition_louvain = detect_communities_louvain(g_slot)
        embeddings, nodes = get_node_embeddings(g_slot)
        labels = run_kmeans(embeddings, n_clusters=5)
        partition_hybrid = {node: labels[i] for i, node in enumerate(nodes)}

        visualize_combined_dashboard(
            g_slot,
            partition_louvain,
            partition_hybrid,
            filename=f"dashboard_{slot_start.strftime('%y%m%d')}_to_{slot_end.strftime('%y%m%d')}.html"
        )


    print("Community detection completed.")


# --- Timeline Dashboard Generator ---
import os

def generate_timeline_dashboard(output_file="timeline_dashboard.html"):
    """
    Generates an interactive dashboard that allows switching between time slot graphs.
    """
    dashboards = sorted([
        f for f in os.listdir(".")
        if f.startswith("dashboard_") and f.endswith(".html") and "_to_" in f
    ])

    if not dashboards:
        print("No dashboard_*.html files found.")
        return

    options_html = ""
    iframes_html = ""

    for i, dash in enumerate(dashboards):
        visible = "block" if i == 0 else "none"
        label = dash.replace("dashboard_", "").replace(".html", "").replace("_to_", " → ")
        options_html += f'<option value="{dash}">{label}</option>\n'
        iframes_html += f'<iframe id="{dash}" src="{dash}" style="display: {visible}; width: 100%; height: 650px; border: none;"></iframe>\n'

    html_content = f"""
    <html>
    <head>
        <title>Timeline Graph Viewer</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
            }}
            select {{
                font-size: 16px;
                padding: 8px;
                margin-bottom: 20px;
            }}
        </style>
        <script>
            function showGraph() {{
                var selected = document.getElementById("graphSelector").value;
                var iframes = document.getElementsByTagName("iframe");
                for (var i = 0; i < iframes.length; i++) {{
                    iframes[i].style.display = "none";
                }}
                document.getElementById(selected).style.display = "block";
            }}
        </script>
    </head>
    <body>
        <h2>Select Time Slot:</h2>
        <select id="graphSelector" onchange="showGraph()">
            {options_html}
        </select>
        <div>
            {iframes_html}
        </div>
    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Timeline dashboard saved to {output_file}")

    generate_timeline_dashboard()
