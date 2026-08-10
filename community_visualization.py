"""Community Visualization Module"""
import networkx as nx
from collections import defaultdict
from community_naming import (
    DEFAULT_COMMUNITY_LABEL,
    analyze_community_content,
    build_community_classification_prompt,
    build_community_profile,
    coerce_allowed_label,
    get_community_label_color,
)

def build_community_visualization(
    g, partition, network, method="Unknown", node_label_map=None, classifier=None
):
    """Build enhanced network visualization for community detection."""
    # Initialize network
    network.from_nx(g)
    
    # Analyze community structure
    centers = {}  # Will store community centers
    communities = defaultdict(list)  # Will store community members
    
    # Find centers using various centrality measures
    for comm_id in set(partition.values()):
        comm_nodes = [n for n, c in partition.items() if c == comm_id]
        subgraph = g.subgraph(comm_nodes)
        
        # Try different centrality measures
        try:
            centrality = nx.betweenness_centrality(subgraph)
        except nx.NetworkXError:
            try:
                centrality = nx.eigenvector_centrality(subgraph)
            except nx.NetworkXError:
                centrality = dict(subgraph.degree())
        
        centers[comm_id] = max(centrality.items(), key=lambda x: x[1])[0]
        communities[comm_id].extend(comm_nodes)
    
    # Generate community info
    comm_info = {}
    for comm_id, members in communities.items():
        center = centers[comm_id]
        neighbors = [n for n in members if n != center]
        
        # Generate meaningful name
        text_content = analyze_community_content(center, neighbors)

        prompt = build_community_classification_prompt(
            members=members,
            text_content=text_content or f"Center account: {center}",
        )
        fallback_label = DEFAULT_COMMUNITY_LABEL
        if node_label_map:
            fallback_label = coerce_allowed_label(
                node_label_map.get(center),
                default=DEFAULT_COMMUNITY_LABEL,
            )

        llm_response = None
        if classifier:
            llm_response = classifier(prompt)
        if not llm_response:
            llm_response = {
                "selected_label": fallback_label,
                "confidence": 0,
                "reasoning": "fallback"
            }

        profile = build_community_profile(
            members=members,
            text_content=text_content,
            llm_response=llm_response,
        )
        label = profile.name
        
        # Get active members
        active = sorted(
            members,
            key=lambda x: g.degree(x),
            reverse=True
        )[:3]

        # Store community info
        comm_info[comm_id] = {
            'label': label,
            'confidence': profile.confidence,
            'reasoning': profile.reasoning,
            'description': profile.description,
            'size': len(members),
            'center': center,
            'members': members,
            'active': active
        }
    
    colors = {}  # Will map labels to colors
    
    # Sort communities by size
    sorted_comms = sorted(
        comm_info.items(),
        key=lambda x: x[1]['size'],
        reverse=True
    )
    
    # Assign colors and style nodes
    for comm_id, info in sorted_comms:
        color = get_community_label_color(info['label'])
        info['color'] = color
        colors[info['label']] = color
        
        # Style community nodes
        for node in info['members']:
            is_center = (node == info['center'])
            is_active = node in info['active']
            
            # Calculate node size
            base_size = 15
            degree_boost = min(g.degree(node) * 2, 25)
            role_boost = 15 if is_center else 10 if is_active else 0
            size = base_size + degree_boost + role_boost
            
            # Build tooltip
            roles = []
            if is_center:
                roles.append("مرکز جامعه")
            if is_active:
                roles.append("عضو فعال")
                
            tooltip = f"{node}"
            if roles:
                tooltip += f" ({' - '.join(roles)})"
            tooltip += f"\nجامعه: {info['label']}"
            
            # Update node styling
            node_data = network.get_node(node)
            node_data.update({
                'color': color,
                'label': node,
                'title': tooltip,
                'size': size,
                'community_label': info['label'],
                'borderWidth': 2 if is_center else 1,
                'borderWidthSelected': 3,
                'font': {'size': 14, 'face': 'Vazirmatn'}
            })
    
    # Generate legend data
    legend_data = []
    for comm_id, info in sorted_comms:
        # Format active members for tooltip
        active_str = ", ".join(info['active'][:3])
        
        # Create legend entry
        legend_entry = {
            'color': info['color'],
            'label': info['label'],
            'count': len(info['members']),
            'tooltip': f"""
مرکز: {info['center']}
اعضای فعال: {active_str}
تعداد کل: {len(info['members'])}
"""
        }
        legend_data.append(legend_entry)
    
    # Optimize network display
    network_options = {
        "physics": {
            "stabilization": {
                "enabled": True,
                "iterations": 100
            },
            "barnesHut": {
                "gravitationalConstant": -2000,
                "springConstant": 0.04,
                "springLength": 150
            }
        },
        "edges": {
            "color": {"inherit": False, "color": "#cccccc"},
            "width": 0.5,
            "smooth": {"enabled": False}
        },
        "groups": legend_data  # Add legend data to options
    }
    if hasattr(network.options, "update"):
        network.options.update(network_options)
    else:
        network.options = network_options
    
    return colors
