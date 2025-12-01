"""Community Visualization Module"""
import networkx as nx
from collections import defaultdict
import random
import json
from community_naming import (
    analyze_community_content,
    get_community_type,
    get_community_role,
    name_community_from_center
)

def build_community_visualization(g, partition, network, method="Unknown", node_label_map=None):
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
        
        if text_content:
            # Get community type and role
            comm_type = get_community_type(text_content)
            comm_role = get_community_role(text_content)
            
            # Generate name based on analysis
            if comm_role and comm_type:
                label = f"{comm_role} {comm_type}"
            elif comm_type:
                prefix = random.choice([
                    "شبکه", "گروه", "کانال", "انجمن", "محفل"
                ])
                label = f"{prefix} {comm_type}"
            else:
                # Fallback to center-based naming
                if node_label_map and (name := node_label_map.get(center)):
                    label = name
                elif name := name_community_from_center(center):
                    label = name
                else:
                    label = f"گروه {comm_id}"
        else:
            label = f"گروه {comm_id}"
        
        # Get active members
        active = sorted(
            members,
            key=lambda x: g.degree(x),
            reverse=True
        )[:3]
        
        # Store community info
        comm_info[comm_id] = {
            'label': label,
            'size': len(members),
            'center': center,
            'members': members,
            'active': active
        }
    
    # Generate colors using golden angle
    golden_angle = 0.618033988749895
    colors = {}  # Will map labels to colors
    
    # Sort communities by size
    sorted_comms = sorted(
        comm_info.items(),
        key=lambda x: x[1]['size'],
        reverse=True
    )
    
    # Assign colors and style nodes
    for i, (comm_id, info) in enumerate(sorted_comms):
        # Generate distinct color
        hue = (i * 360 * golden_angle) % 360
        saturation = min(70 + (info['size'] * 1.5), 90)
        lightness = 55
        
        color = f"hsl({hue},{saturation}%,{lightness}%)"
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
    network.options.update({
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
    })
    
    return colors
