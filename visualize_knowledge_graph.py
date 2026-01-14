#!/usr/bin/env python3
"""
Lenny's Podcast Knowledge Graph Visualizer

Generate interactive knowledge graph visualizations showing relationships
between themes, episodes, and guests with frequency-based node sizing.

Requirements:
    pip install networkx pyvis matplotlib

Usage:
    python visualize_knowledge_graph.py [OPTIONS]

Examples:
    python visualize_knowledge_graph.py --mode themes
    python visualize_knowledge_graph.py --mode episodes --min-shared-themes 3
    python visualize_knowledge_graph.py --mode guests --output guest_network.html
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional

try:
    import networkx as nx
    from pyvis.network import Network
except ImportError as e:
    print(f"Error: Missing required dependency - {e}")
    print("\nInstall dependencies with:")
    print("  pip install networkx pyvis")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_EPISODES_DIR = "episodes"
METADATA_FILENAME = "metadata.json"

# Color palettes for different node types
COLORS = {
    'theme': {
        'background': '#6366f1',  # Indigo
        'border': '#4f46e5',
        'highlight': '#818cf8'
    },
    'episode': {
        'background': '#10b981',  # Emerald
        'border': '#059669',
        'highlight': '#34d399'
    },
    'guest': {
        'background': '#f59e0b',  # Amber
        'border': '#d97706',
        'highlight': '#fbbf24'
    },
    'topic': {
        'background': '#ec4899',  # Pink
        'border': '#db2777',
        'highlight': '#f472b6'
    },
    'framework': {
        'background': '#8b5cf6',  # Violet
        'border': '#7c3aed',
        'highlight': '#a78bfa'
    }
}

# Category color mapping for themes
CATEGORY_COLORS = [
    '#6366f1', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6',
    '#14b8a6', '#f97316', '#06b6d4', '#84cc16', '#e11d48'
]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GraphConfig:
    """Configuration for graph visualization."""
    width: str = "100%"
    height: str = "800px"
    background_color: str = "#ffffff"
    font_color: str = "#333333"
    edge_color: str = "#cccccc"
    physics_enabled: bool = True
    physics_solver: str = "forceAtlas2Based"
    min_node_size: int = 10
    max_node_size: int = 60
    min_edge_width: float = 0.5
    max_edge_width: float = 8.0


# ============================================================================
# Data Loading
# ============================================================================

def load_all_metadata(episodes_dir: Path) -> list[dict]:
    """Load metadata from all episode directories."""
    metadata_list = []

    for episode_dir in sorted(episodes_dir.iterdir()):
        if not episode_dir.is_dir():
            continue
        metadata_path = episode_dir / METADATA_FILENAME
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_list.append(json.load(f))

    return metadata_list


# ============================================================================
# Graph Building Functions
# ============================================================================

def build_theme_graph(
    metadata_list: list[dict],
    min_occurrences: int = 2,
    min_co_occurrence: int = 2,
    max_nodes: int = 75
) -> nx.Graph:
    """
    Build a graph where nodes are themes and edges represent co-occurrence.

    Node size = frequency of theme across episodes
    Edge weight = number of episodes where themes appear together
    """
    G = nx.Graph()

    # Count theme occurrences
    theme_counts = Counter()
    theme_episodes = defaultdict(list)

    for metadata in metadata_list:
        episode_info = {
            'id': metadata.get('episode_id', ''),
            'guest': metadata.get('guest', ''),
            'title': metadata.get('title', '')
        }
        for theme in metadata.get('themes', []):
            theme_clean = theme.lower().strip()
            theme_counts[theme_clean] += 1
            theme_episodes[theme_clean].append(episode_info)

    # Filter themes by minimum occurrences
    valid_themes = {t for t, c in theme_counts.items() if c >= min_occurrences}

    # Limit to top N themes
    top_themes = [t for t, _ in theme_counts.most_common(max_nodes) if t in valid_themes]

    # Calculate co-occurrences
    co_occurrence = defaultdict(lambda: defaultdict(int))

    for metadata in metadata_list:
        themes = [t.lower().strip() for t in metadata.get('themes', []) if t.lower().strip() in top_themes]
        for i, theme1 in enumerate(themes):
            for theme2 in themes[i+1:]:
                co_occurrence[theme1][theme2] += 1
                co_occurrence[theme2][theme1] += 1

    # Add nodes
    for theme in top_themes:
        G.add_node(
            theme,
            count=theme_counts[theme],
            episodes=theme_episodes[theme][:10],  # Limit for tooltip
            node_type='theme'
        )

    # Add edges
    for theme1 in top_themes:
        for theme2, weight in co_occurrence[theme1].items():
            if theme2 in top_themes and weight >= min_co_occurrence:
                if not G.has_edge(theme1, theme2):
                    G.add_edge(theme1, theme2, weight=weight)

    return G


def build_episode_graph(
    metadata_list: list[dict],
    min_shared_themes: int = 2,
    max_nodes: int = 100
) -> nx.Graph:
    """
    Build a graph where nodes are episodes and edges represent shared themes.

    Node size = number of themes in episode
    Edge weight = number of shared themes
    """
    G = nx.Graph()

    # Limit episodes if needed
    episodes = metadata_list[:max_nodes]

    # Add episode nodes
    for metadata in episodes:
        episode_id = metadata.get('episode_id', '')
        themes = [t.lower().strip() for t in metadata.get('themes', [])]

        G.add_node(
            episode_id,
            label=metadata.get('guest', episode_id),
            title=metadata.get('title', ''),
            guest=metadata.get('guest', ''),
            theme_count=len(themes),
            themes=themes,
            summary=metadata.get('summary', ''),
            node_type='episode'
        )

    # Calculate shared themes between episodes
    for i, meta1 in enumerate(episodes):
        ep1 = meta1.get('episode_id', '')
        themes1 = set(t.lower().strip() for t in meta1.get('themes', []))

        for meta2 in episodes[i+1:]:
            ep2 = meta2.get('episode_id', '')
            themes2 = set(t.lower().strip() for t in meta2.get('themes', []))

            shared = themes1 & themes2
            if len(shared) >= min_shared_themes:
                G.add_edge(ep1, ep2, weight=len(shared), shared_themes=list(shared))

    return G


def build_guest_graph(
    metadata_list: list[dict],
    min_shared_topics: int = 3,
    max_nodes: int = 100
) -> nx.Graph:
    """
    Build a graph where nodes are guests and edges represent topic overlap.

    Node size = number of topics discussed
    Edge weight = topic similarity
    """
    G = nx.Graph()

    # Group by guest (handle potential duplicates)
    guest_data = {}

    for metadata in metadata_list:
        guest = metadata.get('guest', 'Unknown')
        if not guest or guest == 'Unknown':
            continue

        if guest not in guest_data:
            guest_data[guest] = {
                'topics': set(),
                'themes': set(),
                'expertise': set(),
                'episodes': [],
                'companies': set()
            }

        data = guest_data[guest]
        data['topics'].update(t.lower().strip() for t in metadata.get('topics_discussed', []))
        data['themes'].update(t.lower().strip() for t in metadata.get('themes', []))

        bg = metadata.get('guest_background', {})
        if bg:
            data['expertise'].update(e.lower().strip() for e in bg.get('expertise_areas', []))
            data['companies'].update(bg.get('notable_companies', []))

        data['episodes'].append({
            'id': metadata.get('episode_id', ''),
            'title': metadata.get('title', '')
        })

    # Add nodes (limit to max)
    guests = list(guest_data.keys())[:max_nodes]

    for guest in guests:
        data = guest_data[guest]
        G.add_node(
            guest,
            topic_count=len(data['topics']),
            expertise=list(data['expertise'])[:5],
            companies=list(data['companies'])[:5],
            episodes=data['episodes'],
            node_type='guest'
        )

    # Add edges based on topic overlap
    for i, guest1 in enumerate(guests):
        topics1 = guest_data[guest1]['topics'] | guest_data[guest1]['themes']

        for guest2 in guests[i+1:]:
            topics2 = guest_data[guest2]['topics'] | guest_data[guest2]['themes']

            shared = topics1 & topics2
            if len(shared) >= min_shared_topics:
                G.add_edge(guest1, guest2, weight=len(shared), shared_topics=list(shared)[:10])

    return G


def build_mixed_graph(
    metadata_list: list[dict],
    include_themes: bool = True,
    include_episodes: bool = True,
    max_themes: int = 30,
    max_episodes: int = 50,
    min_theme_occurrences: int = 3
) -> nx.Graph:
    """
    Build a bipartite-style graph with both themes and episodes.

    Shows which episodes contain which themes.
    """
    G = nx.Graph()

    # Count themes
    theme_counts = Counter()
    for metadata in metadata_list:
        for theme in metadata.get('themes', []):
            theme_counts[theme.lower().strip()] += 1

    # Get top themes
    top_themes = [t for t, c in theme_counts.most_common(max_themes) if c >= min_theme_occurrences]

    # Add theme nodes
    if include_themes:
        for theme in top_themes:
            G.add_node(
                f"theme:{theme}",
                label=theme,
                count=theme_counts[theme],
                node_type='theme'
            )

    # Add episode nodes and edges to themes
    episodes = metadata_list[:max_episodes]

    for metadata in episodes:
        episode_id = metadata.get('episode_id', '')

        if include_episodes:
            G.add_node(
                f"episode:{episode_id}",
                label=metadata.get('guest', episode_id),
                title=metadata.get('title', ''),
                node_type='episode'
            )

        # Connect episodes to their themes
        if include_themes and include_episodes:
            for theme in metadata.get('themes', []):
                theme_clean = theme.lower().strip()
                if theme_clean in top_themes:
                    G.add_edge(f"episode:{episode_id}", f"theme:{theme_clean}", weight=1)

    return G


# ============================================================================
# Visualization Functions
# ============================================================================

def calculate_node_size(value: float, min_val: float, max_val: float, config: GraphConfig) -> int:
    """Scale a value to node size range."""
    if max_val == min_val:
        return (config.min_node_size + config.max_node_size) // 2

    normalized = (value - min_val) / (max_val - min_val)
    return int(config.min_node_size + normalized * (config.max_node_size - config.min_node_size))


def calculate_edge_width(weight: float, min_weight: float, max_weight: float, config: GraphConfig) -> float:
    """Scale a weight to edge width range."""
    if max_weight == min_weight:
        return (config.min_edge_width + config.max_edge_width) / 2

    normalized = (weight - min_weight) / (max_weight - min_weight)
    return config.min_edge_width + normalized * (config.max_edge_width - config.min_edge_width)


def create_tooltip(node_id: str, node_data: dict, node_type: str) -> str:
    """Create HTML tooltip for a node."""
    if node_type == 'theme':
        episodes = node_data.get('episodes', [])
        episode_list = '<br>'.join(f"• {e['guest']}" for e in episodes[:5])
        if len(episodes) > 5:
            episode_list += f"<br>... and {len(episodes) - 5} more"

        return f"""
        <div style="max-width: 300px;">
            <strong style="font-size: 14px;">{node_id}</strong><br>
            <em>Appears in {node_data.get('count', 0)} episodes</em><br><br>
            <strong>Episodes:</strong><br>
            {episode_list}
        </div>
        """

    elif node_type == 'episode':
        themes = node_data.get('themes', [])
        theme_list = ', '.join(themes[:5])
        if len(themes) > 5:
            theme_list += f", +{len(themes) - 5} more"

        return f"""
        <div style="max-width: 350px;">
            <strong style="font-size: 14px;">{node_data.get('guest', node_id)}</strong><br>
            <em>{node_data.get('title', '')[:80]}...</em><br><br>
            <strong>Themes:</strong> {theme_list}<br><br>
            <strong>Summary:</strong> {node_data.get('summary', '')[:200]}...
        </div>
        """

    elif node_type == 'guest':
        expertise = ', '.join(node_data.get('expertise', [])[:4])
        companies = ', '.join(node_data.get('companies', [])[:4])
        episodes = node_data.get('episodes', [])

        return f"""
        <div style="max-width: 300px;">
            <strong style="font-size: 14px;">{node_id}</strong><br><br>
            <strong>Expertise:</strong> {expertise or 'N/A'}<br>
            <strong>Companies:</strong> {companies or 'N/A'}<br>
            <strong>Episodes:</strong> {len(episodes)}<br>
            <strong>Topics discussed:</strong> {node_data.get('topic_count', 0)}
        </div>
        """

    return str(node_id)


def visualize_graph(
    G: nx.Graph,
    output_path: Path,
    config: GraphConfig,
    title: str = "Knowledge Graph"
) -> Path:
    """Render a NetworkX graph to interactive HTML using PyVis."""

    # Create PyVis network
    net = Network(
        height=config.height,
        width=config.width,
        bgcolor=config.background_color,
        font_color=config.font_color,
        heading=title
    )

    # Get size scaling values
    node_types = nx.get_node_attributes(G, 'node_type')

    # Determine sizing attribute based on node type
    size_values = []
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data.get('node_type') == 'theme':
            size_values.append(node_data.get('count', 1))
        elif node_data.get('node_type') == 'episode':
            size_values.append(node_data.get('theme_count', 1))
        elif node_data.get('node_type') == 'guest':
            size_values.append(node_data.get('topic_count', 1))
        else:
            size_values.append(1)

    min_size_val = min(size_values) if size_values else 1
    max_size_val = max(size_values) if size_values else 1

    # Get edge weight scaling
    edge_weights = [d.get('weight', 1) for _, _, d in G.edges(data=True)]
    min_weight = min(edge_weights) if edge_weights else 1
    max_weight = max(edge_weights) if edge_weights else 1

    # Add nodes
    for i, node in enumerate(G.nodes()):
        node_data = G.nodes[node]
        node_type = node_data.get('node_type', 'theme')

        # Get color based on type
        color = COLORS.get(node_type, COLORS['theme'])['background']

        # Calculate size
        if node_type == 'theme':
            size_val = node_data.get('count', 1)
        elif node_type == 'episode':
            size_val = node_data.get('theme_count', 1)
        elif node_type == 'guest':
            size_val = node_data.get('topic_count', 1)
        else:
            size_val = 1

        size = calculate_node_size(size_val, min_size_val, max_size_val, config)

        # Get label
        label = node_data.get('label', node)
        if label.startswith('theme:') or label.startswith('episode:'):
            label = label.split(':', 1)[1]

        # Truncate long labels
        if len(label) > 25:
            label = label[:22] + "..."

        # Create tooltip
        tooltip = create_tooltip(node, node_data, node_type)

        net.add_node(
            node,
            label=label,
            size=size,
            color=color,
            title=tooltip,
            borderWidth=2,
            borderWidthSelected=4
        )

    # Add edges
    for source, target, edge_data in G.edges(data=True):
        weight = edge_data.get('weight', 1)
        width = calculate_edge_width(weight, min_weight, max_weight, config)

        # Create edge tooltip
        shared = edge_data.get('shared_themes', edge_data.get('shared_topics', []))
        if shared:
            edge_title = f"Shared: {', '.join(shared[:5])}"
            if len(shared) > 5:
                edge_title += f" +{len(shared) - 5} more"
        else:
            edge_title = f"Connection strength: {weight}"

        net.add_edge(
            source,
            target,
            width=width,
            color=config.edge_color,
            title=edge_title
        )

    # Configure physics
    if config.physics_enabled:
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08,
                    "damping": 0.4,
                    "avoidOverlap": 0.5
                },
                "stabilization": {
                    "enabled": true,
                    "iterations": 200,
                    "updateInterval": 25
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 100,
                "hideEdgesOnDrag": true,
                "multiselect": true
            },
            "nodes": {
                "font": {
                    "size": 14,
                    "face": "arial"
                },
                "scaling": {
                    "label": {
                        "enabled": true,
                        "min": 10,
                        "max": 20
                    }
                }
            },
            "edges": {
                "smooth": {
                    "enabled": true,
                    "type": "continuous"
                },
                "color": {
                    "inherit": false,
                    "opacity": 0.6
                }
            }
        }
        """)
    else:
        net.toggle_physics(False)

    # Save
    output_path = output_path.with_suffix('.html')
    net.save_graph(str(output_path))

    return output_path


def print_graph_stats(G: nx.Graph, mode: str):
    """Print statistics about the graph."""
    print(f"\n{'='*60}")
    print(f"GRAPH STATISTICS ({mode.upper()} MODE)")
    print('='*60)
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        avg_degree = sum(degrees.values()) / len(degrees)
        print(f"Average degree: {avg_degree:.2f}")

        # Top connected nodes
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 most connected nodes:")
        for node, degree in top_nodes:
            label = node
            if len(label) > 40:
                label = label[:37] + "..."
            print(f"  {label:42} {degree:3} connections")

    # Connected components
    if G.number_of_nodes() > 0:
        components = list(nx.connected_components(G))
        print(f"\nConnected components: {len(components)}")
        if len(components) > 1:
            sizes = sorted([len(c) for c in components], reverse=True)
            print(f"Component sizes: {sizes[:5]}{'...' if len(sizes) > 5 else ''}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate knowledge graph visualizations from podcast metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  themes    - Themes as nodes, co-occurrence as edges (default)
  episodes  - Episodes as nodes, shared themes as edges
  guests    - Guests as nodes, topic overlap as edges
  mixed     - Both themes and episodes (bipartite graph)

Examples:
  python visualize_knowledge_graph.py --mode themes
  python visualize_knowledge_graph.py --mode episodes --min-shared 3
  python visualize_knowledge_graph.py --mode guests -o guest_network.html
        """
    )

    parser.add_argument(
        '--episodes-dir',
        type=str,
        default=DEFAULT_EPISODES_DIR,
        help='Path to episodes directory'
    )
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default='themes',
        choices=['themes', 'episodes', 'guests', 'mixed'],
        help='Graph visualization mode (default: themes)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output filename (default: knowledge_graph_<mode>.html)'
    )
    parser.add_argument(
        '--min-occurrences',
        type=int,
        default=2,
        help='Minimum theme occurrences to include (themes mode, default: 2)'
    )
    parser.add_argument(
        '--min-shared',
        type=int,
        default=2,
        help='Minimum shared themes/topics for edge (default: 2)'
    )
    parser.add_argument(
        '--max-nodes',
        type=int,
        default=75,
        help='Maximum number of nodes to display (default: 75)'
    )
    parser.add_argument(
        '--no-physics',
        action='store_true',
        help='Disable physics simulation (static layout)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print graph statistics'
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    episodes_dir = script_dir / args.episodes_dir

    if not episodes_dir.exists():
        print(f"Error: Episodes directory not found: {episodes_dir}")
        sys.exit(1)

    # Load metadata
    print(f"Loading metadata from: {episodes_dir}")
    metadata_list = load_all_metadata(episodes_dir)

    if not metadata_list:
        print("Error: No metadata files found. Run analyze_transcripts.py first.")
        sys.exit(1)

    print(f"Loaded metadata for {len(metadata_list)} episodes")

    # Build graph based on mode
    print(f"\nBuilding {args.mode} graph...")

    config = GraphConfig(
        physics_enabled=not args.no_physics
    )

    if args.mode == 'themes':
        G = build_theme_graph(
            metadata_list,
            min_occurrences=args.min_occurrences,
            min_co_occurrence=args.min_shared,
            max_nodes=args.max_nodes
        )
        title = "Lenny's Podcast - Theme Network"

    elif args.mode == 'episodes':
        G = build_episode_graph(
            metadata_list,
            min_shared_themes=args.min_shared,
            max_nodes=args.max_nodes
        )
        title = "Lenny's Podcast - Episode Connections"

    elif args.mode == 'guests':
        G = build_guest_graph(
            metadata_list,
            min_shared_topics=args.min_shared,
            max_nodes=args.max_nodes
        )
        title = "Lenny's Podcast - Guest Network"

    elif args.mode == 'mixed':
        G = build_mixed_graph(
            metadata_list,
            max_themes=min(30, args.max_nodes // 2),
            max_episodes=min(50, args.max_nodes // 2)
        )
        title = "Lenny's Podcast - Themes & Episodes"

    # Print stats if requested
    if args.stats:
        print_graph_stats(G, args.mode)

    # Check if graph has content
    if G.number_of_nodes() == 0:
        print("Warning: Graph has no nodes. Try adjusting --min-occurrences or --min-shared parameters.")
        sys.exit(0)

    # Determine output path
    if args.output:
        output_path = script_dir / args.output
    else:
        output_path = script_dir / f"knowledge_graph_{args.mode}.html"

    # Generate visualization
    print(f"\nGenerating visualization...")
    result_path = visualize_graph(G, output_path, config, title)

    print(f"\nKnowledge graph saved to: {result_path}")
    print(f"Open in a web browser to explore the interactive visualization.")
    print("\nTips:")
    print("  - Hover over nodes to see details")
    print("  - Drag nodes to rearrange")
    print("  - Scroll to zoom in/out")
    print("  - Double-click to focus on a node")


if __name__ == '__main__':
    main()
