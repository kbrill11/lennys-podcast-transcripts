#!/usr/bin/env python3
"""
Lenny's Podcast Theme Visualizer

Generate word cloud visualizations from podcast transcript metadata.
Supports multiple visualization styles, color schemes, and output formats.

Requirements:
    pip install wordcloud matplotlib numpy pillow

Usage:
    python visualize_themes.py [OPTIONS]

Examples:
    python visualize_themes.py --style themes
    python visualize_themes.py --style topics --colormap plasma --output my_wordcloud.png
    python visualize_themes.py --style all --format svg
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional
from dataclasses import dataclass

try:
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"Error: Missing required dependency - {e}")
    print("\nInstall dependencies with:")
    print("  pip install wordcloud matplotlib numpy pillow")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_EPISODES_DIR = "episodes"
METADATA_FILENAME = "metadata.json"
AGGREGATE_FILENAME = "themes_aggregate.json"

# Available colormaps for word clouds
COLORMAPS = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'Blues', 'Greens', 'Oranges', 'Purples', 'Reds',
    'YlOrRd', 'YlGnBu', 'RdYlBu', 'Spectral', 'coolwarm',
    'twilight', 'ocean', 'terrain', 'rainbow'
]

# Custom color functions
CUSTOM_PALETTES = {
    'podcast': ['#1DB954', '#191414', '#535353', '#B3B3B3', '#FFFFFF'],  # Spotify-inspired
    'tech': ['#00D4FF', '#7B2CBF', '#E500A4', '#FF6D00', '#00E676'],
    'warm': ['#FF6B6B', '#FFA07A', '#FFD93D', '#FF8C42', '#FF5252'],
    'cool': ['#4ECDC4', '#45B7D1', '#96CEB4', '#88D8B0', '#5DADE2'],
    'mono': ['#2C3E50', '#34495E', '#7F8C8D', '#95A5A6', '#BDC3C7'],
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VisualizationConfig:
    """Configuration for word cloud generation."""
    width: int = 1600
    height: int = 800
    background_color: str = 'white'
    colormap: str = 'viridis'
    max_words: int = 150
    min_font_size: int = 10
    max_font_size: int = 120
    prefer_horizontal: float = 0.7
    relative_scaling: float = 0.5
    margin: int = 10
    contour_width: int = 0
    contour_color: str = 'black'


# ============================================================================
# Data Loading
# ============================================================================

def load_aggregate_data(filepath: Path) -> Optional[dict]:
    """Load pre-aggregated theme data."""
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


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


def aggregate_from_metadata(metadata_list: list[dict]) -> dict:
    """Aggregate themes and topics from individual metadata files."""
    theme_counter = Counter()
    topic_counter = Counter()
    framework_counter = Counter()
    expertise_counter = Counter()
    insight_counter = Counter()
    guest_counter = Counter()

    for metadata in metadata_list:
        # Count themes
        for theme in metadata.get('themes', []):
            theme_counter[theme.lower().strip()] += 1

        # Count topics
        for topic in metadata.get('topics_discussed', []):
            topic_counter[topic.lower().strip()] += 1

        # Count frameworks
        for framework in metadata.get('frameworks_mentioned', []):
            framework_counter[framework.lower().strip()] += 1

        # Count expertise areas
        bg = metadata.get('guest_background', {})
        if bg:
            for area in bg.get('expertise_areas', []):
                expertise_counter[area.lower().strip()] += 1

        # Count insights (simplified)
        for insight in metadata.get('key_insights', []):
            # Extract key phrases from insights
            words = insight.lower().split()
            for word in words:
                if len(word) > 4:
                    insight_counter[word] += 1

        # Count guests
        guest = metadata.get('guest', '')
        if guest:
            guest_counter[guest] += 1

    return {
        'theme_counts': dict(theme_counter.most_common(200)),
        'topic_counts': dict(topic_counter.most_common(300)),
        'framework_counts': dict(framework_counter.most_common(100)),
        'guest_expertise_areas': dict(expertise_counter.most_common(100)),
        'insight_words': dict(insight_counter.most_common(200)),
        'guest_counts': dict(guest_counter),
        'total_episodes': len(metadata_list)
    }


# ============================================================================
# Word Cloud Generation
# ============================================================================

def create_color_func(palette_name: str):
    """Create a color function from a custom palette."""
    if palette_name not in CUSTOM_PALETTES:
        return None

    colors = CUSTOM_PALETTES[palette_name]

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return colors[hash(word) % len(colors)]

    return color_func


def generate_wordcloud(
    word_freq: dict,
    config: VisualizationConfig,
    custom_palette: Optional[str] = None,
    mask: Optional[np.ndarray] = None
) -> WordCloud:
    """Generate a word cloud from word frequencies."""

    # Build WordCloud parameters
    wc_params = {
        'width': config.width,
        'height': config.height,
        'background_color': config.background_color,
        'max_words': config.max_words,
        'min_font_size': config.min_font_size,
        'max_font_size': config.max_font_size,
        'prefer_horizontal': config.prefer_horizontal,
        'relative_scaling': config.relative_scaling,
        'margin': config.margin,
        'contour_width': config.contour_width,
        'contour_color': config.contour_color,
    }

    # Add mask if provided
    if mask is not None:
        wc_params['mask'] = mask
        wc_params['contour_width'] = 2

    # Set colormap or custom color function
    if custom_palette and custom_palette in CUSTOM_PALETTES:
        wc_params['color_func'] = create_color_func(custom_palette)
    else:
        wc_params['colormap'] = config.colormap

    # Create and generate
    wordcloud = WordCloud(**wc_params)
    wordcloud.generate_from_frequencies(word_freq)

    return wordcloud


def get_word_frequencies(data: dict, style: str) -> dict:
    """Get word frequencies based on visualization style."""

    if style == 'themes':
        return data.get('theme_counts', {})

    elif style == 'topics':
        return data.get('topic_counts', {})

    elif style == 'frameworks':
        return data.get('framework_counts', {})

    elif style == 'expertise':
        return data.get('guest_expertise_areas', {})

    elif style == 'insights':
        return data.get('insight_words', {})

    elif style == 'combined':
        # Combine themes, topics, and frameworks with different weights
        combined = {}

        for theme, count in data.get('theme_counts', {}).items():
            combined[theme] = count * 3  # Weight themes more

        for topic, count in data.get('topic_counts', {}).items():
            if topic in combined:
                combined[topic] += count
            else:
                combined[topic] = count

        for framework, count in data.get('framework_counts', {}).items():
            if framework in combined:
                combined[framework] += count * 2
            else:
                combined[framework] = count * 2

        return combined

    elif style == 'all':
        # Everything combined
        combined = {}

        for key in ['theme_counts', 'topic_counts', 'framework_counts', 'guest_expertise_areas']:
            for word, count in data.get(key, {}).items():
                if word in combined:
                    combined[word] += count
                else:
                    combined[word] = count

        return combined

    else:
        print(f"Unknown style: {style}. Using 'combined'.")
        return get_word_frequencies(data, 'combined')


# ============================================================================
# Visualization Functions
# ============================================================================

def save_wordcloud(
    wordcloud: WordCloud,
    output_path: Path,
    title: Optional[str] = None,
    figsize: tuple = (20, 10),
    dpi: int = 150,
    file_format: str = 'png'
):
    """Save word cloud to file with optional title."""

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=24, pad=20, fontweight='bold')

    plt.tight_layout(pad=0)

    # Ensure correct extension
    output_path = output_path.with_suffix(f'.{file_format}')

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', format=file_format)
    plt.close()

    return output_path


def create_multi_panel_visualization(
    data: dict,
    config: VisualizationConfig,
    output_path: Path,
    dpi: int = 150
):
    """Create a multi-panel visualization with different word clouds."""

    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.suptitle("Lenny's Podcast - Theme Analysis", fontsize=28, fontweight='bold', y=1.02)

    panels = [
        ('themes', 'Main Themes', 'viridis'),
        ('topics', 'Topics Discussed', 'plasma'),
        ('frameworks', 'Frameworks & Methods', 'inferno'),
        ('expertise', 'Guest Expertise Areas', 'cividis'),
    ]

    for ax, (style, title, cmap) in zip(axes.flat, panels):
        word_freq = get_word_frequencies(data, style)

        if not word_freq:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=16)
            ax.set_title(title, fontsize=18, fontweight='bold')
            ax.axis('off')
            continue

        panel_config = VisualizationConfig(
            width=800,
            height=400,
            colormap=cmap,
            max_words=75,
            background_color=config.background_color
        )

        wc = generate_wordcloud(word_freq, panel_config)
        ax.imshow(wc, interpolation='bilinear')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def create_comparison_visualization(
    data: dict,
    config: VisualizationConfig,
    output_path: Path,
    dpi: int = 150
):
    """Create side-by-side comparison of themes vs topics."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle("Themes vs Topics Comparison", fontsize=24, fontweight='bold', y=1.02)

    # Themes
    theme_freq = get_word_frequencies(data, 'themes')
    if theme_freq:
        theme_config = VisualizationConfig(
            width=800, height=600, colormap='YlOrRd', max_words=100
        )
        wc1 = generate_wordcloud(theme_freq, theme_config)
        ax1.imshow(wc1, interpolation='bilinear')
    ax1.set_title('High-Level Themes', fontsize=18, fontweight='bold')
    ax1.axis('off')

    # Topics
    topic_freq = get_word_frequencies(data, 'topics')
    if topic_freq:
        topic_config = VisualizationConfig(
            width=800, height=600, colormap='YlGnBu', max_words=100
        )
        wc2 = generate_wordcloud(topic_freq, topic_config)
        ax2.imshow(wc2, interpolation='bilinear')
    ax2.set_title('Specific Topics', fontsize=18, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def create_circular_wordcloud(
    word_freq: dict,
    config: VisualizationConfig,
    output_path: Path,
    dpi: int = 150
):
    """Create a circular word cloud using a mask."""

    # Create circular mask
    x, y = np.ogrid[:800, :800]
    center = 400
    mask = (x - center) ** 2 + (y - center) ** 2 > center ** 2
    mask = 255 * mask.astype(int)

    # Update config for circular layout
    circular_config = VisualizationConfig(
        width=800,
        height=800,
        colormap=config.colormap,
        max_words=config.max_words,
        background_color=config.background_color,
        prefer_horizontal=0.5  # More varied orientation for circular
    )

    wc = generate_wordcloud(word_freq, circular_config, mask=mask)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title("Lenny's Podcast Themes", fontsize=24, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def print_statistics(data: dict):
    """Print statistics about the aggregated data."""

    print("\n" + "=" * 60)
    print("THEME STATISTICS")
    print("=" * 60)

    total = data.get('total_episodes', 0)
    print(f"\nTotal episodes analyzed: {total}")

    theme_counts = data.get('theme_counts', {})
    topic_counts = data.get('topic_counts', {})
    framework_counts = data.get('framework_counts', {})

    print(f"Unique themes: {len(theme_counts)}")
    print(f"Unique topics: {len(topic_counts)}")
    print(f"Unique frameworks: {len(framework_counts)}")

    if theme_counts:
        print("\nTop 15 Themes:")
        for theme, count in list(theme_counts.items())[:15]:
            bar = "█" * min(count, 30)
            print(f"  {theme:40} {count:3} {bar}")

    if framework_counts:
        print("\nTop 10 Frameworks:")
        for fw, count in list(framework_counts.items())[:10]:
            print(f"  {fw:40} {count:3}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate word cloud visualizations from podcast metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Styles:
  themes      - High-level episode themes
  topics      - Specific topics discussed
  frameworks  - Frameworks and methodologies mentioned
  expertise   - Guest expertise areas
  insights    - Key words from insights
  combined    - Weighted combination of themes, topics, frameworks
  all         - Everything combined equally
  multi       - Multi-panel visualization (4 word clouds)
  comparison  - Side-by-side themes vs topics
  circular    - Circular word cloud

Color Palettes:
  Built-in matplotlib colormaps: viridis, plasma, inferno, magma, etc.
  Custom palettes: podcast, tech, warm, cool, mono
        """
    )

    parser.add_argument(
        '--episodes-dir',
        type=str,
        default=DEFAULT_EPISODES_DIR,
        help='Path to episodes directory'
    )
    parser.add_argument(
        '--style',
        type=str,
        default='combined',
        choices=['themes', 'topics', 'frameworks', 'expertise', 'insights',
                 'combined', 'all', 'multi', 'comparison', 'circular'],
        help='Visualization style (default: combined)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output filename (default: wordcloud_<style>.png)'
    )
    parser.add_argument(
        '--colormap', '-c',
        type=str,
        default='viridis',
        help=f'Matplotlib colormap or custom palette (default: viridis)'
    )
    parser.add_argument(
        '--background',
        type=str,
        default='white',
        help='Background color (default: white)'
    )
    parser.add_argument(
        '--max-words',
        type=int,
        default=150,
        help='Maximum number of words (default: 150)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1600,
        help='Image width in pixels (default: 1600)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=800,
        help='Image height in pixels (default: 800)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='Output DPI (default: 150)'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='png',
        choices=['png', 'svg', 'pdf', 'jpg'],
        help='Output format (default: png)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print theme statistics'
    )
    parser.add_argument(
        '--list-colormaps',
        action='store_true',
        help='List available colormaps'
    )

    args = parser.parse_args()

    # List colormaps and exit
    if args.list_colormaps:
        print("Available Matplotlib Colormaps:")
        for cm in COLORMAPS:
            print(f"  {cm}")
        print("\nCustom Palettes:")
        for name, colors in CUSTOM_PALETTES.items():
            print(f"  {name}: {colors}")
        sys.exit(0)

    # Resolve paths
    script_dir = Path(__file__).parent
    episodes_dir = script_dir / args.episodes_dir

    # Try to load aggregate data first, fall back to individual files
    aggregate_path = script_dir / AGGREGATE_FILENAME
    data = load_aggregate_data(aggregate_path)

    if not data:
        print(f"Aggregate file not found at {aggregate_path}")
        print("Loading metadata from individual episode files...")

        if not episodes_dir.exists():
            print(f"Error: Episodes directory not found: {episodes_dir}")
            sys.exit(1)

        metadata_list = load_all_metadata(episodes_dir)

        if not metadata_list:
            print("Error: No metadata files found. Run analyze_transcripts.py first.")
            sys.exit(1)

        print(f"Loaded metadata from {len(metadata_list)} episodes")
        data = aggregate_from_metadata(metadata_list)
    else:
        print(f"Loaded aggregate data: {data.get('total_episodes', 0)} episodes")

    # Print statistics if requested
    if args.stats:
        print_statistics(data)

    # Create configuration
    config = VisualizationConfig(
        width=args.width,
        height=args.height,
        background_color=args.background,
        colormap=args.colormap,
        max_words=args.max_words
    )

    # Determine output path
    if args.output:
        output_path = script_dir / args.output
    else:
        output_path = script_dir / f"wordcloud_{args.style}.{args.format}"

    # Generate visualization based on style
    print(f"\nGenerating {args.style} word cloud...")

    if args.style == 'multi':
        result_path = create_multi_panel_visualization(data, config, output_path, args.dpi)

    elif args.style == 'comparison':
        result_path = create_comparison_visualization(data, config, output_path, args.dpi)

    elif args.style == 'circular':
        word_freq = get_word_frequencies(data, 'combined')
        result_path = create_circular_wordcloud(word_freq, config, output_path, args.dpi)

    else:
        word_freq = get_word_frequencies(data, args.style)

        if not word_freq:
            print(f"Error: No data available for style '{args.style}'")
            sys.exit(1)

        # Check for custom palette
        custom_palette = args.colormap if args.colormap in CUSTOM_PALETTES else None

        wc = generate_wordcloud(word_freq, config, custom_palette=custom_palette)

        title = f"Lenny's Podcast - {args.style.title()}"
        result_path = save_wordcloud(
            wc, output_path, title=title, dpi=args.dpi, file_format=args.format
        )

    print(f"Word cloud saved to: {result_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
