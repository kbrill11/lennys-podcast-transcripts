#!/usr/bin/env python3
"""
Lenny's Podcast Transcript Analyzer

This script analyzes all podcast transcripts in the repository using AI to:
1. Generate themes and insights for each episode
2. Save metadata JSON files to each episode folder
3. Create a word cloud visualization of themes and relationships across all episodes

Requirements:
    pip install anthropic pyyaml wordcloud matplotlib numpy tqdm

Usage:
    python analyze_transcripts.py [--episodes-dir PATH] [--max-concurrent N] [--skip-existing]

Environment:
    ANTHROPIC_API_KEY: Your Anthropic API key for Claude
"""

import os
import sys
import json
import yaml
import asyncio
import argparse
import re
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

import anthropic
from tqdm import tqdm

# Optional imports for visualization
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: wordcloud/matplotlib not installed. Visualization will be skipped.")
    print("Install with: pip install wordcloud matplotlib numpy")


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_EPISODES_DIR = "episodes"
METADATA_FILENAME = "metadata.json"
THEMES_AGGREGATE_FILE = "themes_aggregate.json"
WORDCLOUD_OUTPUT = "themes_wordcloud.png"
RELATIONSHIPS_OUTPUT = "relationships_network.json"

# Claude model configuration
CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096


# ============================================================================
# Normalization Utilities
# ============================================================================

# Alias mappings for known equivalent terms
THEME_ALIASES = {
    # Hyphenated variants → space-separated canonical form
    "product-market fit": "product market fit",
    "jobs-to-be-done framework": "jobs to be done framework",
    "jobs-to-be-done": "jobs to be done",
    "product-led growth": "product led growth",
    "go-to-market strategy": "go to market strategy",
    "ai-powered software development": "ai powered software development",

    # Word order variants
    "hiring and team building": "team building and hiring",

    # === BUCKET: AI Product Development ===
    "ai powered software development": "ai product development",
    "building ai agents": "ai product development",
    "product management for ai": "ai product development",
    "ai evaluation strategies": "ai product development",
    "ai product lifecycle management": "ai product development",
    "enterprise ai adoption": "ai product development",
    "leadership in ai transformation": "ai product development",
    "ai integration in product development": "ai product development",
    "non deterministic systems": "ai product development",
    "agency vs control trade offs": "ai product development",
    "continuous calibration methodology": "ai product development",

    # === BUCKET: Product Management ===
    "product management career development": "product management",
    "product management evolution": "product management",
    "product management leadership": "product management",
    "product management strategy": "product management",
    "product management philosophy": "product management",
    "product management fundamentals": "product management",
    "product management in media/journalism": "product management",
    "crisis product management": "product management",

    # === BUCKET: Growth Strategies ===
    "startup growth strategies": "growth strategies",
    "growth team building": "growth strategies",
    "growth strategy execution": "growth strategies",
    "startup growth tactics": "growth strategies",
    "growth marketing strategy": "growth strategies",
    "growth cmo role evolution": "growth strategies",
    "growth strategy and optimization": "growth strategies",

    # === BUCKET: Experimentation ===
    "experimentation and testing": "experimentation",
    "experimentation frameworks": "experimentation",
    "experimentation methodology": "experimentation",
    "experimentation and a/b testing": "experimentation",

    # === BUCKET: Career Development ===
    "career advancement": "career development",
    "career strategy and decision making": "career development",
    "early career development": "career development",
    "career decision making": "career development",

    # === BUCKET: Company Culture ===
    "company culture and decision making": "company culture",
    "company culture and leadership": "company culture",

    # === BUCKET: Work-Life Balance ===
    "work life balance and priorities": "work life balance",
    "burnout vs depression": "work life balance",
    "mental health in tech": "work life balance",
}

SENTIMENT_CANONICAL = {
    "inspiring and authentic": "inspiring",
    "inspiring/tactical": "tactical",
    "deeply reflective and vulnerable": "reflective",
    "reflective and inspiring": "reflective",
}


def normalize_theme(theme: str) -> str:
    """Normalize theme to canonical form."""
    normalized = theme.lower().strip()
    # Replace hyphens between words with spaces
    normalized = re.sub(r'(?<=[a-z])-(?=[a-z])', ' ', normalized)
    # Apply explicit alias mapping
    return THEME_ALIASES.get(normalized, normalized)


def normalize_sentiment(sentiment: str) -> str:
    """Normalize sentiment to one of four canonical values."""
    s = sentiment.lower().strip()
    return SENTIMENT_CANONICAL.get(s, s)


# Analysis prompt template
ANALYSIS_PROMPT = """Analyze this podcast transcript and extract structured metadata.

<transcript>
{transcript}
</transcript>

Formatting rules for themes and frameworks:
- Use spaces instead of hyphens (e.g., "product market fit" not "product-market fit")
- Use consistent naming for well-known concepts:
  - "jobs to be done" (not "jobs-to-be-done" or "JTBD")
  - "product market fit" (not "PMF")
  - "go to market" (not "GTM")
- sentiment must be exactly one of: inspiring, tactical, reflective, conversational

Provide a JSON response with the following structure:
{{
    "themes": [
        // List of 5-10 main themes discussed (e.g., "product management", "growth strategies", "leadership")
    ],
    "key_insights": [
        // List of 5-8 specific, actionable insights from the conversation
    ],
    "guest_background": {{
        "current_role": "Current position/company",
        "expertise_areas": ["area1", "area2"],
        "notable_companies": ["company1", "company2"]
    }},
    "topics_discussed": [
        // Detailed list of specific topics covered (10-15 items)
    ],
    "memorable_quotes": [
        // 2-3 standout quotes from the episode
    ],
    "frameworks_mentioned": [
        // Any specific frameworks, methodologies, or mental models discussed
    ],
    "recommended_for": [
        // Types of professionals who would benefit most from this episode
    ],
    "sentiment": "overall tone (inspiring/tactical/reflective/conversational)",
    "difficulty_level": "beginner/intermediate/advanced",
    "summary": "A 2-3 sentence summary of the episode's main value proposition"
}}

Return ONLY the JSON object, no additional text."""


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EpisodeMetadata:
    """Metadata for a single episode."""
    episode_id: str
    guest: str
    title: str
    youtube_url: str
    duration: str
    duration_seconds: float
    view_count: int
    themes: list
    key_insights: list
    guest_background: dict
    topics_discussed: list
    memorable_quotes: list
    frameworks_mentioned: list
    recommended_for: list
    sentiment: str
    difficulty_level: str
    summary: str
    analyzed_at: str
    analyzer_version: str = "1.0.0"


@dataclass
class AggregateThemes:
    """Aggregated themes across all episodes."""
    total_episodes: int
    theme_counts: dict
    topic_counts: dict
    framework_counts: dict
    guest_expertise_areas: dict
    sentiment_distribution: dict
    difficulty_distribution: dict
    common_insights: list
    generated_at: str


# ============================================================================
# Transcript Parsing
# ============================================================================

def parse_transcript_file(file_path: Path) -> dict:
    """Parse a transcript markdown file and extract frontmatter and content."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split frontmatter from content
    parts = content.split('---', 2)
    if len(parts) >= 3:
        frontmatter_str = parts[1].strip()
        transcript_content = parts[2].strip()

        try:
            frontmatter = yaml.safe_load(frontmatter_str)
        except yaml.YAMLError as e:
            print(f"Warning: Could not parse frontmatter in {file_path}: {e}")
            frontmatter = {}
    else:
        frontmatter = {}
        transcript_content = content

    return {
        'frontmatter': frontmatter,
        'content': transcript_content,
        'file_path': str(file_path)
    }


def truncate_transcript(content: str, max_chars: int = 150000) -> str:
    """Truncate transcript if too long, keeping beginning and end."""
    if len(content) <= max_chars:
        return content

    # Keep 60% from beginning, 40% from end
    begin_chars = int(max_chars * 0.6)
    end_chars = max_chars - begin_chars - 100  # Leave room for indicator

    return (
        content[:begin_chars] +
        "\n\n[... transcript truncated for length ...]\n\n" +
        content[-end_chars:]
    )


# ============================================================================
# AI Analysis
# ============================================================================

class TranscriptAnalyzer:
    """Analyzes transcripts using Claude AI."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it as an environment variable or pass it directly."
            )
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def analyze_episode(self, transcript_data: dict) -> dict:
        """Analyze a single episode transcript."""
        content = truncate_transcript(transcript_data['content'])
        frontmatter = transcript_data['frontmatter']

        # Include frontmatter context in the transcript
        context = f"""Episode: {frontmatter.get('title', 'Unknown')}
Guest: {frontmatter.get('guest', 'Unknown')}
Duration: {frontmatter.get('duration', 'Unknown')}

{content}"""

        prompt = ANALYSIS_PROMPT.format(transcript=context)

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract JSON from response
            response_text = response.content[0].text

            # Try to parse JSON (handle potential markdown code blocks)
            json_str = response_text
            if '```json' in response_text:
                json_str = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                json_str = response_text.split('```')[1].split('```')[0]

            analysis = json.loads(json_str.strip())
            return analysis

        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse AI response as JSON: {e}")
            return self._get_default_analysis()
        except anthropic.APIError as e:
            print(f"API Error: {e}")
            raise

    def _get_default_analysis(self) -> dict:
        """Return default analysis structure if AI fails."""
        return {
            "themes": [],
            "key_insights": [],
            "guest_background": {
                "current_role": "Unknown",
                "expertise_areas": [],
                "notable_companies": []
            },
            "topics_discussed": [],
            "memorable_quotes": [],
            "frameworks_mentioned": [],
            "recommended_for": [],
            "sentiment": "unknown",
            "difficulty_level": "unknown",
            "summary": "Analysis failed - please retry"
        }


# ============================================================================
# Metadata Management
# ============================================================================

def create_episode_metadata(
    episode_id: str,
    frontmatter: dict,
    analysis: dict
) -> EpisodeMetadata:
    """Create an EpisodeMetadata object from frontmatter and analysis."""
    return EpisodeMetadata(
        episode_id=episode_id,
        guest=frontmatter.get('guest', 'Unknown'),
        title=frontmatter.get('title', 'Unknown'),
        youtube_url=frontmatter.get('youtube_url', ''),
        duration=frontmatter.get('duration', ''),
        duration_seconds=frontmatter.get('duration_seconds', 0),
        view_count=frontmatter.get('view_count', 0),
        themes=analysis.get('themes', []),
        key_insights=analysis.get('key_insights', []),
        guest_background=analysis.get('guest_background', {}),
        topics_discussed=analysis.get('topics_discussed', []),
        memorable_quotes=analysis.get('memorable_quotes', []),
        frameworks_mentioned=analysis.get('frameworks_mentioned', []),
        recommended_for=analysis.get('recommended_for', []),
        sentiment=analysis.get('sentiment', 'unknown'),
        difficulty_level=analysis.get('difficulty_level', 'unknown'),
        summary=analysis.get('summary', ''),
        analyzed_at=datetime.now(timezone.utc).isoformat()
    )


def save_episode_metadata(episode_dir: Path, metadata: EpisodeMetadata):
    """Save episode metadata to JSON file in the episode directory."""
    metadata_path = episode_dir / METADATA_FILENAME
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(metadata), f, indent=2, ensure_ascii=False)


def load_episode_metadata(episode_dir: Path) -> Optional[EpisodeMetadata]:
    """Load existing metadata from episode directory if it exists."""
    metadata_path = episode_dir / METADATA_FILENAME
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return EpisodeMetadata(**data)
    return None


# ============================================================================
# Aggregation and Visualization
# ============================================================================

def aggregate_themes(all_metadata: list[EpisodeMetadata]) -> AggregateThemes:
    """Aggregate themes and insights across all episodes."""
    theme_counter = Counter()
    topic_counter = Counter()
    framework_counter = Counter()
    expertise_counter = Counter()
    sentiment_counter = Counter()
    difficulty_counter = Counter()
    all_insights = []

    for metadata in all_metadata:
        # Count themes (normalized)
        for theme in metadata.themes:
            theme_counter[normalize_theme(theme)] += 1

        # Count topics (normalized)
        for topic in metadata.topics_discussed:
            topic_counter[normalize_theme(topic)] += 1

        # Count frameworks (normalized)
        for framework in metadata.frameworks_mentioned:
            framework_counter[normalize_theme(framework)] += 1

        # Count expertise areas (normalized)
        if metadata.guest_background:
            for area in metadata.guest_background.get('expertise_areas', []):
                expertise_counter[normalize_theme(area)] += 1

        # Count sentiments (normalized) and difficulties
        sentiment_counter[normalize_sentiment(metadata.sentiment)] += 1
        difficulty_counter[metadata.difficulty_level] += 1

        # Collect insights
        all_insights.extend(metadata.key_insights)

    # Get most common insights (simplified by taking a sample)
    common_insights = list(set(all_insights))[:50]

    return AggregateThemes(
        total_episodes=len(all_metadata),
        theme_counts=dict(theme_counter.most_common(100)),
        topic_counts=dict(topic_counter.most_common(200)),
        framework_counts=dict(framework_counter.most_common(50)),
        guest_expertise_areas=dict(expertise_counter.most_common(50)),
        sentiment_distribution=dict(sentiment_counter),
        difficulty_distribution=dict(difficulty_counter),
        common_insights=common_insights,
        generated_at=datetime.now(timezone.utc).isoformat()
    )


def generate_wordcloud(aggregate: AggregateThemes, output_path: Path):
    """Generate a word cloud visualization from aggregated themes."""
    if not VISUALIZATION_AVAILABLE:
        print("Skipping word cloud generation - dependencies not installed")
        return

    # Combine themes, topics, and frameworks for the word cloud
    word_freq = {}

    # Weight themes more heavily
    for theme, count in aggregate.theme_counts.items():
        word_freq[theme] = count * 3

    # Add topics
    for topic, count in aggregate.topic_counts.items():
        if topic in word_freq:
            word_freq[topic] += count
        else:
            word_freq[topic] = count

    # Add frameworks
    for framework, count in aggregate.framework_counts.items():
        if framework in word_freq:
            word_freq[framework] += count * 2
        else:
            word_freq[framework] = count * 2

    if not word_freq:
        print("No themes to visualize")
        return

    # Create word cloud
    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        colormap='viridis',
        max_words=150,
        min_font_size=10,
        max_font_size=120,
        relative_scaling=0.5,
        prefer_horizontal=0.7
    ).generate_from_frequencies(word_freq)

    # Create figure
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Lenny's Podcast Themes & Topics", fontsize=24, pad=20)
    plt.tight_layout(pad=0)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Word cloud saved to: {output_path}")


def generate_relationships_data(all_metadata: list[EpisodeMetadata]) -> dict:
    """Generate relationship data for network visualization."""
    # Build relationships between themes that appear together
    theme_co_occurrence = defaultdict(lambda: defaultdict(int))
    theme_episodes = defaultdict(list)

    for metadata in all_metadata:
        themes = [t.lower().strip() for t in metadata.themes]

        # Track which episodes have which themes
        for theme in themes:
            theme_episodes[theme].append({
                'episode_id': metadata.episode_id,
                'guest': metadata.guest,
                'title': metadata.title
            })

        # Count co-occurrences
        for i, theme1 in enumerate(themes):
            for theme2 in themes[i+1:]:
                theme_co_occurrence[theme1][theme2] += 1
                theme_co_occurrence[theme2][theme1] += 1

    # Build network data
    nodes = []
    links = []

    # Create nodes for top themes
    top_themes = Counter()
    for metadata in all_metadata:
        for theme in metadata.themes:
            top_themes[theme.lower().strip()] += 1

    theme_to_id = {}
    for idx, (theme, count) in enumerate(top_themes.most_common(50)):
        theme_to_id[theme] = idx
        nodes.append({
            'id': idx,
            'label': theme,
            'size': count,
            'episodes': theme_episodes.get(theme, [])[:10]  # Limit for size
        })

    # Create links for co-occurrences
    seen_links = set()
    for theme1, related in theme_co_occurrence.items():
        if theme1 not in theme_to_id:
            continue
        for theme2, strength in related.items():
            if theme2 not in theme_to_id:
                continue
            if strength < 2:  # Filter weak connections
                continue
            link_key = tuple(sorted([theme1, theme2]))
            if link_key in seen_links:
                continue
            seen_links.add(link_key)
            links.append({
                'source': theme_to_id[theme1],
                'target': theme_to_id[theme2],
                'strength': strength
            })

    return {
        'nodes': nodes,
        'links': links,
        'metadata': {
            'total_episodes': len(all_metadata),
            'total_themes': len(top_themes),
            'generated_at': datetime.now(timezone.utc).isoformat()
        }
    }


# ============================================================================
# Main Processing
# ============================================================================

def discover_episodes(episodes_dir: Path) -> list[Path]:
    """Discover all episode directories."""
    episodes = []
    for item in sorted(episodes_dir.iterdir()):
        if item.is_dir() and (item / 'transcript.md').exists():
            episodes.append(item)
    return episodes


def process_episode(
    episode_dir: Path,
    analyzer: TranscriptAnalyzer,
    skip_existing: bool = False
) -> Optional[EpisodeMetadata]:
    """Process a single episode directory."""
    episode_id = episode_dir.name

    # Check for existing metadata
    if skip_existing:
        existing = load_episode_metadata(episode_dir)
        if existing:
            return existing

    # Parse transcript
    transcript_path = episode_dir / 'transcript.md'
    transcript_data = parse_transcript_file(transcript_path)

    # Analyze with AI
    analysis = analyzer.analyze_episode(transcript_data)

    # Create metadata
    metadata = create_episode_metadata(
        episode_id=episode_id,
        frontmatter=transcript_data['frontmatter'],
        analysis=analysis
    )

    # Save metadata
    save_episode_metadata(episode_dir, metadata)

    return metadata


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze Lenny's Podcast transcripts using AI"
    )
    parser.add_argument(
        '--episodes-dir',
        type=str,
        default=DEFAULT_EPISODES_DIR,
        help='Path to episodes directory'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip episodes that already have metadata.json'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of episodes to process (for testing)'
    )
    parser.add_argument(
        '--only-aggregate',
        action='store_true',
        help='Only run aggregation on existing metadata (skip AI analysis)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)'
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    episodes_dir = script_dir / args.episodes_dir

    if not episodes_dir.exists():
        print(f"Error: Episodes directory not found: {episodes_dir}")
        sys.exit(1)

    # Discover episodes
    print(f"Discovering episodes in: {episodes_dir}")
    episode_dirs = discover_episodes(episodes_dir)
    print(f"Found {len(episode_dirs)} episodes")

    if args.limit:
        episode_dirs = episode_dirs[:args.limit]
        print(f"Limited to {len(episode_dirs)} episodes")

    all_metadata = []

    if args.only_aggregate:
        # Only load existing metadata
        print("\nLoading existing metadata...")
        for episode_dir in tqdm(episode_dirs, desc="Loading"):
            metadata = load_episode_metadata(episode_dir)
            if metadata:
                all_metadata.append(metadata)
        print(f"Loaded metadata for {len(all_metadata)} episodes")
    else:
        # Initialize analyzer
        try:
            analyzer = TranscriptAnalyzer(api_key=args.api_key)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        # Process episodes
        print("\nAnalyzing episodes...")
        for episode_dir in tqdm(episode_dirs, desc="Analyzing"):
            try:
                metadata = process_episode(
                    episode_dir,
                    analyzer,
                    skip_existing=args.skip_existing
                )
                if metadata:
                    all_metadata.append(metadata)
            except Exception as e:
                print(f"\nError processing {episode_dir.name}: {e}")
                continue

    if not all_metadata:
        print("No metadata to aggregate")
        sys.exit(0)

    # Aggregate themes
    print("\nAggregating themes and insights...")
    aggregate = aggregate_themes(all_metadata)

    # Save aggregate data
    aggregate_path = script_dir / THEMES_AGGREGATE_FILE
    with open(aggregate_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(aggregate), f, indent=2, ensure_ascii=False)
    print(f"Aggregate data saved to: {aggregate_path}")

    # Generate word cloud
    print("\nGenerating word cloud...")
    wordcloud_path = script_dir / WORDCLOUD_OUTPUT
    generate_wordcloud(aggregate, wordcloud_path)

    # Generate relationships data
    print("\nGenerating relationship network data...")
    relationships = generate_relationships_data(all_metadata)
    relationships_path = script_dir / RELATIONSHIPS_OUTPUT
    with open(relationships_path, 'w', encoding='utf-8') as f:
        json.dump(relationships, f, indent=2, ensure_ascii=False)
    print(f"Relationships data saved to: {relationships_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Episodes analyzed: {len(all_metadata)}")
    print(f"Unique themes found: {len(aggregate.theme_counts)}")
    print(f"Unique topics found: {len(aggregate.topic_counts)}")
    print(f"Frameworks mentioned: {len(aggregate.framework_counts)}")
    print("\nTop 10 Themes:")
    for theme, count in list(aggregate.theme_counts.items())[:10]:
        print(f"  - {theme}: {count} episodes")
    print("\nOutput files:")
    print(f"  - Episode metadata: {episodes_dir}/*/metadata.json")
    print(f"  - Aggregate themes: {aggregate_path}")
    print(f"  - Word cloud: {wordcloud_path}")
    print(f"  - Relationships: {relationships_path}")


if __name__ == '__main__':
    main()
