# Transcript Analyzer Setup Guide

This guide walks you through setting up a Python environment from scratch to run the transcript analysis script.

## Prerequisites

- macOS, Linux, or Windows with WSL
- Terminal access
- An Anthropic API key (get one at [console.anthropic.com](https://console.anthropic.com))

## Step 1: Install pyenv

pyenv lets you easily install and manage multiple Python versions.

### macOS (using Homebrew)

```bash
brew update
brew install pyenv
```

### Linux / WSL

```bash
curl https://pyenv.run | bash
```

### Add pyenv to your shell

Add these lines to your shell configuration file (`~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`):

```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Restart your terminal or run:

```bash
source ~/.bashrc  # or ~/.zshrc
```

## Step 2: Install Python with pyenv

```bash
# List available Python versions
pyenv install --list | grep "3.12"

# Install Python 3.12 (recommended)
pyenv install 3.12.8

# Set as local version for this project
cd /path/to/lennys-podcast-transcripts
pyenv local 3.12.8

# Verify installation
python --version
# Should output: Python 3.12.8
```

## Step 3: Create a Virtual Environment

```bash
# Navigate to the project directory
cd /path/to/lennys-podcast-transcripts

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

Your terminal prompt should now show `(.venv)` indicating the virtual environment is active.

## Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

## Step 5: Set Up Your API Key

The script requires an Anthropic API key to analyze transcripts with Claude.

### Option A: Environment Variable (Recommended)

```bash
# Add to your shell config for persistence
echo 'export ANTHROPIC_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc

# Or set for current session only
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Option B: Pass Directly to Script

```bash
python analyze_transcripts.py --api-key "your-api-key-here"
```

## Script Usage

### Basic Usage

```bash
# Run full analysis on all episodes
python analyze_transcripts.py
```

This will:
1. Analyze all 269 episode transcripts using Claude AI
2. Generate a `metadata.json` file in each episode folder
3. Create `themes_aggregate.json` with combined statistics
4. Generate `themes_wordcloud.png` visualization
5. Export `relationships_network.json` for theme connections

### Command Line Options

| Option | Description |
|--------|-------------|
| `--episodes-dir PATH` | Path to episodes directory (default: `episodes`) |
| `--skip-existing` | Skip episodes that already have `metadata.json` |
| `--limit N` | Process only N episodes (useful for testing) |
| `--only-aggregate` | Only regenerate visualizations from existing metadata |
| `--api-key KEY` | Provide API key directly (alternative to env var) |

### Example Commands

```bash
# Test with 5 episodes first
python analyze_transcripts.py --limit 5

# Resume analysis, skipping already-processed episodes
python analyze_transcripts.py --skip-existing

# Regenerate word cloud and aggregates from existing metadata
python analyze_transcripts.py --only-aggregate

# Use a different episodes directory
python analyze_transcripts.py --episodes-dir /path/to/episodes
```

## Output Files

After running the script, you'll find:

### Per-Episode Metadata

Each episode folder will contain a `metadata.json`:

```
episodes/
├── brian-chesky/
│   ├── transcript.md
│   └── metadata.json    <- Generated
├── shreyas-doshi/
│   ├── transcript.md
│   └── metadata.json    <- Generated
└── ...
```

### Aggregate Files (in root directory)

| File | Description |
|------|-------------|
| `themes_aggregate.json` | Combined theme/topic counts across all episodes |
| `themes_wordcloud.png` | Visual word cloud of themes and topics |
| `relationships_network.json` | Network graph data for theme relationships |

## Metadata Structure

Each `metadata.json` contains:

```json
{
  "episode_id": "brian-chesky",
  "guest": "Brian Chesky",
  "title": "Episode title",
  "themes": ["leadership", "product management", "company building"],
  "key_insights": ["Insight 1", "Insight 2"],
  "guest_background": {
    "current_role": "CEO of Airbnb",
    "expertise_areas": ["product", "design"],
    "notable_companies": ["Airbnb"]
  },
  "topics_discussed": ["Topic 1", "Topic 2"],
  "memorable_quotes": ["Quote 1"],
  "frameworks_mentioned": ["Framework 1"],
  "recommended_for": ["Product managers", "Founders"],
  "sentiment": "inspiring",
  "difficulty_level": "intermediate",
  "summary": "Episode summary..."
}
```

## Troubleshooting

### "ANTHROPIC_API_KEY not found"

Make sure you've set the environment variable:

```bash
echo $ANTHROPIC_API_KEY
# Should print your API key
```

### "wordcloud/matplotlib not installed"

The visualization is optional. Install with:

```bash
pip install wordcloud matplotlib numpy
```

### Rate Limiting

If you hit API rate limits, the script will fail. Solutions:
- Use `--limit` to process in batches
- Use `--skip-existing` to resume where you left off
- Wait and retry

### Virtual Environment Not Active

Always activate your virtual environment before running:

```bash
source .venv/bin/activate
```

## Visualization Script

A separate script `visualize_themes.py` provides advanced word cloud generation capabilities.

### Basic Visualization Usage

```bash
# Generate default combined word cloud
python visualize_themes.py

# Generate themes-only word cloud
python visualize_themes.py --style themes

# Multi-panel visualization (4 word clouds in one image)
python visualize_themes.py --style multi
```

### Visualization Styles

| Style | Description |
|-------|-------------|
| `themes` | High-level episode themes |
| `topics` | Specific topics discussed |
| `frameworks` | Frameworks and methodologies mentioned |
| `expertise` | Guest expertise areas |
| `insights` | Key words from insights |
| `combined` | Weighted combination (default) |
| `all` | Everything combined equally |
| `multi` | Multi-panel with 4 word clouds |
| `comparison` | Side-by-side themes vs topics |
| `circular` | Circular word cloud shape |

### Customization Options

```bash
# Change color scheme
python visualize_themes.py --colormap plasma
python visualize_themes.py --colormap tech  # custom palette

# Change dimensions
python visualize_themes.py --width 2400 --height 1200

# Different output formats
python visualize_themes.py --format svg
python visualize_themes.py --format pdf

# Custom output filename
python visualize_themes.py --output my_wordcloud.png

# Dark background
python visualize_themes.py --background black --colormap viridis

# Limit word count
python visualize_themes.py --max-words 100
```

### Available Color Palettes

```bash
# List all available colormaps
python visualize_themes.py --list-colormaps
```

**Matplotlib colormaps:** viridis, plasma, inferno, magma, cividis, Blues, Greens, Oranges, etc.

**Custom palettes:**
- `podcast` - Spotify-inspired green/black
- `tech` - Vibrant tech colors
- `warm` - Warm reds and oranges
- `cool` - Cool blues and greens
- `mono` - Monochromatic grays

### Print Statistics

```bash
# Show theme statistics without generating image
python visualize_themes.py --stats
```

## Knowledge Graph Visualization

The `visualize_knowledge_graph.py` script generates interactive network graphs showing relationships between themes, episodes, and guests.

### Install Dependencies

```bash
pip install networkx pyvis
```

### Basic Usage

```bash
# Generate theme network (default)
python visualize_knowledge_graph.py

# Generate with statistics
python visualize_knowledge_graph.py --mode themes --stats
```

This creates an interactive HTML file you can open in any web browser.

### Visualization Modes

| Mode | Nodes | Edges | Best For |
|------|-------|-------|----------|
| `themes` | Themes | Co-occurrence | See which topics cluster together |
| `episodes` | Episodes | Shared themes | Find related episodes |
| `guests` | Guests | Topic overlap | Find guests with similar expertise |
| `mixed` | Themes + Episodes | Episode-to-theme links | See the full picture |

### Example Commands

```bash
# Theme network - shows how topics relate to each other
python visualize_knowledge_graph.py --mode themes

# Episode connections - find episodes that cover similar ground
python visualize_knowledge_graph.py --mode episodes --min-shared 2

# Guest network - see which guests have overlapping expertise
python visualize_knowledge_graph.py --mode guests

# Custom output filename
python visualize_knowledge_graph.py --mode themes -o my_graph.html

# For smaller datasets, lower the thresholds
python visualize_knowledge_graph.py --mode themes --min-occurrences 1 --min-shared 1
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--mode`, `-m` | Visualization mode: `themes`, `episodes`, `guests`, `mixed` |
| `--output`, `-o` | Output filename (default: `knowledge_graph_<mode>.html`) |
| `--min-occurrences` | Minimum theme occurrences to include (default: 2) |
| `--min-shared` | Minimum shared themes/topics for an edge (default: 2) |
| `--max-nodes` | Maximum number of nodes to display (default: 75) |
| `--no-physics` | Disable physics simulation (static layout) |
| `--stats` | Print graph statistics |

### Interacting with the Graph

Once you open the HTML file in a browser:

- **Hover** over nodes to see details (episodes, topics, summaries)
- **Drag** nodes to rearrange the layout
- **Scroll** to zoom in/out
- **Double-click** a node to focus on it
- **Click and drag** the background to pan

### Node Sizing

Node size reflects importance:
- **Themes mode**: Size = number of episodes containing the theme
- **Episodes mode**: Size = number of themes in the episode
- **Guests mode**: Size = number of topics discussed

### Edge Thickness

Edge thickness reflects relationship strength:
- **Themes mode**: Number of episodes where both themes appear
- **Episodes mode**: Number of shared themes between episodes
- **Guests mode**: Number of overlapping topics/expertise areas

### Output Files

| File | Description |
|------|-------------|
| `knowledge_graph_themes.html` | Theme co-occurrence network |
| `knowledge_graph_episodes.html` | Episode similarity network |
| `knowledge_graph_guests.html` | Guest expertise network |
| `knowledge_graph_mixed.html` | Combined themes and episodes |

## Deactivating the Environment

When finished, deactivate the virtual environment:

```bash
deactivate
```

## Updating Dependencies

To update packages to their latest versions:

```bash
pip install --upgrade -r requirements.txt
```
