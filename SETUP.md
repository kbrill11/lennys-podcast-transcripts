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
pyenv install --list | grep "3.11"

# Install Python 3.11 (recommended)
pyenv install 3.11.9

# Set as local version for this project
cd /path/to/lennys-podcast-transcripts
pyenv local 3.11.9

# Verify installation
python --version
# Should output: Python 3.11.9
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
