#!/usr/bin/env python3
"""
Topic Evolution & Guest Expertise Analysis

Analyzes podcast transcripts to understand:
1. How themes/topics have evolved over time (by quarter)
2. What topics guests from different backgrounds discuss

Usage:
    python analyze_evolution_and_expertise.py [--mode all|evolution|expertise]

Outputs:
    - topic_evolution.json / topic_evolution.html / TOPIC_EVOLUTION.md
    - guest_expertise_mapping.json / guest_expertise_mapping.html / GUEST_EXPERTISE.md
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from typing import Optional


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_EPISODES_DIR = "episodes"
METADATA_FILENAME = "metadata.json"

# Company categorization
BIG_TECH_COMPANIES = {
    "google", "meta", "facebook", "apple", "amazon", "netflix", "microsoft",
    "linkedin", "twitter", "x", "salesforce", "adobe", "oracle", "ibm",
    "uber", "lyft", "snapchat", "snap", "pinterest", "spotify", "nvidia"
}

UNICORN_COMPANIES = {
    "airbnb", "dropbox", "hubspot", "stripe", "square", "block", "doordash",
    "instacart", "coinbase", "robinhood", "figma", "notion", "slack",
    "zoom", "databricks", "snowflake", "canva", "miro", "airtable",
    "plaid", "chime", "klarna", "flexport", "opendoor"
}

VC_FIRMS = {
    "andreessen horowitz", "a16z", "sequoia", "benchmark", "greylock",
    "kleiner perkins", "accel", "index ventures", "first round",
    "y combinator", "yc", "founders fund", "bessemer", "lightspeed"
}

FINTECH_COMPANIES = {
    "paypal", "stripe", "square", "block", "coinbase", "robinhood",
    "plaid", "chime", "klarna", "affirm", "brex", "ramp", "mercury"
}


# ============================================================================
# Theme Normalization (copied from analyze_transcripts.py)
# ============================================================================

THEME_ALIASES = {
    "product-market fit": "product market fit",
    "jobs-to-be-done framework": "jobs to be done framework",
    "jobs-to-be-done": "jobs to be done",
    "product-led growth": "product led growth",
    "go-to-market strategy": "go to market strategy",
    "ai-powered software development": "ai powered software development",
    "hiring and team building": "team building and hiring",
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
    "product management career development": "product management",
    "product management evolution": "product management",
    "product management leadership": "product management",
    "product management strategy": "product management",
    "product management philosophy": "product management",
    "product management fundamentals": "product management",
    "product management in media/journalism": "product management",
    "crisis product management": "product management",
    "startup growth strategies": "growth strategies",
    "growth team building": "growth strategies",
    "growth strategy execution": "growth strategies",
    "startup growth tactics": "growth strategies",
    "growth marketing strategy": "growth strategies",
    "growth cmo role evolution": "growth strategies",
    "growth strategy and optimization": "growth strategies",
    "experimentation and testing": "experimentation",
    "experimentation frameworks": "experimentation",
    "experimentation methodology": "experimentation",
    "experimentation and a/b testing": "experimentation",
    "career advancement": "career development",
    "career strategy and decision making": "career development",
    "early career development": "career development",
    "career decision making": "career development",
    "company culture and decision making": "company culture",
    "company culture and leadership": "company culture",
    "work life balance and priorities": "work life balance",
    "burnout vs depression": "work life balance",
    "mental health in tech": "work life balance",
}


def normalize_theme(theme: str) -> str:
    """Normalize theme to canonical form."""
    normalized = theme.lower().strip()
    normalized = re.sub(r'(?<=[a-z])-(?=[a-z])', ' ', normalized)
    return THEME_ALIASES.get(normalized, normalized)


# ============================================================================
# Data Loading
# ============================================================================

def load_metadata(episode_dir: Path) -> Optional[dict]:
    """Load metadata from episode directory."""
    metadata_path = episode_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def discover_episodes(episodes_dir: Path) -> list[Path]:
    """Discover all episode directories with metadata."""
    episodes = []
    for item in sorted(episodes_dir.iterdir()):
        if item.is_dir() and (item / METADATA_FILENAME).exists():
            episodes.append(item)
    return episodes


def load_all_metadata(episodes_dir: Path) -> list[dict]:
    """Load all episode metadata."""
    episode_dirs = discover_episodes(episodes_dir)
    all_metadata = []

    for episode_dir in episode_dirs:
        metadata = load_metadata(episode_dir)
        if metadata:
            all_metadata.append(metadata)

    return all_metadata


# ============================================================================
# Guest Categorization
# ============================================================================

def categorize_by_company_type(notable_companies: list[str], current_role: str) -> list[str]:
    """Categorize guest by company type based on their background."""
    categories = []
    companies_lower = [c.lower() for c in notable_companies]
    role_lower = current_role.lower() if current_role else ""

    # Check for VC/investor
    if any(vc in c for c in companies_lower for vc in VC_FIRMS) or \
       any(word in role_lower for word in ["partner", "investor", "venture", "vc"]):
        categories.append("vc_investor")

    # Check for Big Tech
    if any(bt in c for c in companies_lower for bt in BIG_TECH_COMPANIES):
        categories.append("big_tech")

    # Check for Unicorn
    if any(u in c for c in companies_lower for u in UNICORN_COMPANIES):
        categories.append("unicorn")

    # Check for Fintech
    if any(ft in c for c in companies_lower for ft in FINTECH_COMPANIES):
        categories.append("fintech")

    # Check for Founder
    if any(word in role_lower for word in ["founder", "co-founder", "cofounder", "ceo"]):
        categories.append("founder")

    # Check for Consultant/Coach/Advisor
    if any(word in role_lower for word in ["consultant", "coach", "advisor", "author"]):
        categories.append("consulting")

    # Default to "other" if no category matched
    if not categories:
        categories.append("other")

    return categories


def categorize_by_expertise(expertise_areas: list[str]) -> str:
    """Determine primary expertise category."""
    if not expertise_areas:
        return "general"

    expertise_lower = [e.lower() for e in expertise_areas]

    # Check for specific expertise patterns
    if any("ai" in e or "machine learning" in e or "ml" in e for e in expertise_lower):
        return "ai_ml"
    if any("growth" in e for e in expertise_lower):
        return "growth"
    if any("engineer" in e or "technical" in e or "software" in e for e in expertise_lower):
        return "engineering"
    if any("design" in e or "ux" in e for e in expertise_lower):
        return "design"
    if any("marketing" in e or "brand" in e for e in expertise_lower):
        return "marketing"
    if any("product" in e for e in expertise_lower):
        return "product"
    if any("leader" in e or "management" in e or "executive" in e for e in expertise_lower):
        return "leadership"

    return "general"


# ============================================================================
# Topic Evolution Analysis
# ============================================================================

def get_quarter(date_str: str) -> str:
    """Convert date string to quarter (e.g., '2023-Q2')."""
    if not date_str:
        return "Unknown"

    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year}-Q{quarter}"
    except ValueError:
        return "Unknown"


def analyze_topic_evolution(all_metadata: list[dict]) -> dict:
    """Analyze how topics have evolved over time."""
    # Group episodes by quarter
    episodes_by_quarter = defaultdict(list)

    for metadata in all_metadata:
        publish_date = metadata.get("publish_date")
        quarter = get_quarter(publish_date)
        if quarter != "Unknown":
            episodes_by_quarter[quarter].append(metadata)

    # Sort quarters chronologically
    sorted_quarters = sorted(episodes_by_quarter.keys())

    # Build timeline data
    timeline = []
    theme_first_appearance = {}
    all_theme_counts = Counter()

    for quarter in sorted_quarters:
        episodes = episodes_by_quarter[quarter]
        quarter_themes = Counter()

        for ep in episodes:
            for theme in ep.get("themes", []):
                normalized = normalize_theme(theme)
                quarter_themes[normalized] += 1
                all_theme_counts[normalized] += 1

                if normalized not in theme_first_appearance:
                    theme_first_appearance[normalized] = quarter

        # Get top themes for this quarter
        top_themes = [
            {
                "theme": theme,
                "count": count,
                "percentage": round(count / len(episodes) * 100, 1)
            }
            for theme, count in quarter_themes.most_common(10)
        ]

        # Calculate quarter date range
        year, q = quarter.split("-Q")
        q = int(q)
        start_month = (q - 1) * 3 + 1
        end_month = q * 3

        timeline.append({
            "period": quarter,
            "start_date": f"{year}-{start_month:02d}-01",
            "end_date": f"{year}-{end_month:02d}-{'30' if end_month in [4, 6, 9, 11] else '31' if end_month in [1, 3, 5, 7, 8, 10, 12] else '28'}",
            "episode_count": len(episodes),
            "top_themes": top_themes,
            "episodes": [
                {
                    "id": ep.get("episode_id"),
                    "guest": ep.get("guest"),
                    "title": ep.get("title"),
                    "date": ep.get("publish_date")
                }
                for ep in episodes
            ]
        })

    # Build theme trajectories for significant themes
    significant_themes = [t for t, c in all_theme_counts.most_common(20)]
    theme_trajectories = {}

    for theme in significant_themes:
        trajectory = []
        for quarter in sorted_quarters:
            episodes = episodes_by_quarter[quarter]
            count = sum(
                1 for ep in episodes
                if theme in [normalize_theme(t) for t in ep.get("themes", [])]
            )
            if count > 0 or trajectory:  # Include zeros after first appearance
                trajectory.append({
                    "period": quarter,
                    "count": count,
                    "percentage": round(count / len(episodes) * 100, 1) if episodes else 0
                })

        theme_trajectories[theme] = {
            "first_appearance": theme_first_appearance.get(theme),
            "total_count": all_theme_counts[theme],
            "trajectory": trajectory
        }

    # Identify notable shifts with more relaxed criteria
    notable_shifts = []

    # Define time periods
    recent_quarters = sorted_quarters[-4:] if len(sorted_quarters) >= 4 else sorted_quarters
    early_quarters = sorted_quarters[:4] if len(sorted_quarters) >= 4 else sorted_quarters
    mid_quarters = sorted_quarters[4:8] if len(sorted_quarters) >= 8 else []

    # Find emerging themes (appeared in recent half and growing)
    midpoint = len(sorted_quarters) // 2
    recent_half = sorted_quarters[midpoint:]

    for theme, data in theme_trajectories.items():
        first_q = data["first_appearance"]
        if first_q and first_q in recent_half and data["total_count"] >= 2:
            notable_shifts.append({
                "theme": theme,
                "shift_type": "emergence",
                "description": f"'{theme}' is a newer topic, first appearing in {first_q} with {data['total_count']} total episodes"
            })

    # Find themes with growth trends (comparing first half to second half)
    for theme, data in theme_trajectories.items():
        trajectory = data.get("trajectory", [])
        if len(trajectory) >= 4:
            mid = len(trajectory) // 2
            early_sum = sum(p["count"] for p in trajectory[:mid])
            recent_sum = sum(p["count"] for p in trajectory[mid:])

            if early_sum > 0 and recent_sum > early_sum * 1.5 and recent_sum >= 3:
                notable_shifts.append({
                    "theme": theme,
                    "shift_type": "growth",
                    "description": f"'{theme}' is trending up: {early_sum} episodes in first half → {recent_sum} in second half"
                })
            elif recent_sum > 0 and early_sum > recent_sum * 1.5 and early_sum >= 3:
                notable_shifts.append({
                    "theme": theme,
                    "shift_type": "decline",
                    "description": f"'{theme}' has declined: {early_sum} episodes in first half → {recent_sum} in second half"
                })

    # Find consistent themes (steady presence throughout)
    for theme, data in theme_trajectories.items():
        trajectory = data.get("trajectory", [])
        if len(trajectory) >= 6 and data["total_count"] >= 8:
            # Check if it appears in most quarters
            quarters_present = sum(1 for p in trajectory if p["count"] > 0)
            if quarters_present >= len(trajectory) * 0.6:
                notable_shifts.append({
                    "theme": theme,
                    "shift_type": "consistent",
                    "description": f"'{theme}' is a consistent topic, appearing in {quarters_present}/{len(trajectory)} quarters ({data['total_count']} total episodes)"
                })

    # Deduplicate (a theme might match multiple criteria)
    seen_themes = set()
    unique_shifts = []
    for shift in notable_shifts:
        if shift["theme"] not in seen_themes:
            seen_themes.add(shift["theme"])
            unique_shifts.append(shift)
    notable_shifts = unique_shifts

    return {
        "timeline": timeline,
        "theme_trajectories": theme_trajectories,
        "notable_shifts": notable_shifts,
        "metadata": {
            "total_episodes": len(all_metadata),
            "episodes_with_dates": sum(1 for m in all_metadata if m.get("publish_date")),
            "quarters_covered": len(sorted_quarters),
            "date_range": f"{sorted_quarters[0]} to {sorted_quarters[-1]}" if sorted_quarters else "N/A",
            "generated_at": datetime.utcnow().isoformat()
        }
    }


# ============================================================================
# Guest Expertise Mapping
# ============================================================================

def analyze_guest_expertise(all_metadata: list[dict]) -> dict:
    """Analyze what topics guests from different backgrounds discuss."""
    # Categorize all guests
    guests_by_company_type = defaultdict(list)
    guests_by_expertise = defaultdict(list)

    for metadata in all_metadata:
        guest_bg = metadata.get("guest_background", {})
        notable_companies = guest_bg.get("notable_companies", [])
        current_role = guest_bg.get("current_role", "")
        expertise_areas = guest_bg.get("expertise_areas", [])

        # Categorize by company type
        company_categories = categorize_by_company_type(notable_companies, current_role)
        for cat in company_categories:
            guests_by_company_type[cat].append(metadata)

        # Categorize by expertise
        expertise_cat = categorize_by_expertise(expertise_areas)
        guests_by_expertise[expertise_cat].append(metadata)

    # Build category analysis
    guest_categories = {}

    for cat, episodes in guests_by_company_type.items():
        theme_counts = Counter()
        for ep in episodes:
            for theme in ep.get("themes", []):
                theme_counts[normalize_theme(theme)] += 1

        guests = [
            {
                "name": ep.get("guest"),
                "companies": ep.get("guest_background", {}).get("notable_companies", []),
                "current_role": ep.get("guest_background", {}).get("current_role", ""),
                "episode_id": ep.get("episode_id"),
                "title": ep.get("title"),
                "themes_discussed": [normalize_theme(t) for t in ep.get("themes", [])][:5]
            }
            for ep in episodes
        ]

        guest_categories[cat] = {
            "total_episodes": len(episodes),
            "guests": guests,
            "aggregate_themes": dict(theme_counts.most_common(15)),
            "top_themes": [t for t, _ in theme_counts.most_common(5)]
        }

    # Build expertise to topics mapping
    expertise_to_topics = {}

    for exp, episodes in guests_by_expertise.items():
        theme_counts = Counter()
        frameworks = Counter()

        for ep in episodes:
            for theme in ep.get("themes", []):
                theme_counts[normalize_theme(theme)] += 1
            for fw in ep.get("frameworks_mentioned", []):
                frameworks[fw.lower()] += 1

        expertise_to_topics[exp] = {
            "total_episodes": len(episodes),
            "common_themes": [t for t, _ in theme_counts.most_common(8)],
            "theme_counts": dict(theme_counts.most_common(10)),
            "common_frameworks": [f for f, _ in frameworks.most_common(5)]
        }

    # Cross-category analysis - use relative frequencies to find distinctive themes
    cross_analysis = {}

    def get_distinctive_themes(cat1_data: dict, cat2_data: dict, cat1_name: str, cat2_name: str) -> dict:
        """Find themes that are distinctively more common in one category vs another."""
        cat1_themes = cat1_data.get("aggregate_themes", {})
        cat2_themes = cat2_data.get("aggregate_themes", {})
        cat1_total = cat1_data.get("total_episodes", 1)
        cat2_total = cat2_data.get("total_episodes", 1)

        # Calculate relative frequency for each theme
        all_themes = set(cat1_themes.keys()) | set(cat2_themes.keys())

        cat1_distinctive = []
        cat2_distinctive = []
        shared = []

        for theme in all_themes:
            cat1_freq = cat1_themes.get(theme, 0) / cat1_total
            cat2_freq = cat2_themes.get(theme, 0) / cat2_total

            # Theme is distinctive if it appears 2x more often (relatively) in one category
            if cat1_freq > 0 and cat2_freq > 0:
                if cat1_freq > cat2_freq * 1.5:
                    cat1_distinctive.append((theme, cat1_freq))
                elif cat2_freq > cat1_freq * 1.5:
                    cat2_distinctive.append((theme, cat2_freq))
                else:
                    shared.append((theme, (cat1_freq + cat2_freq) / 2))
            elif cat1_freq > 0.05:  # Only in cat1 and meaningful
                cat1_distinctive.append((theme, cat1_freq))
            elif cat2_freq > 0.05:  # Only in cat2 and meaningful
                cat2_distinctive.append((theme, cat2_freq))

        # Sort by frequency and take top items
        cat1_distinctive.sort(key=lambda x: -x[1])
        cat2_distinctive.sort(key=lambda x: -x[1])
        shared.sort(key=lambda x: -x[1])

        return {
            f"{cat1_name}_unique": [t[0] for t in cat1_distinctive[:5]],
            f"{cat2_name}_unique": [t[0] for t in cat2_distinctive[:5]],
            "shared": [t[0] for t in shared[:5]]
        }

    # Founders vs Big Tech comparison
    if "founder" in guest_categories and "big_tech" in guest_categories:
        cross_analysis["founders_vs_big_tech"] = get_distinctive_themes(
            guest_categories["founder"],
            guest_categories["big_tech"],
            "founder", "big_tech"
        )

    # Consulting vs Operators comparison
    if "consulting" in guest_categories and "big_tech" in guest_categories:
        cross_analysis["consultants_vs_operators"] = get_distinctive_themes(
            guest_categories["consulting"],
            guest_categories["big_tech"],
            "consultant", "operator"
        )

    # Unicorn vs Big Tech comparison
    if "unicorn" in guest_categories and "big_tech" in guest_categories:
        cross_analysis["unicorn_vs_big_tech"] = get_distinctive_themes(
            guest_categories["unicorn"],
            guest_categories["big_tech"],
            "unicorn", "big_tech"
        )

    return {
        "guest_categories": guest_categories,
        "expertise_to_topics": expertise_to_topics,
        "cross_category_analysis": cross_analysis,
        "metadata": {
            "total_episodes": len(all_metadata),
            "company_type_distribution": {k: len(v) for k, v in guests_by_company_type.items()},
            "expertise_distribution": {k: len(v) for k, v in guests_by_expertise.items()},
            "generated_at": datetime.utcnow().isoformat()
        }
    }


# ============================================================================
# HTML Generation
# ============================================================================

def generate_evolution_html(evolution_data: dict, output_path: Path):
    """Generate interactive HTML for topic evolution."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lenny's Podcast - Topic Evolution</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 30px;
        }
        header h1 { font-size: 2.5em; margin-bottom: 10px; }
        header p { opacity: 0.9; font-size: 1.1em; }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stat-card .number { font-size: 2em; font-weight: bold; color: #667eea; }
        .stat-card .label { color: #666; font-size: 0.9em; }

        .section { background: white; border-radius: 10px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .section h2 { margin-bottom: 20px; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }

        .timeline { display: flex; flex-direction: column; gap: 20px; }
        .quarter-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }
        .quarter-header {
            background: #f8f9fa;
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .quarter-header:hover { background: #e9ecef; }
        .quarter-title { font-weight: bold; font-size: 1.1em; }
        .quarter-meta { color: #666; font-size: 0.9em; }
        .quarter-content { padding: 20px; display: none; }
        .quarter-card.active .quarter-content { display: block; }

        .theme-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 15px; }
        .theme-tag {
            background: #e9ecef;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
        }
        .theme-tag.hot { background: #ffeaa7; }
        .theme-tag.rising { background: #81ecec; }

        .episode-list { list-style: none; }
        .episode-list li {
            padding: 10px 0;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
        }
        .episode-list li:last-child { border-bottom: none; }
        .episode-guest { font-weight: 500; }
        .episode-date { color: #666; font-size: 0.85em; }

        .shifts { display: grid; gap: 15px; }
        .shift-card {
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid;
        }
        .shift-card.emergence { border-color: #00b894; background: #e8f8f5; }
        .shift-card.growth { border-color: #0984e3; background: #e8f4fd; }
        .shift-card.decline { border-color: #e17055; background: #ffeaa7; }
        .shift-card.consistent { border-color: #6c5ce7; background: #e8e4fd; }
        .shift-card .shift-theme { font-weight: bold; margin-bottom: 5px; }
        .shift-card .shift-desc { font-size: 0.9em; color: #555; }

        .search-box {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            margin-bottom: 20px;
        }
        .search-box:focus { outline: none; border-color: #667eea; }

        .trajectories { display: grid; gap: 15px; }
        .trajectory-card {
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
        }
        .trajectory-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .trajectory-name { font-weight: bold; }
        .trajectory-count { color: #666; font-size: 0.9em; }
        .trajectory-bar {
            display: flex;
            gap: 2px;
            height: 30px;
            align-items: flex-end;
        }
        .bar-segment {
            background: #667eea;
            min-width: 20px;
            border-radius: 3px 3px 0 0;
        }
    </style>
</head>
<body>
    <header>
        <h1>Topic Evolution</h1>
        <p>How Lenny's Podcast themes have changed over time</p>
    </header>

    <div class="container">
        <div class="stats">
            <div class="stat-card">
                <div class="number">""" + str(evolution_data["metadata"]["total_episodes"]) + """</div>
                <div class="label">Total Episodes</div>
            </div>
            <div class="stat-card">
                <div class="number">""" + str(evolution_data["metadata"]["quarters_covered"]) + """</div>
                <div class="label">Quarters Covered</div>
            </div>
            <div class="stat-card">
                <div class="number">""" + str(len(evolution_data["theme_trajectories"])) + """</div>
                <div class="label">Tracked Themes</div>
            </div>
            <div class="stat-card">
                <div class="number">""" + str(len(evolution_data["notable_shifts"])) + """</div>
                <div class="label">Notable Shifts</div>
            </div>
        </div>

        <div class="section">
            <h2>Notable Shifts</h2>
            <div class="shifts">
"""

    for shift in evolution_data.get("notable_shifts", []):
        shift_type = shift.get("shift_type", "other")
        html += f"""                <div class="shift-card {shift_type}">
                    <div class="shift-theme">{shift.get("theme", "")}</div>
                    <div class="shift-desc">{shift.get("description", "")}</div>
                </div>
"""

    html += """            </div>
        </div>

        <div class="section">
            <h2>Theme Trajectories</h2>
            <input type="text" class="search-box" placeholder="Search themes..." id="themeSearch">
            <div class="trajectories" id="trajectories">
"""

    for theme, data in evolution_data.get("theme_trajectories", {}).items():
        trajectory = data.get("trajectory", [])
        max_count = max((p["count"] for p in trajectory), default=1)

        html += f"""                <div class="trajectory-card" data-theme="{theme}">
                    <div class="trajectory-header">
                        <span class="trajectory-name">{theme}</span>
                        <span class="trajectory-count">{data.get("total_count", 0)} total episodes</span>
                    </div>
                    <div class="trajectory-bar">
"""
        for point in trajectory:
            height = int((point["count"] / max_count) * 100) if max_count > 0 else 0
            html += f"""                        <div class="bar-segment" style="height: {max(height, 5)}%;" title="{point['period']}: {point['count']}"></div>
"""
        html += """                    </div>
                </div>
"""

    html += """            </div>
        </div>

        <div class="section">
            <h2>Timeline by Quarter</h2>
            <div class="timeline">
"""

    for period in reversed(evolution_data.get("timeline", [])):
        html += f"""                <div class="quarter-card">
                    <div class="quarter-header" onclick="this.parentElement.classList.toggle('active')">
                        <span class="quarter-title">{period.get("period", "")}</span>
                        <span class="quarter-meta">{period.get("episode_count", 0)} episodes</span>
                    </div>
                    <div class="quarter-content">
                        <div class="theme-tags">
"""
        for t in period.get("top_themes", [])[:8]:
            html += f"""                            <span class="theme-tag">{t.get("theme", "")} ({t.get("count", 0)})</span>
"""
        html += """                        </div>
                        <ul class="episode-list">
"""
        for ep in period.get("episodes", [])[:10]:
            html += f"""                            <li>
                                <span class="episode-guest">{ep.get("guest", "")}</span>
                                <span class="episode-date">{ep.get("date", "")}</span>
                            </li>
"""
        html += """                        </ul>
                    </div>
                </div>
"""

    html += """            </div>
        </div>
    </div>

    <script>
        document.getElementById('themeSearch').addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            document.querySelectorAll('.trajectory-card').forEach(card => {
                const theme = card.dataset.theme.toLowerCase();
                card.style.display = theme.includes(query) ? 'block' : 'none';
            });
        });
    </script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def generate_expertise_html(expertise_data: dict, output_path: Path):
    """Generate interactive HTML for guest expertise mapping."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lenny's Podcast - Guest Expertise Mapping</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            margin-bottom: 30px;
        }
        header h1 { font-size: 2.5em; margin-bottom: 10px; }
        header p { opacity: 0.9; font-size: 1.1em; }

        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .tab {
            padding: 10px 20px;
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab:hover { border-color: #11998e; }
        .tab.active { background: #11998e; color: white; border-color: #11998e; }

        .section { background: white; border-radius: 10px; padding: 25px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .section h2 { margin-bottom: 20px; color: #333; border-bottom: 2px solid #11998e; padding-bottom: 10px; }

        .category-view { display: none; }
        .category-view.active { display: block; }

        .guest-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; }
        .guest-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            transition: box-shadow 0.2s;
        }
        .guest-card:hover { box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        .guest-name { font-weight: bold; font-size: 1.1em; margin-bottom: 5px; }
        .guest-role { color: #666; font-size: 0.9em; margin-bottom: 10px; }
        .guest-companies { font-size: 0.85em; color: #888; margin-bottom: 10px; }
        .guest-themes { display: flex; flex-wrap: wrap; gap: 5px; }
        .mini-tag {
            background: #e8f8f5;
            color: #11998e;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75em;
        }

        .theme-breakdown { margin-top: 20px; }
        .theme-bar {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .theme-bar-label { width: 200px; font-size: 0.9em; }
        .theme-bar-fill {
            height: 20px;
            background: linear-gradient(90deg, #11998e, #38ef7d);
            border-radius: 3px;
            min-width: 5px;
        }
        .theme-bar-count { margin-left: 10px; font-size: 0.85em; color: #666; }

        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
        }
        .comparison-col { text-align: center; }
        .comparison-col h4 { margin-bottom: 15px; color: #666; }
        .comparison-list { list-style: none; }
        .comparison-list li {
            padding: 8px;
            margin-bottom: 5px;
            border-radius: 5px;
            font-size: 0.9em;
        }
        .comparison-col:first-child .comparison-list li { background: #e8f8f5; }
        .comparison-col:nth-child(2) .comparison-list li { background: #ffeaa7; }
        .comparison-col:last-child .comparison-list li { background: #dfe6e9; }

        .search-box {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 1em;
            margin-bottom: 20px;
        }
        .search-box:focus { outline: none; border-color: #11998e; }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .stat-card .number { font-size: 2em; font-weight: bold; color: #11998e; }
        .stat-card .label { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <header>
        <h1>Guest Expertise Mapping</h1>
        <p>What topics do guests from different backgrounds discuss?</p>
    </header>

    <div class="container">
        <div class="stats">
            <div class="stat-card">
                <div class="number">""" + str(expertise_data["metadata"]["total_episodes"]) + """</div>
                <div class="label">Total Episodes</div>
            </div>
"""

    for cat, count in sorted(expertise_data["metadata"]["company_type_distribution"].items(), key=lambda x: -x[1])[:4]:
        cat_display = cat.replace("_", " ").title()
        html += f"""            <div class="stat-card">
                <div class="number">{count}</div>
                <div class="label">{cat_display}</div>
            </div>
"""

    html += """        </div>

        <div class="tabs">
"""

    for i, cat in enumerate(expertise_data.get("guest_categories", {}).keys()):
        cat_display = cat.replace("_", " ").title()
        active = "active" if i == 0 else ""
        html += f"""            <div class="tab {active}" onclick="showCategory('{cat}')">{cat_display}</div>
"""

    html += """        </div>
"""

    for i, (cat, data) in enumerate(expertise_data.get("guest_categories", {}).items()):
        active = "active" if i == 0 else ""
        cat_display = cat.replace("_", " ").title()

        html += f"""        <div class="category-view {active}" id="category-{cat}">
            <div class="section">
                <h2>{cat_display} ({data.get("total_episodes", 0)} episodes)</h2>

                <h3 style="margin-bottom: 15px;">Top Themes Discussed</h3>
                <div class="theme-breakdown">
"""

        max_count = max(data.get("aggregate_themes", {}).values(), default=1)
        for theme, count in list(data.get("aggregate_themes", {}).items())[:10]:
            width = int((count / max_count) * 300)
            html += f"""                    <div class="theme-bar">
                        <span class="theme-bar-label">{theme}</span>
                        <div class="theme-bar-fill" style="width: {width}px;"></div>
                        <span class="theme-bar-count">{count}</span>
                    </div>
"""

        html += """                </div>

                <h3 style="margin: 25px 0 15px;">Guests</h3>
                <input type="text" class="search-box" placeholder="Search guests..." onkeyup="filterGuests(this, '""" + cat + """')">
                <div class="guest-grid" id="guests-""" + cat + """">
"""

        for guest in data.get("guests", []):
            companies = ", ".join(guest.get("companies", [])[:3])
            html += f"""                    <div class="guest-card" data-name="{guest.get('name', '').lower()}">
                        <div class="guest-name">{guest.get("name", "")}</div>
                        <div class="guest-role">{guest.get("current_role", "")[:60]}</div>
                        <div class="guest-companies">{companies}</div>
                        <div class="guest-themes">
"""
            for theme in guest.get("themes_discussed", [])[:4]:
                html += f"""                            <span class="mini-tag">{theme}</span>
"""
            html += """                        </div>
                    </div>
"""

        html += """                </div>
            </div>
        </div>
"""

    # Cross-category comparisons
    if expertise_data.get("cross_category_analysis"):
        html += """        <div class="section">
            <h2>Cross-Category Comparisons</h2>
"""

        for comp_name, comp_data in expertise_data.get("cross_category_analysis", {}).items():
            comp_display = comp_name.replace("_", " ").title()

            # Find the two unique keys dynamically from the data
            unique_keys = [k for k in comp_data.keys() if k.endswith("_unique")]
            if len(unique_keys) >= 2:
                key1, key2 = unique_keys[0], unique_keys[1]
                label1 = key1.replace("_unique", "").replace("_", " ").title()
                label2 = key2.replace("_unique", "").replace("_", " ").title()
            else:
                continue

            html += f"""            <h3 style="margin: 20px 0 15px;">{comp_display}</h3>
            <div class="comparison">
                <div class="comparison-col">
                    <h4>{label1} Focus</h4>
                    <ul class="comparison-list">
"""
            themes1 = comp_data.get(key1, [])
            if themes1:
                for theme in themes1[:5]:
                    html += f"""                        <li>{theme}</li>
"""
            else:
                html += """                        <li style="color: #999; font-style: italic;">No distinctive themes</li>
"""
            html += """                    </ul>
                </div>
                <div class="comparison-col">
                    <h4>Shared Topics</h4>
                    <ul class="comparison-list">
"""
            shared = comp_data.get("shared", [])
            if shared:
                for theme in shared[:5]:
                    html += f"""                        <li>{theme}</li>
"""
            else:
                html += """                        <li style="color: #999; font-style: italic;">No shared themes</li>
"""
            html += """                    </ul>
                </div>
                <div class="comparison-col">
                    <h4>{0} Focus</h4>
                    <ul class="comparison-list">
""".format(label2)

            themes2 = comp_data.get(key2, [])
            if themes2:
                for theme in themes2[:5]:
                    html += f"""                        <li>{theme}</li>
"""
            else:
                html += """                        <li style="color: #999; font-style: italic;">No distinctive themes</li>
"""
            html += """                    </ul>
                </div>
            </div>
"""

        html += """        </div>
"""

    html += """    </div>

    <script>
        function showCategory(cat) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.category-view').forEach(v => v.classList.remove('active'));
            document.querySelector(`.tab[onclick="showCategory('${cat}')"]`).classList.add('active');
            document.getElementById('category-' + cat).classList.add('active');
        }

        function filterGuests(input, cat) {
            const query = input.value.toLowerCase();
            document.querySelectorAll('#guests-' + cat + ' .guest-card').forEach(card => {
                const name = card.dataset.name;
                card.style.display = name.includes(query) ? 'block' : 'none';
            });
        }
    </script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


# ============================================================================
# Markdown Generation
# ============================================================================

def generate_evolution_markdown(evolution_data: dict, output_path: Path):
    """Generate narrative markdown report for topic evolution."""
    md = """# The Evolution of Lenny's Podcast Topics

*How themes and discussions have changed over time*

---

## Overview

"""

    meta = evolution_data.get("metadata", {})
    md += f"""- **Total Episodes Analyzed**: {meta.get("total_episodes", 0)}
- **Date Range**: {meta.get("date_range", "N/A")}
- **Quarters Covered**: {meta.get("quarters_covered", 0)}

---

## Key Findings

"""

    # Notable shifts section
    shifts = evolution_data.get("notable_shifts", [])
    if shifts:
        md += """### Notable Topic Shifts

"""
        for shift in shifts:
            emoji = "🚀" if shift.get("shift_type") == "emergence" else "📈"
            md += f"""{emoji} **{shift.get("theme", "")}**: {shift.get("description", "")}

"""

    # Theme trajectories
    md += """---

## Theme Trajectories

The following themes have shown significant presence across the podcast:

"""

    trajectories = evolution_data.get("theme_trajectories", {})
    sorted_themes = sorted(trajectories.items(), key=lambda x: -x[1].get("total_count", 0))

    for theme, data in sorted_themes[:15]:
        trajectory = data.get("trajectory", [])
        if trajectory:
            first = data.get("first_appearance", "Unknown")
            total = data.get("total_count", 0)
            recent_trend = ""

            if len(trajectory) >= 2:
                recent = trajectory[-1]["count"]
                prev = trajectory[-2]["count"]
                if recent > prev:
                    recent_trend = " ↑"
                elif recent < prev:
                    recent_trend = " ↓"
                else:
                    recent_trend = " →"

            md += f"""### {theme.title()}{recent_trend}

- First appeared: {first}
- Total episodes: {total}
- Recent trend: {trajectory[-1]["count"]} episodes in {trajectory[-1]["period"]}

"""

    # Timeline summary
    md += """---

## Quarterly Breakdown

"""

    timeline = evolution_data.get("timeline", [])
    for period in reversed(timeline):
        md += f"""### {period.get("period", "")} ({period.get("episode_count", 0)} episodes)

**Top themes**: {", ".join(t["theme"] for t in period.get("top_themes", [])[:5])}

**Notable guests**: {", ".join(ep["guest"] for ep in period.get("episodes", [])[:5])}

"""

    md += """---

*Generated automatically from podcast transcript analysis*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)


def generate_expertise_markdown(expertise_data: dict, output_path: Path):
    """Generate narrative markdown report for guest expertise."""
    md = """# Who Talks About What

*Understanding guest backgrounds and their discussion topics*

---

## Overview

"""

    meta = expertise_data.get("metadata", {})
    md += f"""- **Total Episodes Analyzed**: {meta.get("total_episodes", 0)}

### Guest Distribution by Background

"""

    dist = meta.get("company_type_distribution", {})
    for cat, count in sorted(dist.items(), key=lambda x: -x[1]):
        cat_display = cat.replace("_", " ").title()
        md += f"- **{cat_display}**: {count} episodes\n"

    md += """
---

## By Company Background

"""

    categories = expertise_data.get("guest_categories", {})

    for cat, data in sorted(categories.items(), key=lambda x: -x[1].get("total_episodes", 0)):
        cat_display = cat.replace("_", " ").title()
        md += f"""### {cat_display}

**{data.get("total_episodes", 0)} episodes**

**What they talk about most:**
"""
        for theme in data.get("top_themes", [])[:5]:
            count = data.get("aggregate_themes", {}).get(theme, 0)
            md += f"- {theme} ({count} episodes)\n"

        md += f"""
**Featured guests:**
"""
        for guest in data.get("guests", [])[:5]:
            companies = ", ".join(guest.get("companies", [])[:2])
            md += f"- **{guest.get('name', '')}** ({companies})\n"

        md += "\n"

    # Cross-category insights
    md += """---

## Cross-Category Insights

"""

    cross = expertise_data.get("cross_category_analysis", {})

    if "founders_vs_big_tech" in cross:
        comp = cross["founders_vs_big_tech"]
        md += """### Founders vs Big Tech Alumni

**Topics founders emphasize more:**
"""
        for t in comp.get("founder_unique", []):
            md += f"- {t}\n"

        md += """
**Topics Big Tech alumni emphasize more:**
"""
        for t in comp.get("big_tech_unique", []):
            md += f"- {t}\n"

        md += """
**Shared focus areas:**
"""
        for t in comp.get("shared", []):
            md += f"- {t}\n"

        md += "\n"

    if "consultants_vs_operators" in cross:
        comp = cross["consultants_vs_operators"]
        md += """### Consultants/Coaches vs Operators

**Topics consultants emphasize more:**
"""
        for t in comp.get("consultant_unique", []):
            md += f"- {t}\n"

        md += """
**Topics operators emphasize more:**
"""
        for t in comp.get("operator_unique", []):
            md += f"- {t}\n"

        md += "\n"

    md += """---

*Generated automatically from podcast transcript analysis*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze topic evolution and guest expertise")
    parser.add_argument(
        '--episodes-dir',
        type=str,
        default=DEFAULT_EPISODES_DIR,
        help='Path to episodes directory'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'evolution', 'expertise'],
        default='all',
        help='Analysis mode: all, evolution, or expertise'
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    episodes_dir = script_dir / args.episodes_dir

    if not episodes_dir.exists():
        print(f"Error: Episodes directory not found: {episodes_dir}")
        sys.exit(1)

    # Load all metadata
    print(f"Loading metadata from: {episodes_dir}")
    all_metadata = load_all_metadata(episodes_dir)
    print(f"Loaded {len(all_metadata)} episodes")

    # Check for publish dates
    with_dates = sum(1 for m in all_metadata if m.get("publish_date"))
    print(f"Episodes with publish dates: {with_dates}/{len(all_metadata)}")

    if args.mode in ['all', 'evolution']:
        print("\n--- Topic Evolution Analysis ---")
        evolution_data = analyze_topic_evolution(all_metadata)

        # Save JSON
        json_path = script_dir / "topic_evolution.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(evolution_data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {json_path}")

        # Generate HTML
        html_path = script_dir / "topic_evolution.html"
        generate_evolution_html(evolution_data, html_path)
        print(f"Saved: {html_path}")

        # Generate Markdown
        md_path = script_dir / "TOPIC_EVOLUTION.md"
        generate_evolution_markdown(evolution_data, md_path)
        print(f"Saved: {md_path}")

    if args.mode in ['all', 'expertise']:
        print("\n--- Guest Expertise Analysis ---")
        expertise_data = analyze_guest_expertise(all_metadata)

        # Save JSON
        json_path = script_dir / "guest_expertise_mapping.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(expertise_data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {json_path}")

        # Generate HTML
        html_path = script_dir / "guest_expertise_mapping.html"
        generate_expertise_html(expertise_data, html_path)
        print(f"Saved: {html_path}")

        # Generate Markdown
        md_path = script_dir / "GUEST_EXPERTISE.md"
        generate_expertise_markdown(expertise_data, md_path)
        print(f"Saved: {md_path}")

    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
