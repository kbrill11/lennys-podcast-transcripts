#!/usr/bin/env python3
"""
Fetch YouTube Upload Dates

Fetches the upload/publish date for each episode's YouTube video using yt-dlp
and updates the metadata.json files with the publish_date field.

Usage:
    python fetch_youtube_dates.py [--limit N] [--skip-existing] [--delay SECONDS]

Requirements:
    pip install yt-dlp
"""

import os
import sys
import json
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


DEFAULT_EPISODES_DIR = "episodes"
METADATA_FILENAME = "metadata.json"


def get_video_id_from_url(url: str) -> str | None:
    """Extract video ID from YouTube URL."""
    if not url:
        return None

    # Handle various YouTube URL formats
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    elif "/embed/" in url:
        return url.split("/embed/")[1].split("?")[0]
    return None


def fetch_upload_date(video_id: str) -> str | None:
    """Fetch upload date for a YouTube video using yt-dlp."""
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--dump-json",
                "--no-download",
                "--no-warnings",
                f"https://www.youtube.com/watch?v={video_id}"
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        upload_date = data.get("upload_date")  # Format: YYYYMMDD

        if upload_date:
            # Convert to YYYY-MM-DD format
            return f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"

        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"Error fetching date for {video_id}: {e}")
        return None


def load_metadata(episode_dir: Path) -> dict | None:
    """Load metadata from episode directory."""
    metadata_path = episode_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_metadata(episode_dir: Path, metadata: dict):
    """Save metadata to episode directory."""
    metadata_path = episode_dir / METADATA_FILENAME
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def discover_episodes(episodes_dir: Path) -> list[Path]:
    """Discover all episode directories with metadata."""
    episodes = []
    for item in sorted(episodes_dir.iterdir()):
        if item.is_dir() and (item / METADATA_FILENAME).exists():
            episodes.append(item)
    return episodes


def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube upload dates for podcast episodes")
    parser.add_argument(
        '--episodes-dir',
        type=str,
        default=DEFAULT_EPISODES_DIR,
        help='Path to episodes directory'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of episodes to process (for testing)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip episodes that already have publish_date'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay between requests in seconds (default: 1.0)'
    )

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    episodes_dir = script_dir / args.episodes_dir

    if not episodes_dir.exists():
        print(f"Error: Episodes directory not found: {episodes_dir}")
        sys.exit(1)

    # Check if yt-dlp is available
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: yt-dlp is not installed. Install with: pip install yt-dlp")
        sys.exit(1)

    # Discover episodes
    print(f"Discovering episodes in: {episodes_dir}")
    episode_dirs = discover_episodes(episodes_dir)
    print(f"Found {len(episode_dirs)} episodes with metadata")

    if args.limit:
        episode_dirs = episode_dirs[:args.limit]
        print(f"Limited to {len(episode_dirs)} episodes")

    # Process episodes
    updated = 0
    skipped = 0
    failed = 0

    for episode_dir in tqdm(episode_dirs, desc="Fetching dates"):
        metadata = load_metadata(episode_dir)
        if not metadata:
            failed += 1
            continue

        # Check if already has publish_date
        if args.skip_existing and metadata.get("publish_date"):
            skipped += 1
            continue

        # Get video ID
        youtube_url = metadata.get("youtube_url", "")
        video_id = get_video_id_from_url(youtube_url)

        if not video_id:
            tqdm.write(f"  No video ID for {episode_dir.name}")
            failed += 1
            continue

        # Fetch upload date
        upload_date = fetch_upload_date(video_id)

        if upload_date:
            metadata["publish_date"] = upload_date
            save_metadata(episode_dir, metadata)
            updated += 1
        else:
            tqdm.write(f"  Failed to fetch date for {episode_dir.name}")
            failed += 1

        # Rate limiting
        if args.delay > 0:
            time.sleep(args.delay)

    # Summary
    print("\n" + "=" * 50)
    print("FETCH COMPLETE")
    print("=" * 50)
    print(f"Updated: {updated}")
    print(f"Skipped (already had date): {skipped}")
    print(f"Failed: {failed}")


if __name__ == '__main__':
    main()
