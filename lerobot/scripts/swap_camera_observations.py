#!/usr/bin/env python3
"""
Script to swap head and wrist camera observations in a LeRobot dataset.
This handles both video directory names and statistics in episodes_stats.jsonl.
"""

import argparse
import json
import shutil
from pathlib import Path


def load_jsonl(file_path: Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""
    data = []
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], file_path: Path) -> None:
    """Save a list of dictionaries to a JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def swap_video_directories(dataset_path: Path) -> None:
    """Swap observation.images.head and observation.images.wrist video directories."""
    videos_path = dataset_path / "videos"
    
    if not videos_path.exists():
        print(f"No videos directory found at {videos_path}")
        return
    
    # Process each chunk directory
    for chunk_dir in sorted(videos_path.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
            
        head_dir = chunk_dir / "observation.images.head"
        wrist_dir = chunk_dir / "observation.images.wrist"
        temp_dir = chunk_dir / "_temp_swap_dir"
        
        # Check if both directories exist
        if not head_dir.exists() and not wrist_dir.exists():
            print(f"Warning: Neither head nor wrist directories found in {chunk_dir}")
            continue
        
        print(f"Swapping video directories in {chunk_dir}...")
        
        # Perform the swap using a temporary directory
        if head_dir.exists() and wrist_dir.exists():
            # Both exist - swap them
            shutil.move(str(head_dir), str(temp_dir))
            shutil.move(str(wrist_dir), str(head_dir))
            shutil.move(str(temp_dir), str(wrist_dir))
        elif head_dir.exists():
            # Only head exists - rename to wrist
            shutil.move(str(head_dir), str(wrist_dir))
        elif wrist_dir.exists():
            # Only wrist exists - rename to head
            shutil.move(str(wrist_dir), str(head_dir))
        
        print(f"  ✓ Swapped directories in {chunk_dir}")


def swap_episodes_stats(dataset_path: Path) -> None:
    """Swap observation.images.head and observation.images.wrist in episodes_stats.jsonl."""
    stats_file = dataset_path / "meta" / "episodes_stats.jsonl"
    
    if not stats_file.exists():
        print(f"episodes_stats.jsonl not found at {stats_file}")
        return
    
    print(f"Updating episodes_stats.jsonl...")
    
    # Load the stats
    episodes_stats = load_jsonl(stats_file)
    
    # Swap the keys for each episode
    for episode in episodes_stats:
        if "stats" in episode:
            stats = episode["stats"]
            
            # Check if both keys exist
            head_key = "observation.images.head"
            wrist_key = "observation.images.wrist"
            
            head_stats = stats.get(head_key)
            wrist_stats = stats.get(wrist_key)
            
            if head_stats is not None and wrist_stats is not None:
                # Both exist - swap them
                stats[head_key] = wrist_stats
                stats[wrist_key] = head_stats
            elif head_stats is not None:
                # Only head exists - move to wrist
                stats[wrist_key] = head_stats
                del stats[head_key]
            elif wrist_stats is not None:
                # Only wrist exists - move to head
                stats[head_key] = wrist_stats
                del stats[wrist_key]
    
    # Save the updated stats
    save_jsonl(episodes_stats, stats_file)
    print(f"  ✓ Updated {len(episodes_stats)} episode statistics")


def main():
    parser = argparse.ArgumentParser(
        description="Swap head and wrist camera observations in a LeRobot dataset."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the LeRobot dataset directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually making changes"
    )
    
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path).resolve()
    
    # Validate dataset path
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist")
        return 1
    
    if not (dataset_path / "meta").exists():
        print(f"Error: {dataset_path} does not appear to be a valid LeRobot dataset (missing meta directory)")
        return 1
    
    print(f"Processing dataset: {dataset_path}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")
        # TODO: Implement dry run mode that shows what would be changed
        print("Dry run mode not yet implemented. Run without --dry-run to make changes.")
        return 0
    
    # Create a backup of episodes_stats.jsonl before modifying
    stats_file = dataset_path / "meta" / "episodes_stats.jsonl"
    if stats_file.exists():
        backup_file = stats_file.with_suffix(".jsonl.backup")
        shutil.copy2(stats_file, backup_file)
        print(f"Created backup: {backup_file}")
    
    try:
        # Swap video directories
        swap_video_directories(dataset_path)
        
        # Swap statistics
        swap_episodes_stats(dataset_path)
        
        print(f"\n✓ Successfully swapped head and wrist observations in {dataset_path}")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        
        # Attempt to restore backup if something went wrong
        if stats_file.exists() and backup_file.exists():
            print("Restoring episodes_stats.jsonl from backup...")
            shutil.copy2(backup_file, stats_file)
            
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 