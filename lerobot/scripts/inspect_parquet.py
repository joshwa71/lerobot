#!/usr/bin/env python3

"""
Utility script to inspect the contents of a .parquet file.
Prints the first 10 rows of the file, similar to pandas df.head().

Usage:
    python lerobot/scripts/inspect_parquet.py path/to/episode_000000.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def inspect_parquet(file_path: Path, num_rows: int = 10) -> None:
    """
    Load and display the first few rows of a parquet file.
    
    Args:
        file_path: Path to the .parquet file
        num_rows: Number of rows to display (default: 10)
    """
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        print(f"Parquet file: {file_path}")
        print(f"Shape: {df.shape} (rows, columns)")
        print(f"Columns: {list(df.columns)}")
        print("\n" + "="*80)
        print(f"First {min(num_rows, len(df))} rows:")
        print("="*80)
        
        # Display the first n rows
        print(df.head(num_rows).to_string())
        
        # Display data types
        print("\n" + "="*80)
        print("Data types:")
        print("="*80)
        print(df.dtypes.to_string())
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a LeRobot dataset parquet file by displaying the first few rows."
    )
    parser.add_argument(
        "file_path", 
        type=str, 
        help="Path to the .parquet file to inspect"
    )
    parser.add_argument(
        "-n", "--num-rows",
        type=int,
        default=10,
        help="Number of rows to display (default: 10)"
    )
    
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    
    if not file_path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)
    
    if not file_path.suffix.lower() == '.parquet':
        print(f"Warning: File '{file_path}' does not have a .parquet extension.")
    
    inspect_parquet(file_path, args.num_rows)


if __name__ == "__main__":
    main() 