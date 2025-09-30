#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Visualize and explore parquet files with various viewing options.

This script provides comprehensive tools for examining parquet files including:
- Schema and metadata inspection
- Data statistics and summaries  
- Sample data viewing
- Column-specific analysis
- Basic plotting capabilities

Examples:

- Show basic info about a parquet file:
```
python visualize_parquet.py --file path/to/file.parquet --info
```

- Display first 10 rows:
```
python visualize_parquet.py --file path/to/file.parquet --head 10
```

- Show statistics for numeric columns:
```
python visualize_parquet.py --file path/to/file.parquet --stats
```

- Explore specific columns:
```
python visualize_parquet.py --file path/to/file.parquet --columns action timestamp --head 5
```

- Plot histograms for numeric columns:
```
python visualize_parquet.py --file path/to/file.parquet --plot --columns episode_index
```
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def print_file_info(parquet_file: pq.ParquetFile, file_path: Path) -> None:
    """Print basic information about the parquet file."""
    print(f"\n📁 File: {file_path}")
    print(f"📏 Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"🗂️ Number of row groups: {parquet_file.num_row_groups}")
    print(f"📊 Total rows: {parquet_file.metadata.num_rows:,}")
    
    # Schema information
    schema = parquet_file.schema_arrow
    print(f"\n📋 Schema ({len(schema)} columns):")
    for i, field in enumerate(schema):
        print(f"  {i+1:2d}. {field.name:25s} {str(field.type):20s}")


def print_metadata(parquet_file: pq.ParquetFile) -> None:
    """Print detailed metadata about the parquet file."""
    metadata = parquet_file.metadata
    print(f"\n🔍 Metadata:")
    print(f"  Created by: {metadata.created_by}")
    print(f"  Format version: {metadata.format_version}")
    print(f"  Serialized size: {metadata.serialized_size:,} bytes")
    
    if metadata.metadata:
        print(f"  Custom metadata: {len(metadata.metadata)} entries")
        for key, value in metadata.metadata.items():
            print(f"    {key}: {value}")


def print_row_group_info(parquet_file: pq.ParquetFile, max_groups: int = 5) -> None:
    """Print information about row groups."""
    print(f"\n📦 Row Groups (showing first {min(max_groups, parquet_file.num_row_groups)}):")
    
    for i in range(min(max_groups, parquet_file.num_row_groups)):
        rg = parquet_file.metadata.row_group(i)
        print(f"  Group {i}: {rg.num_rows:,} rows, {rg.total_byte_size:,} bytes")


def display_head(df: pd.DataFrame, n_rows: int, columns: Optional[List[str]] = None) -> None:
    """Display the first n rows of the dataframe."""
    if columns:
        available_cols = [col for col in columns if col in df.columns]
        if not available_cols:
            print(f"❌ None of the specified columns {columns} found in data")
            return
        df_display = df[available_cols]
    else:
        df_display = df
    
    print(f"\n📄 First {n_rows} rows:")
    print(df_display.head(n_rows).to_string())


def display_stats(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """Display statistics for numeric columns."""
    if columns:
        available_cols = [col for col in columns if col in df.columns]
        if not available_cols:
            print(f"❌ None of the specified columns {columns} found in data")
            return
        df_stats = df[available_cols]
    else:
        df_stats = df
    
    # Select only numeric columns
    numeric_cols = df_stats.select_dtypes(include=[np.number])
    
    if numeric_cols.empty:
        print("ℹ️ No numeric columns found for statistics")
        return
    
    print(f"\n📈 Statistics for numeric columns:")
    print(numeric_cols.describe().to_string())
    
    # Show data types for all selected columns
    print(f"\n🏷️ Data types:")
    for col in df_stats.columns:
        dtype = df_stats[col].dtype
        non_null = df_stats[col].count()
        total = len(df_stats[col])
        print(f"  {col:25s} {str(dtype):15s} ({non_null}/{total} non-null)")


def plot_distributions(df: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """Create simple plots for numeric columns."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("❌ matplotlib not available. Install with: pip install matplotlib")
        return
    
    if columns:
        available_cols = [col for col in columns if col in df.columns]
        if not available_cols:
            print(f"❌ None of the specified columns {columns} found in data")
            return
        df_plot = df[available_cols]
    else:
        df_plot = df
    
    # Select only numeric columns
    numeric_cols = df_plot.select_dtypes(include=[np.number])
    
    if numeric_cols.empty:
        print("ℹ️ No numeric columns found for plotting")
        return
    
    n_cols = len(numeric_cols.columns)
    n_rows = (n_cols + 2) // 3  # 3 columns per row
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols.columns):
        ax = axes[i] if n_cols > 1 else axes
        numeric_cols[col].hist(bins=50, ax=ax, alpha=0.7)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')
    
    # Hide unused subplots
    for i in range(n_cols, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    print(f"\n📊 Displaying histograms for {n_cols} numeric columns")
    plt.show()


def explore_column_values(df: pd.DataFrame, column: str, max_unique: int = 20) -> None:
    """Explore unique values in a specific column."""
    if column not in df.columns:
        print(f"❌ Column '{column}' not found in data")
        return
    
    series = df[column]
    print(f"\n🔍 Exploring column '{column}':")
    print(f"  Data type: {series.dtype}")
    print(f"  Total values: {len(series):,}")
    print(f"  Non-null values: {series.count():,}")
    print(f"  Unique values: {series.nunique():,}")
    
    if series.nunique() <= max_unique:
        print(f"  Value counts:")
        value_counts = series.value_counts()
        for value, count in value_counts.items():
            percentage = (count / len(series)) * 100
            print(f"    {value}: {count:,} ({percentage:.1f}%)")
    else:
        print(f"  Top {max_unique} most frequent values:")
        value_counts = series.value_counts().head(max_unique)
        for value, count in value_counts.items():
            percentage = (count / len(series)) * 100
            print(f"    {value}: {count:,} ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize and explore parquet files", 
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--file", 
        type=Path, 
        required=True,
        help="Path to the parquet file to visualize"
    )
    
    parser.add_argument(
        "--info", 
        action="store_true",
        help="Show basic file information and schema"
    )
    
    parser.add_argument(
        "--metadata", 
        action="store_true",
        help="Show detailed metadata"
    )
    
    parser.add_argument(
        "--head", 
        type=int, 
        default=0,
        help="Show first N rows (default: 0, don't show)"
    )
    
    parser.add_argument(
        "--stats", 
        action="store_true",
        help="Show statistics for numeric columns"
    )
    
    parser.add_argument(
        "--plot", 
        action="store_true",
        help="Create histograms for numeric columns"
    )
    
    parser.add_argument(
        "--columns", 
        nargs="+",
        help="Specific columns to focus on (for --head, --stats, --plot)"
    )
    
    parser.add_argument(
        "--explore", 
        type=str,
        help="Explore unique values in a specific column"
    )
    
    parser.add_argument(
        "--max-rows", 
        type=int, 
        default=None,
        help="Maximum number of rows to load (useful for large files)"
    )
    
    parser.add_argument(
        "--row-groups", 
        action="store_true",
        help="Show information about row groups"
    )

    args = parser.parse_args()
    
    # Check if file exists
    if not args.file.exists():
        print(f"❌ File not found: {args.file}")
        sys.exit(1)
    
    # If no action specified, show info by default
    if not any([args.info, args.metadata, args.head, args.stats, args.plot, args.explore, args.row_groups]):
        args.info = True
    
    try:
        # Open parquet file
        parquet_file = pq.ParquetFile(args.file)
        
        # Show basic info
        if args.info:
            print_file_info(parquet_file, args.file)
        
        # Show metadata
        if args.metadata:
            print_metadata(parquet_file)
        
        # Show row group info
        if args.row_groups:
            print_row_group_info(parquet_file)
        
        # For data operations, load into pandas
        if args.head or args.stats or args.plot or args.explore:
            print("\n🔄 Loading data...")
            
            # Load data with optional row limit
            if args.max_rows:
                # Read in batches to limit memory usage
                batch_size = min(args.max_rows, 10000)
                batches = []
                rows_read = 0
                
                for batch in parquet_file.iter_batches(batch_size=batch_size):
                    batch_df = batch.to_pandas()
                    rows_needed = args.max_rows - rows_read
                    if len(batch_df) > rows_needed:
                        batch_df = batch_df.head(rows_needed)
                    batches.append(batch_df)
                    rows_read += len(batch_df)
                    if rows_read >= args.max_rows:
                        break
                
                df = pd.concat(batches, ignore_index=True)
                print(f"✅ Loaded {len(df):,} rows (limited to {args.max_rows:,})")
            else:
                df = parquet_file.read().to_pandas()
                print(f"✅ Loaded {len(df):,} rows")
            
            # Display head
            if args.head:
                display_head(df, args.head, args.columns)
            
            # Show statistics
            if args.stats:
                display_stats(df, args.columns)
            
            # Create plots
            if args.plot:
                plot_distributions(df, args.columns)
            
            # Explore specific column
            if args.explore:
                explore_column_values(df, args.explore)
    
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
