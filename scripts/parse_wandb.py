#!/usr/bin/env python3
"""Utility for parsing wandb run data from local .wandb binary files.

Designed for offline metrics analysis without requiring the wandb API.
Parses the LevelDB log format used by wandb to store run data locally.

Usage examples:
    # Load a run from a wandb directory
    run = WandbRun.from_wandb_dir("/path/to/outputs/run_name/wandb")

    # Load from a specific .wandb file
    run = WandbRun.from_file("/path/to/run-XXXX.wandb")

    # Get training metrics as a DataFrame
    df = run.history_df()

    # Get specific metrics
    loss = run.get_metric("train/loss")

    # Get system stats (GPU, CPU, memory)
    stats_df = run.system_stats_df()

    # Get run config
    config = run.config

    # Get final summary
    summary = run.summary

    # CLI: print summary of a run
    python parse_wandb.py /path/to/wandb/dir
    python parse_wandb.py /path/to/run-XXXX.wandb

    # CLI: export history to CSV
    python parse_wandb.py /path/to/wandb/dir --csv output.csv

    # CLI: list all metric keys
    python parse_wandb.py /path/to/wandb/dir --keys

    # CLI: print specific metrics
    python parse_wandb.py /path/to/wandb/dir --metrics train/loss train/lr
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any

# LevelDB log format constants
_HEADER_LEN = 7
_BLOCK_LEN = 32768
_FULL, _FIRST, _MIDDLE, _LAST = 1, 2, 3, 4

# Precomputed CRC seeds per record type
_CRC_SEEDS = [0] * 5
for _x in range(1, 5):
    _CRC_SEEDS[_x] = zlib.crc32(bytes(chr(_x), "iso8859-1")) & 0xFFFFFFFF


class _LevelDBParser:
    """Low-level parser for wandb's LevelDB log format (.wandb files)."""

    def __init__(self, fp):
        self._fp = fp
        self._index = _HEADER_LEN
        # Skip the 7-byte file header (ident + magic + version)
        header = fp.read(_HEADER_LEN)
        if len(header) < _HEADER_LEN:
            raise ValueError("File too small to be a valid .wandb file")
        ident = header[:4]
        if ident != b":W&B":
            raise ValueError(f"Invalid .wandb file (magic bytes: {ident!r})")

    def _read_record(self):
        """Read a single record from the current block."""
        offset = self._index % _BLOCK_LEN
        space_left = _BLOCK_LEN - offset
        if space_left < _HEADER_LEN:
            # Skip padding at end of block
            self._fp.read(space_left)
            self._index += space_left

        rec_header = self._fp.read(_HEADER_LEN)
        if len(rec_header) < _HEADER_LEN:
            return None

        checksum, dlength, dtype = struct.unpack("<IHB", rec_header)
        self._index += _HEADER_LEN
        data = self._fp.read(dlength)
        self._index += dlength

        checksum_computed = zlib.crc32(data, _CRC_SEEDS[dtype]) & 0xFFFFFFFF
        valid = checksum == checksum_computed
        return valid, dtype, data

    def read_data(self):
        """Read a complete data record (handles multi-block spanning).

        Returns (valid: bool, data: bytes) or None on EOF.
        """
        result = self._read_record()
        if result is None:
            return None

        valid, dtype, data = result
        if not valid:
            return False, data
        if dtype == _FULL:
            return True, data
        if dtype != _FIRST:
            return False, data

        # Multi-block record: read MIDDLE/LAST continuations
        while True:
            result = self._read_record()
            if result is None:
                return None
            valid, dtype, new_data = result
            if not valid:
                return False, data + new_data
            data += new_data
            if dtype == _LAST:
                break

        return True, data


def _try_import_pb():
    """Try to import wandb protobuf definitions."""
    try:
        from wandb.proto import wandb_internal_pb2

        return wandb_internal_pb2
    except ImportError:
        return None


@dataclass
class WandbRun:
    """Parsed wandb run data from local files.

    Attributes:
        history: List of dicts, one per logged step. Each dict maps metric
            names to values. Always includes '_step'.
        system_stats: List of dicts with system metrics (GPU, CPU, memory, etc.).
            Each dict includes '_timestamp'.
        config: Run configuration dict (from config.yaml or binary).
        summary: Final summary metrics dict (from wandb-summary.json or binary).
        run_id: The wandb run ID.
        run_dir: Path to the run directory.
    """

    history: list[dict[str, Any]] = field(default_factory=list)
    system_stats: list[dict[str, Any]] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    run_dir: str = ""

    @classmethod
    def from_wandb_dir(cls, wandb_dir: str) -> "WandbRun":
        """Load a run from a wandb/ directory.

        Finds the latest run directory and parses both the .wandb binary
        and companion JSON/YAML files.
        """
        wandb_dir = str(Path(wandb_dir).resolve())

        # Find run directories
        run_dirs = sorted(glob(os.path.join(wandb_dir, "run-*")))
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found in {wandb_dir}")

        # Use the latest run (or 'latest-run' symlink)
        latest_link = os.path.join(wandb_dir, "latest-run")
        if os.path.islink(latest_link):
            run_dir = os.path.realpath(latest_link)
        else:
            run_dir = run_dirs[-1]

        # Find the .wandb binary file
        wandb_files = glob(os.path.join(run_dir, "*.wandb"))
        if not wandb_files:
            raise FileNotFoundError(f"No .wandb file found in {run_dir}")

        run = cls.from_file(wandb_files[0])
        run.run_dir = run_dir

        # Augment with companion files
        files_dir = os.path.join(run_dir, "files")
        run._load_companion_files(files_dir)

        return run

    @classmethod
    def from_file(cls, wandb_file: str) -> "WandbRun":
        """Load a run from a .wandb binary file."""
        wandb_file = str(Path(wandb_file).resolve())
        run_dir = os.path.dirname(wandb_file)

        # Extract run ID from filename (run-XXXXX.wandb)
        fname = os.path.basename(wandb_file)
        run_id = fname.replace("run-", "").replace(".wandb", "")

        run = cls(run_id=run_id, run_dir=run_dir)

        pb = _try_import_pb()
        if pb is None:
            print(
                "Warning: wandb protobuf definitions not available. "
                "Only companion JSON/YAML files will be read. "
                "Install wandb to parse binary .wandb files."
            )
            files_dir = os.path.join(run_dir, "files")
            run._load_companion_files(files_dir)
            return run

        run._parse_binary(wandb_file, pb)

        # Also load companion files (they may have richer summary data)
        files_dir = os.path.join(run_dir, "files")
        run._load_companion_files(files_dir)

        return run

    def _parse_binary(self, filepath: str, pb) -> None:
        """Parse the .wandb binary file for history and stats."""
        with open(filepath, "rb") as fp:
            parser = _LevelDBParser(fp)

            while True:
                result = parser.read_data()
                if result is None:
                    break
                valid, data = result
                if not valid:
                    continue

                try:
                    record = pb.Record()
                    record.ParseFromString(data)
                    field_name = record.WhichOneof("record_type")
                except Exception:
                    continue

                if field_name == "history":
                    self._parse_history_record(record.history)
                elif field_name == "stats":
                    self._parse_stats_record(record.stats)
                elif field_name == "run":
                    self._parse_run_record(record.run)

        # Sort history by step
        self.history.sort(key=lambda x: x.get("_step", 0))

    def _parse_history_record(self, hist) -> None:
        """Extract metrics from a history protobuf record."""
        items = {}
        for item in hist.item:
            # In wandb 0.22+, actual key is in nested_key[0], key field is empty
            key = item.nested_key[0] if len(item.nested_key) > 0 else item.key
            if not key:
                continue
            try:
                val = json.loads(item.value_json)
            except (json.JSONDecodeError, TypeError):
                val = item.value_json
            items[key] = val

        # Step is stored in the step field, not in items
        step = hist.step.num
        items["_step"] = step
        self.history.append(items)

    def _parse_stats_record(self, stats) -> None:
        """Extract system stats from a stats protobuf record."""
        items = {}
        for item in stats.item:
            try:
                items[item.key] = json.loads(item.value_json)
            except (json.JSONDecodeError, TypeError):
                items[item.key] = item.value_json

        # Add timestamp
        ts = stats.timestamp
        items["_timestamp"] = ts.seconds + ts.nanos / 1e9
        self.system_stats.append(items)

    def _parse_run_record(self, run) -> None:
        """Extract run config from a run protobuf record."""
        if self.run_id == "" and run.run_id:
            self.run_id = run.run_id

    def _load_companion_files(self, files_dir: str) -> None:
        """Load config.yaml, wandb-summary.json, and wandb-metadata.json."""
        # wandb-summary.json (final metric values)
        summary_path = os.path.join(files_dir, "wandb-summary.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                self.summary = json.load(f)

        # config.yaml
        config_path = os.path.join(files_dir, "config.yaml")
        if os.path.exists(config_path):
            try:
                import yaml

                with open(config_path) as f:
                    raw = yaml.safe_load(f)
                # wandb config format: each key maps to {"value": ...}
                if raw and isinstance(raw, dict):
                    self.config = {
                        k: v.get("value", v) if isinstance(v, dict) and "value" in v else v
                        for k, v in raw.items()
                    }
            except ImportError:
                # Fall back to basic parsing without yaml
                pass

        # wandb-metadata.json
        meta_path = os.path.join(files_dir, "wandb-metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.config["_metadata"] = json.load(f)

    # ── Query methods ──────────────────────────────────────────────

    def history_df(self):
        """Return history as a pandas DataFrame indexed by step.

        Requires pandas.
        """
        import pandas as pd

        if not self.history:
            return pd.DataFrame()
        df = pd.DataFrame(self.history)
        if "_step" in df.columns:
            df = df.set_index("_step").sort_index()
        return df

    def system_stats_df(self):
        """Return system stats as a pandas DataFrame.

        Requires pandas.
        """
        import pandas as pd

        if not self.system_stats:
            return pd.DataFrame()
        return pd.DataFrame(self.system_stats)

    def get_metric(self, key: str) -> list[tuple[int, Any]]:
        """Get a specific metric across all steps.

        Returns list of (step, value) tuples, skipping steps where
        the metric is not present.
        """
        result = []
        for entry in self.history:
            if key in entry:
                result.append((entry.get("_step", 0), entry[key]))
        return result

    def metric_keys(self, prefix: str | None = None) -> list[str]:
        """Get all unique metric keys from history.

        Args:
            prefix: Optional prefix filter (e.g. "train/", "eval/",
                "memory_stats/").
        """
        keys = set()
        for entry in self.history:
            keys.update(entry.keys())
        # Remove internal keys
        keys -= {"_step", "_runtime", "_timestamp"}
        if prefix:
            keys = {k for k in keys if k.startswith(prefix)}
        return sorted(keys)

    def steps(self) -> list[int]:
        """Get all step numbers."""
        return [h["_step"] for h in self.history if "_step" in h]

    def final_metrics(self) -> dict[str, Any]:
        """Get the final value of each metric (from summary or last history)."""
        if self.summary:
            return dict(self.summary)
        if self.history:
            return dict(self.history[-1])
        return {}

    def media_files(self) -> list[str]:
        """List media files (images, videos) in the run directory."""
        media_dir = os.path.join(self.run_dir, "files", "media")
        if not os.path.isdir(media_dir):
            return []
        files = []
        for root, _, filenames in os.walk(media_dir):
            for f in filenames:
                files.append(os.path.join(root, f))
        return sorted(files)

    def __repr__(self) -> str:
        n_steps = len(self.history)
        n_keys = len(self.metric_keys()) if self.history else 0
        n_stats = len(self.system_stats)
        step_range = ""
        if self.history:
            steps = self.steps()
            step_range = f", steps={steps[0]}-{steps[-1]}"
        return (
            f"WandbRun(id={self.run_id!r}, "
            f"history={n_steps} records, "
            f"metrics={n_keys} keys{step_range}, "
            f"system_stats={n_stats} records)"
        )


# ── CLI ──────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse wandb local run data")
    parser.add_argument("path", help="Path to wandb/ dir or .wandb file")
    parser.add_argument("--csv", help="Export history to CSV file")
    parser.add_argument("--keys", action="store_true", help="List all metric keys")
    parser.add_argument(
        "--metrics", nargs="+", help="Print specific metrics (e.g. train/loss train/lr)"
    )
    parser.add_argument("--stats", action="store_true", help="Show system stats summary")
    parser.add_argument(
        "--json", action="store_true", help="Output final summary as JSON"
    )
    args = parser.parse_args()

    path = args.path
    if path.endswith(".wandb"):
        run = WandbRun.from_file(path)
    else:
        run = WandbRun.from_wandb_dir(path)

    if args.keys:
        print("Metric keys:")
        for prefix in ["train/", "eval/", "memory_iou/", "memory_stats/"]:
            keys = run.metric_keys(prefix=prefix)
            if keys:
                print(f"\n  [{prefix}] ({len(keys)} keys)")
                for k in keys:
                    print(f"    {k}")
        other = [
            k
            for k in run.metric_keys()
            if not any(
                k.startswith(p) for p in ["train/", "eval/", "memory_iou/", "memory_stats/"]
            )
        ]
        if other:
            print(f"\n  [other] ({len(other)} keys)")
            for k in other:
                print(f"    {k}")
        return

    if args.csv:
        df = run.history_df()
        df.to_csv(args.csv)
        print(f"Exported {len(df)} rows x {len(df.columns)} columns to {args.csv}")
        return

    if args.metrics:
        for key in args.metrics:
            values = run.get_metric(key)
            print(f"\n{key} ({len(values)} points):")
            for step, val in values:
                if isinstance(val, float):
                    print(f"  step {step:>6d}: {val:.6f}")
                else:
                    print(f"  step {step:>6d}: {val}")
        return

    if args.stats:
        df = run.system_stats_df()
        gpu_cols = [c for c in df.columns if c.startswith("gpu.")]
        if gpu_cols:
            print("GPU stats summary:")
            print(df[gpu_cols].describe().to_string())
        mem_cols = [c for c in df.columns if "memory" in c.lower() or "mem" in c.lower()]
        if mem_cols:
            print("\nMemory stats summary:")
            print(df[mem_cols].describe().to_string())
        return

    if args.json:
        print(json.dumps(run.final_metrics(), indent=2, default=str))
        return

    # Default: print summary
    print(run)
    print(f"\nConfig keys: {len(run.config)}")
    print(f"Summary keys: {len(run.summary)}")
    print(f"Media files: {len(run.media_files())}")

    if run.history:
        print(f"\nStep range: {run.steps()[0]} - {run.steps()[-1]}")
        print(f"Step interval: {run.steps()[1] - run.steps()[0] if len(run.steps()) > 1 else 'N/A'}")

    # Show final values of key training metrics
    final = run.final_metrics()
    key_metrics = ["train/loss", "train/lr", "train/grad_norm", "eval/avg_pc_success_seen"]
    print("\nKey final metrics:")
    for k in key_metrics:
        if k in final:
            v = final[k]
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
