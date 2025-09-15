#!/usr/bin/env python

import argparse
import json
from pathlib import Path
from typing import List

import torch


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def build_video_path(video_path_tmpl: str, episode_index: int, video_key: str, chunks_size: int) -> Path:
    chunk = episode_index // chunks_size
    rel = video_path_tmpl.format(episode_chunk=chunk, video_key=video_key, episode_index=episode_index)
    return Path(rel)


def try_decode_timestamps_torchcodec(video_path: Path, timestamps: List[float], tolerance_s: float) -> bool:
    try:
        from torchcodec.decoders import VideoDecoder
        decoder = VideoDecoder(str(video_path), device="cpu", seek_mode="approximate")
        # Use metadata FPS to convert timestamps to indices
        fps = float(decoder.metadata.average_fps)
        indices = [round(ts * fps) for ts in timestamps]
        frames = decoder.get_frames_at(indices=indices)
        # Basic shape sanity check; ensure number of frames matches and tensors are present
        if frames.data is None or len(frames.data) != len(indices):
            return False
        # Optionally verify tolerance; we allow approximate
        pts = torch.tensor(frames.pts_seconds)
        q = torch.tensor(timestamps)
        dist = torch.cdist(q[:, None], pts[:, None], p=1)
        min_, _ = dist.min(1)
        return torch.all(min_ < tolerance_s).item()
    except Exception:
        return False


def try_decode_timestamps_torchvision(video_path: Path, timestamps: List[float], tolerance_s: float) -> bool:
    try:
        import torchvision

        # Prefer video_reader if available, else pyav
        backend = "video_reader"
        try:
            torchvision.set_video_backend(backend)
        except Exception:
            backend = "pyav"
            torchvision.set_video_backend(backend)

        reader = torchvision.io.VideoReader(str(video_path), "video")
        first_ts = min(timestamps)
        last_ts = max(timestamps)
        keyframes_only = backend == "pyav"
        reader.seek(first_ts, keyframes_only=keyframes_only)

        loaded_ts = []
        for frame in reader:
            loaded_ts.append(frame["pts"])
            if frame["pts"] >= last_ts:
                break

        if backend == "pyav":
            # Force-close underlying container
            reader.container.close()
        reader = None

        if not loaded_ts:
            return False
        q = torch.tensor(timestamps)
        l = torch.tensor(loaded_ts)
        dist = torch.cdist(q[:, None], l[:, None], p=1)
        min_, _ = dist.min(1)
        return torch.all(min_ < tolerance_s).item()
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate video decoding for a LeRobot dataset")
    parser.add_argument("dataset_root", type=str, help="Path to dataset root (contains meta/, data/, videos/)")
    parser.add_argument("--backend", type=str, default="torchcodec", choices=["torchcodec", "torchvision"], help="Decoder to test first")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    meta_dir = root / "meta"
    info = load_json(meta_dir / "info.json")

    video_keys = [k for k, ft in info["features"].items() if ft.get("dtype") == "video"]
    if not video_keys:
        print("No video keys in dataset.")
        return

    chunks_size = int(info["chunks_size"]) if "chunks_size" in info else 1000
    fps = int(info["fps"])
    tolerance_s = 1.0 / fps - 1e-4

    # Load episodes.jsonl
    episodes_path = meta_dir / "episodes.jsonl"
    episodes = []
    with episodes_path.open("r") as f:
        for line in f:
            episodes.append(json.loads(line))

    total = len(episodes) * len(video_keys)
    print(f"Found {len(episodes)} episodes, {len(video_keys)} video keys; total {total} mp4s")

    corrupt = []
    checked = 0
    for ep in episodes:
        ep_idx = int(ep["episode_index"]) if isinstance(ep, dict) else int(ep.episode_index)
        ep_len = int(ep["length"]) if isinstance(ep, dict) else int(ep.length)
        if ep_len <= 0:
            if args.verbose:
                print(f"Episode {ep_idx} has non-positive length; skipping")
            continue

        # Sample first/middle/last timestamps
        first_ts = 0.0
        mid_ts = max(0.0, (ep_len // 2) / fps)
        last_ts = max(0.0, (ep_len - 1) / fps)
        timestamps = [first_ts, mid_ts, last_ts]

        for vkey in video_keys:
            rel = build_video_path(info["video_path"], ep_idx, vkey, chunks_size)
            vpath = root / rel
            if not vpath.is_file():
                corrupt.append((ep_idx, vkey, str(vpath), "missing"))
                if args.verbose:
                    print(f"[MISSING] ep={ep_idx} key={vkey} path={vpath}")
                continue

            ok = False
            if args.backend == "torchcodec":
                ok = try_decode_timestamps_torchcodec(vpath, timestamps, tolerance_s)
                if not ok:
                    # fallback
                    ok = try_decode_timestamps_torchvision(vpath, timestamps, tolerance_s)
            else:
                ok = try_decode_timestamps_torchvision(vpath, timestamps, tolerance_s)
                if not ok:
                    ok = try_decode_timestamps_torchcodec(vpath, timestamps, tolerance_s)

            checked += 1
            if not ok:
                corrupt.append((ep_idx, vkey, str(vpath), "decode_error"))
                if args.verbose:
                    print(f"[DECODE_ERROR] ep={ep_idx} key={vkey} path={vpath}")

    print(f"Checked {checked}/{total} videos")
    if corrupt:
        print(f"Found {len(corrupt)} problematic videos:")
        for ep_idx, vkey, vpath, reason in corrupt:
            print(f" - episode={ep_idx} key={vkey} reason={reason} path={vpath}")
        exit(1)
    else:
        print("All videos decoded successfully.")


if __name__ == "__main__":
    main()


