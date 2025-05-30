# encode_dataset.py — offline embedding for LeRobot‑style datasets
# ---------------------------------------------------------------
# Usage (example):
#   python encode_dataset.py \
#       --root /path/to/lerobot \
#       --model dinov2_vitb14 \
#       --batch 64 \
#       --proj-dim 512
#
# Mirrors `videos/` → `embeddings/`, writing one <frame_XXX.npy> (float16) per
# frame.  Now robust to occasional “corrupted” MP4s: if decord cannot open a
# video it automatically falls back to OpenCV and will skip empty/invalid files
# rather than crashing.

import argparse
import os
from pathlib import Path
import sys
import warnings
from typing import List

import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tqdm import tqdm

# ---------------------------------------------------------------
# Optional fast video backend: decord (falls back to cv2 automatically)
# ---------------------------------------------------------------
decord = None
DECORDError = Exception  # placeholder
import cv2
warnings.warn("[WARN] decord not installed — using OpenCV (slower)")

# ---------------------------------------------------------------
# Model + pre-processing helpers
# ---------------------------------------------------------------

def _dinov2(name: str, device: torch.device):
    model = torch.hub.load("facebookresearch/dinov2", name).to(device)
    model.eval()
    tr = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, tr, model.embed_dim


def _siglip2(name: str, device: torch.device):
    import open_clip  # local import to keep deps optional
    pretrained = "webli" if "/" not in name and name.lower().startswith("vit") else None
    model, _, preprocess = open_clip.create_model_and_transforms(name,
                                                                 pretrained=pretrained or "laion400m",
                                                                 device=device)
    model.eval()
    return model.visual, preprocess, model.visual.output_dim


def load_encoder(model_name: str, device: torch.device):
    m = model_name.lower()
    if m.startswith("dino"):
        return _dinov2(model_name, device)
    return _siglip2(model_name, device)

# ---------------------------------------------------------------
# Video I/O helpers with robust fallback
# ---------------------------------------------------------------

def _read_video_decord(path: Path):
    vr = decord.VideoReader(str(path))
    frames = vr.get_batch(range(len(vr)))  # (T, H, W, 3) RGB uint8
    return (frames.permute(0, 3, 1, 2).float() / 255.0).tolist()  # List[C×H×W]


def _read_video_cv2(path: Path):
    cap = cv2.VideoCapture(str(path))
    ok, out = True, []
    while ok:
        ok, frame_bgr = cap.read()
        if ok and frame_bgr is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            out.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    cap.release()
    return out


def read_video(path: Path):
    """Return list[Tensor C×H×W] or [] on failure."""

    try:
        import av
        frames = []
        with av.open(str(path)) as container:
            for fr in container.decode(video=0):
                img = fr.to_ndarray(format='rgb24')
                frames.append(torch.from_numpy(img)
                               .permute(2,0,1).float() / 255.0)
        return frames
    except Exception as e:
        warnings.warn(f"[pyav] {path.name}: {e} — skipping file.")
        return []

# ---------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------

def encode_frames(frames: List[torch.Tensor], model: nn.Module, transform, device,
                  batch_size: int, projector: nn.Module | None = None):
    embeds: List[torch.Tensor] = []
    for i in range(0, len(frames), batch_size):
        chunk = frames[i:i + batch_size]
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch = torch.stack([transform(f) for f in chunk]).to(device)
            feats = model(batch)
            if isinstance(feats, tuple):  # open_clip returns (img_emb, ...)
                feats = feats[0]
            if projector is not None:
                feats = projector(feats)
            embeds.extend(feats.cpu())
    return embeds

# ---------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------

def mirror_path(src_file: Path, src_root: Path, dst_root: Path, ext: str = "") -> Path:
    rel = src_file.relative_to(src_root)
    dst = dst_root / rel
    if ext:
        dst = dst.with_suffix(ext)
    return dst

# ---------------------------------------------------------------
# Main dataset traversal / processing
# ---------------------------------------------------------------

def process_dataset(root: Path, model_name: str, batch: int, proj_dim: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform, feat_dim = load_encoder(model_name, device)

    projector: nn.Module | None = None
    out_dim = feat_dim
    if 0 < proj_dim != feat_dim:
        projector = nn.Linear(feat_dim, proj_dim, bias=False).to(device)
        nn.init.normal_(projector.weight, std=feat_dim ** -0.5)
        out_dim = proj_dim

    print(f"[INFO] Using {model_name} → {out_dim}-D, batch {batch}, device {device}")

    vids_root = root / "videos"
    out_root = root / "embeddings"

    mp4_files = list(vids_root.rglob("*.mp4"))
    with tqdm(mp4_files, desc="Encoding videos") as tbar:
        for mp4 in tbar:
            frames = read_video(mp4)
            if not frames:
                continue  # skip empty / unreadable video

            embeds = encode_frames(frames, model, transform, device, batch, projector)
            dst_dir = mirror_path(mp4.parent, vids_root, out_root)
            dst_ep_root = dst_dir / mp4.stem
            dst_ep_root.mkdir(parents=True, exist_ok=True)
            for idx, emb in enumerate(embeds):
                np.save(dst_ep_root / f"frame_{idx}.npy", emb.numpy().astype(np.float16))

# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("Offline video frame embedding for LeRobot datasets")
    p.add_argument("--root", type=Path, required=True, help="Dataset root (with videos/)")
    p.add_argument("--model", type=str, default="dinov2_vitl14", help="dinov2_* or ViT-B-16-SigLIP2 etc.")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--proj-dim", type=int, default=0, help="Project features to this dim (0 = keep native)")
    args = p.parse_args()

    try:
        process_dataset(args.root, args.model, args.batch, args.proj_dim)
    except KeyboardInterrupt:
        print("[INTERRUPT] Stopped by user — partial embeddings are kept on disk.")


if __name__ == "__main__":
    main()
