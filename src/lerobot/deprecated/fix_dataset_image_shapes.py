#!/usr/bin/env python3
# /home/josh/phddev/lerobot-upstream/src/lerobot/scripts/fix_dataset_image_shapes.py
import argparse
import json
from pathlib import Path
from typing import Tuple


def _is_chw(shape: Tuple[int, int, int]) -> bool:
    if len(shape) != 3:
        return False
    c, h, w = shape
    return c in (1, 3, 4) and h > 1 and w > 1


def fix_one_dataset(root: Path, dry_run: bool = False) -> bool:
    info_path = root / "meta/info.json"
    if not info_path.exists():
        print(f"[skip] No info.json at {info_path}")
        return False

    with open(info_path, "r") as f:
        info = json.load(f)

    features = info.get("features", {})
    changed = False

    for key, ft in features.items():
        dtype = ft.get("dtype")
        if dtype not in ("image", "video"):
            continue

        shape = tuple(ft.get("shape", ()))
        names = ft.get("names")

        # Target convention: HWC with names ["height","width","channel"]
        desired_names = ["height", "width", "channel"]

        fix_shape = False
        fix_names = False

        if shape and _is_chw(shape):
            # Convert CHW to HWC
            c, h, w = shape
            new_shape = (h, w, c)
            ft["shape"] = list(new_shape)
            fix_shape = True

        # Normalize names
        if not isinstance(names, list) or names != desired_names:
            ft["names"] = desired_names
            fix_names = True

        if fix_shape or fix_names:
            print(
                f"[fix] {root.name}: '{key}' -> "
                f"shape: {shape} {'-> ' + str(ft['shape']) if fix_shape else ''} | "
                f"names: {names} {'-> ' + str(ft['names']) if fix_names else ''}"
            )
            changed = True

    if changed and not dry_run:
        # Write back info.json with indent preserved
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

    if not changed:
        print(f"[ok] {root} already uses HWC with correct names")

    return changed


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fix LeRobot datasets whose image feature shapes/names were stored as CHW/['channels','height','width']. "
            "This script updates meta/info.json to HWC with names ['height','width','channel']. Videos are left untouched."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "One or more dataset roots (directories containing meta/, data/, videos/). "
            "You can pass multiple paths separated by space."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print planned changes without writing")
    args = parser.parse_args()

    any_changed = False
    for p in args.paths:
        root = Path(p).resolve()
        if not root.exists():
            print(f"[skip] Path does not exist: {root}")
            continue
        changed = fix_one_dataset(root, dry_run=args.dry_run)
        any_changed = any_changed or changed

    if any_changed:
        print("[done] Some datasets were updated. If you use cached readers, re-open them to pick up new metadata.")
    else:
        print("[done] No changes needed.")


if __name__ == "__main__":
    main()


