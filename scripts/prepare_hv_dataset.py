from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# Source class ids in css-data:
# 0 Hardhat, 2 NO-Hardhat, 4 NO-Safety Vest, 5 Person, 7 Safety Vest
CLASS_MAP = {
    0: 0,  # Hardhat
    2: 1,  # NO-Hardhat
    7: 2,  # Safety Vest
    4: 3,  # NO-Safety Vest
    5: 4,  # Person
}


def remap_label_file(src_label: Path, dst_label: Path) -> bool:
    if not src_label.exists():
        return False

    kept = []
    with src_label.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            src_cls = int(parts[0])
            if src_cls not in CLASS_MAP:
                continue
            parts[0] = str(CLASS_MAP[src_cls])
            kept.append(" ".join(parts))

    if not kept:
        return False

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    with dst_label.open("w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")
    return True


def process_split(src_root: Path, dst_root: Path, split: str) -> tuple[int, int]:
    src_img_dir = src_root / split / "images"
    src_lbl_dir = src_root / split / "labels"
    dst_img_dir = dst_root / split / "images"
    dst_lbl_dir = dst_root / split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    for img_path in sorted(src_img_dir.glob("*.jpg")):
        total += 1
        label_path = src_lbl_dir / f"{img_path.stem}.txt"
        out_label = dst_lbl_dir / f"{img_path.stem}.txt"
        out_img = dst_img_dir / img_path.name
        has_valid = remap_label_file(label_path, out_label)
        if has_valid:
            shutil.copy2(img_path, out_img)
            kept += 1

    return total, kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare focused Helmet/Vest dataset (5 classes)")
    parser.add_argument("--src", default="datasets/raw/css-data")
    parser.add_argument("--dst", default="datasets/ppe-hv")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")

    if dst_root.exists():
        shutil.rmtree(dst_root)

    # css-data uses train/valid/test split naming
    for split in ["train", "valid", "test"]:
        total, kept = process_split(src_root, dst_root, split)
        print(f"{split}: kept {kept}/{total} images with target classes")

    print(f"Prepared dataset at: {dst_root.resolve()}")


if __name__ == "__main__":
    main()

