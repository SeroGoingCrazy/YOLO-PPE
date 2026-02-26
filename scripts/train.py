from __future__ import annotations

import argparse
import pathlib
import shutil
import yaml
import torch
from ultralytics import YOLO


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO PPE detector")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--name", default="ppe_exp")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    model = YOLO(cfg["model"])
    device = cfg.get("device", "auto")
    if str(device).lower() == "auto":
        # Ultralytics may reject "auto" in some versions; resolve it explicitly.
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Resolved training device: {device}")

    model.train(
        data=cfg["data_yaml"],
        imgsz=cfg["imgsz"],
        epochs=cfg["epochs"],
        batch=cfg["batch"],
        patience=cfg["patience"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        lrf=cfg["lrf"],
        weight_decay=cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        degrees=cfg["degrees"],
        translate=cfg["translate"],
        scale=cfg["scale"],
        fliplr=cfg["fliplr"],
        mosaic=cfg["mosaic"],
        mixup=cfg["mixup"],
        close_mosaic=cfg["close_mosaic"],
        seed=cfg["seed"],
        workers=cfg["workers"],
        device=device,
        project="runs/train",
        name=args.name,
        exist_ok=True,
    )

    save_dir = pathlib.Path(str(getattr(model.trainer, "save_dir", "")))
    best_path = save_dir / "weights" / "best.pt"
    if best_path.exists():
        artifacts = pathlib.Path("artifacts")
        artifacts.mkdir(parents=True, exist_ok=True)
        out = artifacts / "best.pt"
        shutil.copy2(best_path, out)
        print(f"Saved trained weight to: {out.resolve()}")
    else:
        print("Training finished, but best.pt not found.")


if __name__ == "__main__":
    main()

