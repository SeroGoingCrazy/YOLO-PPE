from __future__ import annotations

import argparse
import json
import pathlib
import time

import cv2
from ultralytics import YOLO


def benchmark_fps(model: YOLO, source: str, device: str, max_frames: int = 120) -> float:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source for FPS benchmark: {source}")

    n = 0
    start = time.time()
    while n < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        _ = model.predict(frame, device=device, verbose=False)
        n += 1

    elapsed = time.time() - start
    cap.release()
    if n == 0 or elapsed <= 0:
        return 0.0
    return n / elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model and export evaluation report")
    parser.add_argument("--weights", default="artifacts/best.pt")
    parser.add_argument("--data", default="data/ppe_kaggle.yaml")
    parser.add_argument("--name", default="ppe_val")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fps_source", default="outputs/demo_input.mp4")
    args = parser.parse_args()

    report_dir = pathlib.Path("outputs/eval_report")
    report_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split="test",
        device=args.device,
        project="runs/val",
        name=args.name,
        exist_ok=True,
        plots=True,
    )

    fps = 0.0
    fps_source = pathlib.Path(args.fps_source)
    if fps_source.exists():
        fps = benchmark_fps(model, str(fps_source), args.device)

    result = {
        "weights": args.weights,
        "device": args.device,
        "mAP50-95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "fps": fps,
        "plots_dir": str(pathlib.Path("runs/val") / args.name),
    }

    with open(report_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("PR curve and confusion matrix are saved under runs/val/<name>.")


if __name__ == "__main__":
    main()

