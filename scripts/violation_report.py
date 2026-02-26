from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


VIOLATION_KEYWORDS = {"NO-Hardhat", "NO-Safety Vest"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PPE violation CSV and alert screenshots")
    parser.add_argument("--weights", default="artifacts/best.pt")
    parser.add_argument("--source", default="outputs/demo_input.mp4")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    report_dir = Path("outputs/business")
    screenshot_dir = report_dir / "alerts"
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "violation_stats.csv"

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open source: {source_path}")

    rows = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(frame, conf=args.conf, device=args.device, verbose=False)[0]
        names = result.names

        violation_classes = []
        for cls_id in result.boxes.cls.tolist() if result.boxes is not None else []:
            cls_name = names[int(cls_id)]
            if cls_name in VIOLATION_KEYWORDS:
                violation_classes.append(cls_name)

        if violation_classes:
            timestamp = datetime.now().isoformat(timespec="seconds")
            alert_img = screenshot_dir / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(alert_img), result.plot())
            rows.append(
                {
                    "timestamp": timestamp,
                    "frame_id": frame_id,
                    "violations": "|".join(sorted(set(violation_classes))),
                    "count": len(violation_classes),
                    "alert_image": str(alert_img),
                }
            )

        frame_id += 1

    cap.release()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "frame_id", "violations", "count", "alert_image"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved alert screenshots in: {screenshot_dir}")
    print(f"Total violation events: {len(rows)}")


if __name__ == "__main__":
    main()

