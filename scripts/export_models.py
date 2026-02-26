from __future__ import annotations

import argparse
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX/TensorRT")
    parser.add_argument("--weights", default="artifacts/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--onnx", action="store_true")
    parser.add_argument("--engine", action="store_true")
    args = parser.parse_args()

    model = YOLO(args.weights)
    if args.onnx:
        onnx_path = model.export(format="onnx", imgsz=args.imgsz, device=args.device)
        print(f"ONNX exported: {onnx_path}")
    if args.engine:
        engine_path = model.export(format="engine", imgsz=args.imgsz, device=args.device)
        print(f"TensorRT engine exported: {engine_path}")


if __name__ == "__main__":
    main()

