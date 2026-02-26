from __future__ import annotations

import argparse
import time

import cv2
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time PPE detection with camera")
    parser.add_argument("--weights", default="artifacts/best.pt")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index: {args.cam}")

    prev = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(frame, conf=args.conf, device=args.device, verbose=False)[0]
        out = result.plot()

        now = time.time()
        fps = 1.0 / max(now - prev, 1e-6)
        prev = now
        cv2.putText(out, f"FPS: {fps:.2f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("PPE Camera Demo (press q to quit)", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

