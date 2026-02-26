from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import sys

import torch
import ultralytics


def safe_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""


def main() -> None:
    out_dir = pathlib.Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "ultralytics_version": ultralytics.__version__,
        "nvidia_smi": safe_cmd(["nvidia-smi"]),
    }

    with open(out_dir / "versions.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(f"Saved version info: {(out_dir / 'versions.json').resolve()}")


if __name__ == "__main__":
    main()

