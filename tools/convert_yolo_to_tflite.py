"""Export the trained BISINDO YOLOv8 hand detector to TFLite.

Expected training artifact from the notebook:
  06_models/yolov8s_bisindo_v2/weights/best.pt

Flutter deployment artifact:
  assets/models/bisindo_hand_detector.tflite
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-pt", required=True, type=Path)
    parser.add_argument("--out-dir", default=Path("assets/models"), type=Path)
    parser.add_argument("--data-yaml", type=Path)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.5, type=float)
    return parser.parse_args()


def metric_summary(metrics) -> dict[str, float]:
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def print_metrics(title: str, values: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in values.items():
        print(f"{key:10s}: {value * 100:.2f}%")


def main() -> None:
    args = parse_args()
    best_pt = args.best_pt.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not best_pt.exists():
        raise FileNotFoundError(f"YOLO .pt not found: {best_pt}")

    model = YOLO(str(best_pt))

    pt_metrics = None
    if args.data_yaml:
        pt_metrics = model.val(
            data=str(args.data_yaml),
            split="test",
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )
        print_metrics("best.pt metrics", metric_summary(pt_metrics))

    export_path = Path(model.export(format="tflite", imgsz=args.imgsz, half=True))
    target_path = out_dir / "bisindo_hand_detector.tflite"
    shutil.copy2(export_path, target_path)
    print(f"\nExported TFLite (Float16): {target_path}")

    if args.data_yaml:
        tflite_model = YOLO(str(target_path))
        tflite_metrics = tflite_model.val(
            data=str(args.data_yaml),
            split="test",
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
        )
        print_metrics("bisindo_hand_detector.tflite metrics", metric_summary(tflite_metrics))


if __name__ == "__main__":
    main()
