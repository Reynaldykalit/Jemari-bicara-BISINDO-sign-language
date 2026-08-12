"""Prepare the BISINDO LSTM classifier asset for Flutter.

Use this after the notebook has produced either:
  - lstm_bisindo.tflite, or
  - lstm_best.h5

Flutter deployment artifact:
  assets/models/bisindo_sign_classifier.tflite
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tflite", type=Path)
    parser.add_argument("--h5", type=Path)
    parser.add_argument("--out-dir", default=Path("assets/models"), type=Path)
    return parser.parse_args()


def convert_h5_to_tflite(h5_path: Path, target_path: Path) -> None:
    import tensorflow as tf

    model = tf.keras.models.load_model(str(h5_path))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        target_path.write_bytes(converter.convert())
    except Exception as error:
        print("Builtin-only conversion failed; retrying with SELECT_TF_OPS.")
        print(f"Original error: {error}")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        target_path.write_bytes(converter.convert())


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / "bisindo_sign_classifier.tflite"

    if args.tflite:
        source = args.tflite.resolve()
        if not source.exists():
            raise FileNotFoundError(f"LSTM TFLite not found: {source}")
        shutil.copy2(source, target_path)
        print(f"Copied LSTM TFLite: {target_path}")
        return

    if args.h5:
        source = args.h5.resolve()
        if not source.exists():
            raise FileNotFoundError(f"LSTM .h5 not found: {source}")
        convert_h5_to_tflite(source, target_path)
        print(f"Converted LSTM TFLite: {target_path}")
        return

    raise ValueError("Pass either --tflite /path/lstm_bisindo.tflite or --h5 /path/lstm_best.h5")


if __name__ == "__main__":
    main()
