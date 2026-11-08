"""Task 4.1 - inference module.

The Keras model is loaded ONCE when Predictor() is created (at app start-up), never per request:
loading a .keras file takes seconds and hundreds of MB of RAM, which would make every request slow
and could exhaust memory under concurrent load.

Preprocessing mirrors src/data.py exactly: decode to 3-channel RGB (first frame of animations),
bilinear resize to img_size x img_size WITHOUT keeping aspect ratio, round/clip to uint8, feed raw
[0,255] floats. Model-specific scaling lives inside the model graph.
"""
import io
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        im.seek(0)  # first frame, like decode_image(expand_animations=False)
        # NOTE: no EXIF rotation on purpose - tf.io.decode_image ignores EXIF during training too
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def resize_like_training(rgb: np.ndarray, img_size: int) -> np.ndarray:
    import tensorflow as tf
    x = tf.image.resize(rgb, [img_size, img_size], method="bilinear")
    return tf.cast(tf.clip_by_value(tf.round(x), 0, 255), tf.uint8).numpy().astype("float32")


class Predictor:
    def __init__(self, model_path: str, meta_path: str | None = None, threshold: float | None = None):
        import keras
        t0 = time.time()
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path} (did you run export_champion.py?)")
        self.model = keras.models.load_model(model_path, compile=False)
        meta_path = meta_path or str(Path(model_path).with_name("meta.json"))
        self.meta = json.loads(Path(meta_path).read_text()) if Path(meta_path).exists() else {}
        self.img_size = int(self.meta.get("img_size", self.model.input_shape[1]))
        self.threshold = float(threshold if threshold is not None else self.meta.get("threshold", 0.5))
        self.model_name = self.meta.get("registered_model", self.meta.get("model_name", Path(model_path).stem))
        self.model_version = str(self.meta.get("model_version", "local"))
        self.model(np.zeros((1, self.img_size, self.img_size, 3), "float32"), training=False)  # warm-up
        self.load_seconds = time.time() - t0

    def preprocess(self, data: bytes) -> np.ndarray:
        return resize_like_training(load_rgb(data), self.img_size)[None]

    def predict_bytes(self, data: bytes) -> dict:
        t0 = time.time()
        prob = float(self.model(self.preprocess(data), training=False).numpy().reshape(-1)[0])
        is_ai = prob >= self.threshold
        return {
            "label": "AI-generated" if is_ai else "Real",
            "confidence": prob if is_ai else 1.0 - prob,
            "prob_ai": prob,
            "threshold": self.threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "latency_ms": round((time.time() - t0) * 1000, 1),
        }


def predictor_from_env() -> Predictor:
    thr = os.getenv("THRESHOLD", "").strip()
    return Predictor(os.getenv("MODEL_PATH", "models/champion/visionguard.keras"),
                     os.getenv("MODEL_META_PATH") or None,
                     float(thr) if thr else None)


if __name__ == "__main__":  # console demo for the report: python -m serving.inference img1 img2
    import sys
    p = predictor_from_env()
    print(f"Loaded {p.model_name} v{p.model_version} in {p.load_seconds:.1f}s, threshold={p.threshold:.2f}")
    for f in sys.argv[1:]:
        print(f, "->", p.predict_bytes(Path(f).read_bytes()))
