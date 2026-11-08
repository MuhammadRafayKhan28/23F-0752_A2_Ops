"""Bonus - Grad-CAM heat maps for 4 correct and 4 wrong test predictions.

python -m src.gradcam --run outputs/mobilenetv2_lr0.001_224
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from tensorflow import keras  # noqa: E402

from src.data import decode_and_resize  # noqa: E402


def make_forward(model):
    """Returns f(x) -> (last 4-D feature map, prediction), works for nested backbones and the CNN."""
    base = next((lyr for lyr in model.layers if isinstance(lyr, keras.Model)), None)
    inner = None
    if base is not None:
        last = [lyr for lyr in base.layers if len(lyr.output.shape) == 4][-1]
        inner = keras.Model(base.inputs, [last.output, base.output])

    def forward(x):
        h, conv = x, None
        for layer in model.layers[1:]:
            if layer is base:
                conv, h = inner(h, training=False)
            else:
                h = layer(h, training=False)
                if base is None and len(h.shape) == 4 and not isinstance(layer, keras.layers.Rescaling):
                    conv = h  # ends as the last spatial feature map before global pooling
        return conv, h
    return forward


def gradcam(forward, img):
    x = tf.convert_to_tensor(img[None], tf.float32)
    with tf.GradientTape() as tape:
        conv, pred = forward(x)
        tape.watch(conv)
        score = pred[:, 0]
    grads = tape.gradient(score, conv)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, axis=-1))[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], img.shape[:2])[..., 0].numpy()
    return cam, float(pred[0, 0])


def main(run: Path, out: Path):
    meta = json.loads((run / "meta.json").read_text())
    model = keras.models.load_model(run / "visionguard.keras", compile=False)
    df = pd.read_csv(run / "test_predictions.csv")
    pick = pd.concat([df[df.label == df.pred].sample(4, random_state=0),
                      df[df.label != df.pred].head(4)])
    fwd = make_forward(model)
    fig, axes = plt.subplots(2, len(pick), figsize=(2.6 * len(pick), 5.4))
    for i, r in enumerate(pick.itertuples()):
        img = decode_and_resize(r.path, meta["img_size"]).numpy().astype("float32")
        cam, p = gradcam(fwd, img)
        tag = "correct" if r.label == r.pred else "WRONG"
        axes[0, i].imshow(img.astype("uint8"))
        axes[0, i].set_title(f"{tag}\ntrue={'AI' if r.label else 'Real'} P(AI)={p:.2f}", fontsize=8)
        axes[1, i].imshow(img.astype("uint8"))
        axes[1, i].imshow(cam, cmap="jet", alpha=0.45)
        for ax in axes[:, i]:
            ax.axis("off")
    fig.suptitle("Grad-CAM: where the model looks")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gradcam.png", dpi=110)
    print(f"saved {out / 'gradcam.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/figures"))
    a = ap.parse_args()
    main(a.run, a.out)
