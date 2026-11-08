"""Task 2.1 - tf.data input pipeline.

read split CSV -> (path, label) tensors -> [shuffle] -> read file -> decode (3 channels)
-> resize to img_size x img_size (aspect ratio NOT preserved: images are stretched)
-> uint8 cache -> batch -> [random augmentation, train only] -> float32 -> prefetch

The model receives raw pixels in [0, 255]; each model rescales inside its own graph,
so serving only has to decode + resize exactly the same way.
"""
import pandas as pd
import tensorflow as tf
from tensorflow import keras

AUTOTUNE = tf.data.AUTOTUNE


def load_split(split: str, splits_dir: str = "data/splits") -> pd.DataFrame:
    return pd.read_csv(f"{splits_dir}/{split}.csv")


def decode_and_resize(path, img_size: int):
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)  # GIF -> first frame
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [img_size, img_size], method="bilinear")  # stretch, no crop/pad
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def build_augmenter(seed: int = 42) -> keras.Sequential:
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomRotation(0.05, seed=seed),       # +-18 degrees
        keras.layers.RandomZoom(0.10, seed=seed),
        keras.layers.RandomContrast(0.10, seed=seed),
    ], name="augmentation")


def make_dataset(split: str, img_size: int = 224, batch_size: int = 32, training: bool = False,
                 augment: bool = True, seed: int = 42, splits_dir: str = "data/splits",
                 cache: bool = True) -> tf.data.Dataset:
    df = load_split(split, splits_dir)
    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].astype("float32").values))
    ds = ds.map(lambda p, y: (decode_and_resize(p, img_size), y), num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache()  # decoded uint8 images kept in RAM after the first epoch
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if training and augment:
        aug = build_augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def save_augmented_batch_figure(path: str, img_size: int = 224, n: int = 16, seed: int = 42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = make_dataset("train", img_size, n, training=True, augment=True, seed=seed, cache=False)
    x, y = next(iter(ds))
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(x):
            ax.imshow(tf.cast(tf.clip_by_value(x[i], 0, 255), tf.uint8).numpy())
            ax.set_title("AI" if y[i] == 1 else "Real", fontsize=9)
    fig.suptitle(f"One augmented training batch ({img_size}x{img_size})")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
