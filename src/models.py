"""Task 2.2 / 2.3 - baseline CNN and transfer-learning models.

All models take raw RGB pixels in [0, 255] of shape (img_size, img_size, 3) and do their
own preprocessing inside the graph, so training and serving can never disagree.
"""
from tensorflow import keras
from tensorflow.keras import layers

# name prefix of the LAST block of each backbone (used for "unfreeze only the last block")
LAST_BLOCK = {"mobilenetv2": "block_16", "efficientnetb0": "block7a"}


def build_baseline_cnn(img_size: int = 224, dropout: float = 0.3) -> keras.Model:
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale_0_1")(inputs)
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer(backbone: str = "mobilenetv2", img_size: int = 224, dropout: float = 0.3,
                   weights: str | None = "imagenet"):
    """Returns (model, base). The base is frozen."""
    inputs = keras.Input((img_size, img_size, 3), name="image")
    if backbone == "mobilenetv2":
        # identical to keras.applications.mobilenet_v2.preprocess_input: [0,255] -> [-1,1]
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(inputs)
        base = keras.applications.MobileNetV2(include_top=False, weights=weights,
                                              input_shape=(img_size, img_size, 3))
    elif backbone == "efficientnetb0":
        # EfficientNet in Keras contains its own Rescaling+Normalization and expects [0,255]
        x = inputs
        base = keras.applications.EfficientNetB0(include_top=False, weights=weights,
                                                 input_shape=(img_size, img_size, 3))
    else:
        raise ValueError(backbone)
    base.trainable = False
    x = base(x, training=False)  # BatchNorm stays in inference mode even after unfreezing
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return keras.Model(inputs, outputs, name=f"{backbone}_transfer"), base


def unfreeze(base: keras.Model, backbone: str, n_layers: int | None = None) -> list[str]:
    """Unfreeze the last block (n_layers=None) or the last n_layers layers. BatchNorm stays frozen."""
    base.trainable = True
    if n_layers is None:
        prefix = LAST_BLOCK[backbone]
        start = next(i for i, lyr in enumerate(base.layers) if lyr.name.startswith(prefix))
    else:
        start = max(0, len(base.layers) - n_layers)
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
    return [lyr.name for lyr in base.layers if lyr.trainable]


def compile_model(model: keras.Model, lr: float):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")],
    )
    return model


def layer_table(model: keras.Model) -> list[dict]:
    """Output shape + parameter count per layer (for the report table)."""
    rows = []
    for lyr in model.layers:
        try:
            shape = str(lyr.output.shape)
        except Exception:  # noqa: BLE001
            shape = "?"
        rows.append({"layer": lyr.name, "type": type(lyr).__name__, "output_shape": shape,
                     "params": lyr.count_params(), "trainable": lyr.trainable})
    return rows
