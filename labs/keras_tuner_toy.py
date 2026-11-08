"""Task 2.7 (toy) - Hyperband on Fashion-MNIST: learning rate and number of units."""
from pathlib import Path

import keras_tuner as kt
import pandas as pd
from tensorflow import keras

(x, y), (xt, yt) = keras.datasets.fashion_mnist.load_data()
x, xt = (x / 255.0).astype("float32"), (xt / 255.0).astype("float32")
y, yt = y.astype("int32"), yt.astype("int32")


def build(hp):
    m = keras.Sequential([
        keras.Input((28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(hp.Int("units", 32, 512, step=32), activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])
    m.compile(optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return m


tuner = kt.Hyperband(build, objective="val_accuracy", max_epochs=8, factor=3, seed=42,
                     directory="tuner_dir", project_name="fashion_toy", overwrite=True)
tuner.search(x, y, validation_split=0.2, verbose=1)

rows = [{**t.hyperparameters.values, "val_accuracy": t.score}
        for t in tuner.oracle.get_best_trials(num_trials=50)]
df = pd.DataFrame(rows)[["units", "learning_rate", "val_accuracy"]].sort_values("val_accuracy", ascending=False)
Path("reports").mkdir(exist_ok=True)
df.to_csv("reports/tuner_toy.csv", index=False)
print(df.head(10).to_string(index=False))
best = tuner.get_best_hyperparameters(1)[0]
print("Best settings:", best.values)
best_model = tuner.get_best_models(1)[0]
print("Test accuracy of best model:", best_model.evaluate(xt, yt, verbose=0)[1])
