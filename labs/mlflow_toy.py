"""Task 3.1 / 3.2 - tiny Keras model on MNIST tracked with MLflow.

Local :  python labs/mlflow_toy.py --lr 0.001  &&  python labs/mlflow_toy.py --lr 0.01
         mlflow ui      (open http://127.0.0.1:5000, select both runs -> Compare)
DagsHub: python labs/mlflow_toy.py --lr 0.001 --remote   (reads MLFLOW_* from .env)
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from tensorflow import keras  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--remote", action="store_true")
a = ap.parse_args()

if a.remote:
    load_dotenv()                         # token comes from .env, never from code
    assert os.getenv("MLFLOW_TRACKING_URI"), "set MLFLOW_TRACKING_URI in .env"
else:
    mlflow.set_tracking_uri("file:./mlruns")

(x, y), (xt, yt) = keras.datasets.mnist.load_data()
x, xt = (x[:20000] / 255.0).astype("float32"), (xt / 255.0).astype("float32")
y, yt = y[:20000].astype("int32"), yt.astype("int32")

mlflow.set_experiment("23L-0548_A2_Ops_toy")
with mlflow.start_run(run_name=f"toy_lr{a.lr}"):
    mlflow.log_params({"lr": a.lr, "epochs": a.epochs, "units": 64, "dataset": "mnist[:20000]"})
    mlflow.set_tag("purpose", "tool-lab")
    model = keras.Sequential([keras.Input((28, 28)), keras.layers.Flatten(),
                              keras.layers.Dense(64, activation="relu"), keras.layers.Dense(10, activation="softmax")])
    model.compile(optimizer=keras.optimizers.Adam(a.lr),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

    class Log(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch)

    h = model.fit(x, y, validation_split=0.2, epochs=a.epochs, callbacks=[Log()], verbose=2)
    mlflow.log_metric("test_accuracy", model.evaluate(xt, yt, verbose=0)[1])

    fig, ax = plt.subplots()
    ax.plot(h.history["loss"], label="train")
    ax.plot(h.history["val_loss"], label="val")
    ax.set_title(f"loss (lr={a.lr})")
    ax.legend()
    fig.savefig("toy_loss.png")
    mlflow.log_artifact("toy_loss.png", "plots")
    model.save("toy_model.keras")
    mlflow.log_artifact("toy_model.keras", "model")
    os.remove("toy_loss.png")
    os.remove("toy_model.keras")
print("done - run `mlflow ui` (local) or open the DagsHub Experiments tab")
