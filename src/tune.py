"""Task 2.7 (apply) - Keras Tuner random search for VisionGuard.

Searches learning rate, dropout and number of unfrozen backbone layers (>= 8 trials).
python -m src.tune --trials 10 --epochs 6
"""
import argparse
import json
from pathlib import Path

import keras_tuner as kt
import pandas as pd
from tensorflow import keras

from src.data import make_dataset
from src.models import build_transfer, compile_model, unfreeze


def main(a):
    keras.utils.set_random_seed(a.seed)
    train = make_dataset("train", a.img_size, 32, training=True, seed=a.seed)
    val = make_dataset("val", a.img_size, 32)

    def build(hp):
        model, base = build_transfer("mobilenetv2", a.img_size, hp.Choice("dropout", [0.2, 0.3, 0.5]),
                                     weights=None if a.weights == "none" else a.weights)
        n = hp.Choice("unfrozen_layers", [0, 10, 20, 40])
        if n > 0:
            unfreeze(base, "mobilenetv2", n)
        return compile_model(model, hp.Choice("learning_rate", [1e-3, 3e-4, 1e-4, 3e-5]))

    tuner = kt.RandomSearch(build, objective=kt.Objective("val_loss", "min"), max_trials=a.trials,
                            seed=a.seed, directory="tuner_dir", project_name="visionguard", overwrite=True)
    tuner.search(train, validation_data=val, epochs=a.epochs,
                 callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)], verbose=2)

    rows = []
    for t in tuner.oracle.get_best_trials(num_trials=a.trials):
        m = t.metrics
        rows.append({**t.hyperparameters.values, "val_loss": t.score,
                     "val_accuracy": m.get_best_value("val_accuracy") if m.exists("val_accuracy") else None,
                     "trial": t.trial_id})
    df = pd.DataFrame(rows).sort_values("val_loss")
    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/tuner_visionguard.csv", index=False)
    print(df.to_string(index=False))

    # which hyper-parameter mattered most? spread of mean val_loss across its values
    importance = {hp: float(df.groupby(hp)["val_loss"].mean().pipe(lambda s: s.max() - s.min()))
                  for hp in ["learning_rate", "dropout", "unfrozen_layers"]}
    print("\nRange of mean val_loss per hyper-parameter (bigger = mattered more):")
    print(json.dumps(importance, indent=2))
    Path("reports/tuner_importance.json").write_text(json.dumps(importance, indent=2))
    print("\nBest:", tuner.get_best_hyperparameters(1)[0].values)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", default="imagenet")
    main(ap.parse_args())
