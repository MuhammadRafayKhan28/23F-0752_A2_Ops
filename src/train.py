"""Train + evaluate one VisionGuard model and track everything in MLflow (Tasks 2 and 3).

Examples
  python -m src.train --model cnn --run-name cnn_baseline
  python -m src.train --model mobilenetv2 --run-name mnv2_lr1e-3
  python -m src.train --model mobilenetv2 --autolog --run-name mnv2_autolog
  python -m src.train --model mobilenetv2 --finetune none --run-name mnv2_frozen_only
"""
import argparse
import json
import os
import platform
import re
import subprocess
import time
from importlib.metadata import distributions
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from tensorflow import keras

from src.data import make_dataset, save_augmented_batch_figure
from src.evaluation import choose_threshold, compute_metrics, plot_confusion, plot_history, plot_roc
from src.models import build_baseline_cnn, build_transfer, compile_model, layer_table, unfreeze


# ---------------------------------------------------------------- helpers
def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def dvc_data_version(dvc_file="data/raw.dvc") -> str:
    """md5 that DVC stored in the pointer file = exact version of the dataset used."""
    try:
        m = re.search(r"md5:\s*([0-9a-f]+)", Path(dvc_file).read_text())
        return m.group(1) if m else "unknown"
    except FileNotFoundError:
        return "not-tracked"


class MlflowEpochLogger(keras.callbacks.Callback):
    """Manual per-epoch logging (the 'manual' approach of Task 3.3)."""

    def __init__(self, step_offset=0):
        super().__init__()
        self.step_offset = step_offset

    def on_epoch_end(self, epoch, logs=None):
        logs = dict(logs or {})
        lr = self.model.optimizer.learning_rate
        logs["learning_rate"] = float(lr.numpy() if hasattr(lr, "numpy") else lr)
        mlflow.log_metrics({k: float(v) for k, v in logs.items()}, step=epoch + self.step_offset)


def callbacks(ckpt_path, patience, initial_best=None):
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True,
                                        initial_value_threshold=initial_best),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(1, patience // 2),
                                          min_lr=1e-7),
    ]


def predict_probs(model, split, a):
    ds = make_dataset(split, a.img_size, a.batch_size, training=False, cache=False, splits_dir=a.splits_dir)
    return model.predict(ds, verbose=0).reshape(-1)


# ---------------------------------------------------------------- main
def main(a):
    load_dotenv()
    keras.utils.set_random_seed(a.seed)  # python, numpy and TF seeds
    if a.deterministic:
        tf.config.experimental.enable_op_determinism()

    out = Path("outputs") / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    ckpt = str(out / "best.keras")

    train_ds = make_dataset("train", a.img_size, a.batch_size, training=True, augment=not a.no_aug,
                            seed=a.seed, splits_dir=a.splits_dir)
    val_ds = make_dataset("val", a.img_size, a.batch_size, splits_dir=a.splits_dir)

    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=a.run_name) as run:
        mlflow.set_tags({
            "git_commit": git_commit(), "dvc_data_version": dvc_data_version(), "seed": a.seed,
            "model_family": a.model, "logging_mode": "autolog" if a.autolog else "manual",
            "tf_version": tf.__version__, "python": platform.python_version(), "author": "android",
        })
        mlflow.log_params({k: v for k, v in vars(a).items() if k not in ("experiment",)})
        if a.autolog:
            mlflow.tensorflow.autolog(log_models=False, log_datasets=False)

        # -------------------- build
        base = None
        if a.model == "cnn":
            model = build_baseline_cnn(a.img_size, a.dropout)
        else:
            model, base = build_transfer(a.model, a.img_size, a.dropout,
                                         weights=None if a.weights == "none" else a.weights)
        compile_model(model, a.lr)
        pd.DataFrame(layer_table(model)).to_csv(out / "layer_table.csv", index=False)
        with open(out / "model_summary.txt", "w", encoding="utf-8") as f:
            model.summary(print_fn=lambda s, **k: f.write(s + "\n"))
        model.summary()

        # -------------------- stage 1
        cbs = callbacks(ckpt, a.patience)
        if not a.autolog:
            cbs.append(MlflowEpochLogger())
        t0 = time.time()
        h1 = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=cbs, verbose=2)
        hist = {k: list(v) for k, v in h1.history.items()}
        stage1_epochs = len(hist["loss"])
        trainable = {"stage1_frozen_base": [w.path for w in model.trainable_weights]}

        # -------------------- stage 2: fine-tune last block with a much smaller LR
        if base is not None and a.finetune != "none":
            n = None if a.finetune == "block" else int(a.finetune)
            names = unfreeze(base, a.model, n)
            trainable["stage2_unfrozen_base_layers"] = names
            print(f"Unfroze {len(names)} base layers: {names}")
            compile_model(model, a.ft_lr)
            best1 = float(np.min(hist["val_loss"]))
            cbs2 = callbacks(ckpt, a.patience, initial_best=best1)
            if a.autolog:
                with mlflow.start_run(run_name=f"{a.run_name}_finetune", nested=True):
                    h2 = model.fit(train_ds, validation_data=val_ds, epochs=a.ft_epochs,
                                   callbacks=cbs2, verbose=2)
            else:
                cbs2.append(MlflowEpochLogger(step_offset=stage1_epochs))
                h2 = model.fit(train_ds, validation_data=val_ds, epochs=a.ft_epochs,
                               callbacks=cbs2, verbose=2)
            for k, v in h2.history.items():
                hist.setdefault(k, []).extend(v)
        train_secs = time.time() - t0
        if a.autolog:
            mlflow.tensorflow.autolog(disable=True)
        (out / "trainable_layers.json").write_text(json.dumps(trainable, indent=2))

        best_epoch = int(np.argmin(hist["val_loss"])) + 1
        plot_history(hist, out / "training_curves.png", best_epoch,
                     stage1_epochs if len(hist["loss"]) > stage1_epochs else None, title=a.run_name)
        pd.DataFrame(hist).to_csv(out / "history.csv", index_label="epoch0")

        # -------------------- reload best checkpoint, pick threshold on VALIDATION
        model = keras.models.load_model(ckpt)
        val_df = pd.read_csv(f"{a.splits_dir}/val.csv")
        val_prob = predict_probs(model, "val", a)
        thr, fb = (choose_threshold(val_df.label.values, val_prob) if a.threshold is None
                   else (a.threshold, float("nan")))
        val_m = compute_metrics(val_df.label.values, val_prob, thr)

        # -------------------- evaluate on TEST exactly once
        test_df = pd.read_csv(f"{a.splits_dir}/test.csv")
        test_prob = predict_probs(model, "test", a)
        test_m = compute_metrics(test_df.label.values, test_prob, thr)
        test_df.assign(prob_ai=test_prob, pred=(test_prob >= thr).astype(int)).to_csv(
            out / "test_predictions.csv", index=False)
        plot_confusion(test_df.label.values, test_prob, thr, out / "confusion_matrix.png")
        plot_roc(test_df.label.values, test_prob, out / "roc_curve.png")
        save_augmented_batch_figure(str(out / "augmented_batch.png"), a.img_size, seed=a.seed)

        mlflow.log_metrics({"best_epoch": best_epoch, "val_fbeta05_at_thr": fb, "decision_threshold": thr,
                            "train_seconds": train_secs, "params_total": model.count_params()})
        mlflow.log_metrics({f"val_{k}_at_thr": v for k, v in val_m.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})

        # -------------------- artifacts (Task 3.4)
        meta = {"model_name": a.run_name, "model_family": a.model, "img_size": a.img_size,
                "threshold": thr, "labels": {"0": "Real", "1": "AI-generated"},
                "mlflow_run_id": run.info.run_id, "git_commit": git_commit()}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        frozen = "\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions()))
        (out / "requirements_frozen.txt").write_text(frozen)
        for f in ["training_curves.png", "confusion_matrix.png", "roc_curve.png", "augmented_batch.png"]:
            mlflow.log_artifact(str(out / f), "plots")
        for f in ["layer_table.csv", "model_summary.txt", "trainable_layers.json", "history.csv",
                  "test_predictions.csv", "requirements_frozen.txt", "meta.json"]:
            mlflow.log_artifact(str(out / f), "info")
        for s in ["train", "val", "test"]:
            mlflow.log_artifact(f"{a.splits_dir}/{s}.csv", "splits")
        keras_path = out / "visionguard.keras"
        model.save(keras_path)
        mlflow.log_artifact(str(keras_path), "keras_model")
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=str(Path(__file__).with_name("vg_pyfunc.py")),
            artifacts={"keras_model": str(keras_path), "meta": str(out / "meta.json")},
            pip_requirements=[f"tensorflow-cpu=={tf.__version__}", "numpy", "pillow"],
        )
        print(json.dumps({"run_id": run.info.run_id, "best_epoch": best_epoch, "threshold": thr,
                          **{f"test_{k}": round(v, 4) for k, v in test_m.items()}}, indent=2))


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "mobilenetv2", "efficientnetb0"], default="mobilenetv2")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--experiment", default=os.getenv("MLFLOW_EXPERIMENT", "23L-0548_A2_Ops_VisionGuard"))
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--ft-epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--finetune", default="block", help="'block' (last block), 'none', or an int N layers")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=None, help="fixed threshold (default: tune on val)")
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none' (offline smoke tests)")
    ap.add_argument("--splits-dir", default="data/splits")
    ap.add_argument("--autolog", action="store_true", help="use mlflow.tensorflow.autolog instead of manual")
    ap.add_argument("--deterministic", action="store_true")
    a = ap.parse_args(argv)
    a.run_name = a.run_name or f"{a.model}_lr{a.lr}_{a.img_size}"
    return a


if __name__ == "__main__":
    main(parse_args())
