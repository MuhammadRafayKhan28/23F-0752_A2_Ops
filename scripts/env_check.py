"""Task 1.1 - record exact library versions and whether TensorFlow sees a GPU."""
import getpass
import platform
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PKGS = ["tensorflow", "tensorflow-cpu", "keras", "keras-tuner", "mlflow", "dagshub", "dvc",
        "numpy", "pandas", "pillow", "scikit-learn", "matplotlib", "flask"]

lines = [
    f"Run by: {getpass.getuser()} at {datetime.now():%Y-%m-%d %H:%M:%S}",
    f"Python: {sys.version.split()[0]}  ({sys.executable})",
    f"OS: {platform.platform()}",
]
for p in PKGS:
    try:
        lines.append(f"{p:15s} {version(p)}")
    except PackageNotFoundError:
        lines.append(f"{p:15s} (not installed)")

import tensorflow as tf  # noqa: E402

gpus = tf.config.list_physical_devices("GPU")
lines.append(f"TF built with CUDA: {tf.test.is_built_with_cuda()}")
lines.append(f"GPUs visible to TF: {len(gpus)} {[g.name for g in gpus]}")

text = "\n".join(lines)
print(text)
Path("reports").mkdir(exist_ok=True)
Path("reports/environment.txt").write_text(text + "\n")
