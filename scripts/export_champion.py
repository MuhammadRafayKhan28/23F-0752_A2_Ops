"""Download the @champion model ONCE from the DagsHub registry into models/champion/.

The serving app never calls DagsHub per request: this file is mounted into the web container
(docker compose) and baked into the cloud API image. Run it whenever the champion changes.
"""
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
from dotenv import load_dotenv

load_dotenv()
NAME = os.getenv("REGISTERED_MODEL_NAME", "android_A2_Ops_VisionGuard")
dst = Path("models/champion")
dst.mkdir(parents=True, exist_ok=True)
version = mlflow.MlflowClient().get_model_version_by_alias(NAME, "champion")
with tempfile.TemporaryDirectory() as tmp:
    local = Path(mlflow.artifacts.download_artifacts(f"models:/{NAME}@champion", dst_path=tmp))
    keras_file = next(local.rglob("*.keras"))
    meta_file = next(local.rglob("meta.json"))
    shutil.copy(keras_file, dst / "visionguard.keras")
    import json
    meta = json.loads(meta_file.read_text())
    meta.update(registered_model=NAME, model_version=str(version.version))
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
print(f"Exported {NAME} v{version.version} -> {dst}/visionguard.keras + meta.json")
