"""Task 3.6 - load the registered model BY ALIAS and classify one image.

python scripts/predict_by_alias.py path/to/image.jpg
"""
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serving.inference import load_rgb, resize_like_training  # noqa: E402

load_dotenv()
NAME = os.getenv("REGISTERED_MODEL_NAME", "23F-0752_A2_Ops_VisionGuard")
uri = f"models:/{NAME}@champion"
print(f"Loading {uri} ...")
model = mlflow.pyfunc.load_model(uri)
meta = model.unwrap_python_model().meta
img = resize_like_training(load_rgb(Path(sys.argv[1]).read_bytes()), meta["img_size"])
result = model.predict(np.expand_dims(img, 0))[0]
print(f"model={meta['model_name']} threshold={meta['threshold']:.2f}")
print(f"{sys.argv[1]} -> {result['label']} (P(AI)={result['prob_ai']:.3f}, confidence={result['confidence']:.1%})")
