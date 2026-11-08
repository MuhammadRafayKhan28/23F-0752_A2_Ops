"""MLflow 'models-from-code' pyfunc wrapper around the Keras model (Task 3.4 / 3.6).

Logged with mlflow.pyfunc.log_model(python_model="src/vg_pyfunc.py", artifacts={...}).
Wrapping the .keras file this way avoids flavour/version incompatibilities between Keras 3
and MLflow, and keeps the decision threshold together with the weights.
Input: float array (N, H, W, 3) of raw [0,255] pixels already resized to meta["img_size"].
"""
import json

import mlflow
import numpy as np


class VisionGuardPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import keras
        self.model = keras.models.load_model(context.artifacts["keras_model"], compile=False)
        with open(context.artifacts["meta"]) as f:
            self.meta = json.load(f)

    def predict(self, context, model_input, params=None):
        x = np.asarray(model_input, dtype="float32")
        if x.ndim == 3:
            x = x[None]
        probs = self.model.predict(x, verbose=0).reshape(-1)
        thr = float(self.meta["threshold"])
        return [{"prob_ai": float(p), "label": "AI-generated" if p >= thr else "Real",
                 "confidence": float(p if p >= thr else 1 - p)} for p in probs]


mlflow.models.set_model(VisionGuardPyfunc())
