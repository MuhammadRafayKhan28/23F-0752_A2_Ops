"""Task 3.6 - register models and set the 'champion' alias.

python scripts/register_model.py --run-id <second_best_run_id>                # -> version 1
python scripts/register_model.py --run-id <best_run_id> --champion            # -> version 2 @champion
"""
import argparse
import os

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()
NAME = os.getenv("REGISTERED_MODEL_NAME", "23F-0752_A2_Ops_VisionGuard")

ap = argparse.ArgumentParser()
ap.add_argument("--run-id", required=True)
ap.add_argument("--champion", action="store_true")
ap.add_argument("--alias", default="champion")
a = ap.parse_args()

mv = mlflow.register_model(f"runs:/{a.run_id}/model", NAME)
client = MlflowClient()
client.set_model_version_tag(NAME, mv.version, "source_run", a.run_id)
print(f"Registered {NAME} version {mv.version}")
if a.champion:
    client.set_registered_model_alias(NAME, a.alias, mv.version)
    print(f"Alias '{a.alias}' -> version {mv.version}")
print("Aliases:", client.get_registered_model(NAME).aliases)
for v in client.search_model_versions(f"name='{NAME}'"):
    print(f"  v{v.version}  run={v.run_id}")
