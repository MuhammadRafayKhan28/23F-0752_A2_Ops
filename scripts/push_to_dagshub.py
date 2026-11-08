"""Sync local MLflow runs to DagsHub remote tracking server.

Usage:
    python scripts/push_to_dagshub.py --token <YOUR_DAGSHUB_TOKEN>
    # or if DAGSHUB_USER_TOKEN is set in .env:
    python scripts/push_to_dagshub.py
"""
import argparse
import os
import shutil
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER", "Android1289")
REPO_NAME = os.getenv("DAGSHUB_REPO_NAME", "23L-0548_A2_Ops")
REMOTE_URI = f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow"


def main():
    parser = argparse.ArgumentParser(description="Push local MLflow runs to DagsHub")
    parser.add_argument("--token", help="DagsHub access token (from https://dagshub.com/user/settings/tokens)")
    parser.add_argument("--username", default=REPO_OWNER, help="DagsHub username")
    args = parser.parse_args()

    token = args.token or os.getenv("DAGSHUB_USER_TOKEN") or os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not token:
        print("[!] No token provided.")
        print(f"    Please obtain a token at: https://dagshub.com/user/settings/tokens")
        print(f"    Then run: python scripts/push_to_dagshub.py --token <YOUR_TOKEN>")
        return

    # Set up remote tracking environment
    os.environ["MLFLOW_TRACKING_URI"] = REMOTE_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = args.username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    import mlflow
    from mlflow.entities import Metric, Param, RunTag
    from mlflow.tracking import MlflowClient

    local_client = MlflowClient(tracking_uri="file:./mlruns")
    remote_client = MlflowClient(tracking_uri=REMOTE_URI)

    print(f"Targeting DagsHub MLflow: {REMOTE_URI}")

    experiments = [
        ("23L-0548_A2_Ops_VisionGuard", ["23L-0548_A2_Ops_VisionGuard", "android_A2_Ops_VisionGuard"]),
        ("23L-0548_A2_Ops_toy", ["23L-0548_A2_Ops_toy", "android_A2_Ops_toy"]),
    ]

    for target_name, local_candidates in experiments:
        local_exp = None
        for cand in local_candidates:
            local_exp = local_client.get_experiment_by_name(cand)
            if local_exp:
                break

        if not local_exp:
            print(f"[-] Local experiment for {target_name} not found, skipping.")
            continue

        # Get or create remote experiment
        remote_exp = remote_client.get_experiment_by_name(target_name)
        if not remote_exp:
            print(f"[+] Creating remote experiment '{target_name}' on DagsHub...")
            remote_exp_id = remote_client.create_experiment(target_name)
        else:
            remote_exp_id = remote_exp.experiment_id
            print(f"[=] Remote experiment '{target_name}' exists (id={remote_exp_id})")

        local_runs = local_client.search_runs(local_exp.experiment_id)
        print(f"Found {len(local_runs)} local runs to sync for {target_name}.")

        for lrun in reversed(local_runs):
            run_name = lrun.data.tags.get("mlflow.runName", lrun.info.run_id[:8])
            print(f"  --> Syncing run: {run_name} ...")

            # Create remote run
            tags = {k: v for k, v in lrun.data.tags.items() if not k.startswith("mlflow.runName")}
            tags["mlflow.runName"] = run_name
            rrun = remote_client.create_run(
                experiment_id=remote_exp_id,
                start_time=lrun.info.start_time,
                tags=tags,
            )

            # Log parameters
            params = [Param(k, str(v)) for k, v in lrun.data.params.items()]
            for chunk in [params[i:i+100] for i in range(0, len(params), 100)]:
                remote_client.log_batch(rrun.info.run_id, params=chunk)

            # Log metrics (with histories)
            metrics = []
            for key in lrun.data.metrics.keys():
                history = local_client.get_metric_history(lrun.info.run_id, key)
                metrics.extend([Metric(m.key, m.value, m.timestamp, m.step) for m in history])

            for chunk in [metrics[i:i+100] for i in range(0, len(metrics), 100)]:
                remote_client.log_batch(rrun.info.run_id, metrics=chunk)

            # Log artifacts
            with tempfile.TemporaryDirectory() as tmp_dir:
                try:
                    local_art_dir = local_client.download_artifacts(lrun.info.run_id, "", tmp_dir)
                    if Path(local_art_dir).exists() and any(Path(local_art_dir).iterdir()):
                        remote_client.log_artifacts(rrun.info.run_id, local_art_dir)
                except Exception as e:
                    print(f"      [warn] Artifact sync for {run_name}: {e}")

            remote_client.set_terminated(rrun.info.run_id, status="FINISHED", end_time=lrun.info.end_time)
            print(f"      [done] Synced run {run_name} -> remote {rrun.info.run_id}")

    # Register model on remote if champion exists
    reg_name = "23L-0548_A2_Ops_VisionGuard"
    print(f"\n[+] Syncing model registry on DagsHub...")
    try:
        # Search for mnv2_noaug run on remote
        remote_exp = remote_client.get_experiment_by_name("23L-0548_A2_Ops_VisionGuard")
        runs = remote_client.search_runs(remote_exp.experiment_id, "tags.mlflow.runName = 'mnv2_noaug'")
        if runs:
            champion_run_id = runs[0].info.run_id
            print(f"Found remote champion run: {champion_run_id}")
            mv = remote_client.create_model_version(
                name=reg_name,
                source=f"runs:/{champion_run_id}/model",
                run_id=champion_run_id,
                description="VisionGuard MobileNetV2 fine-tuned Champion model",
            )
            remote_client.set_registered_model_alias(reg_name, "champion", mv.version)
            print(f"Successfully registered {reg_name} v{mv.version} @champion on DagsHub!")
    except Exception as e:
        print(f"Registry note: {e}")

    print("\n[SUCCESS] All runs and artifacts synced to DagsHub!")
    print(f"View live at: https://dagshub.com/{REPO_OWNER}/{REPO_NAME}")
    print(f"MLflow UI:    https://dagshub.com/{REPO_OWNER}/{REPO_NAME}/experiments")


if __name__ == "__main__":
    main()
