# VisionGuard — AI vs Real image classifier (android_A2_Ops)

![CI](https://github.com/<github-username>/android_A2_Ops/actions/workflows/ci.yml/badge.svg)

A Flask web service that tells whether an uploaded image is **AI-generated** or **real**.
TensorFlow/Keras model · MLflow + DVC on DagsHub · PostgreSQL trail · Docker Compose · cloud model API.

| | |
|---|---|
| Author | android |
| GitHub | https://github.com/<github-username>/android_A2_Ops |
| DagsHub | https://dagshub.com/<dagshub-username>/android_A2_Ops |
| Model API | https://<your-api-host> |

```
              ┌──────────────── laptop (docker compose) ─────────────────┐
 browser ───▶ │  web (Flask+gunicorn :5000) ──SQL──▶ db (Postgres, no port)│
              └──────────┬────────────────────────────────────────────────┘
                         │ HTTPS  POST /predict  (X-API-Key)
                         ▼
              cloud: visionguard-api (Flask + champion .keras baked in)
 DagsHub: Git mirror · DVC storage (dataset) · MLflow tracking + model registry (@champion)
```

## Repository map
| Path | Task |
|---|---|
| `scripts/env_check.py` | 1.1 versions + GPU check |
| `scripts/audit_dataset.py`, `scripts/eda.py`, `scripts/make_splits.py` | 1.2, 1.3, split lists |
| `src/data.py` | 2.1 tf.data pipeline |
| `src/models.py` | 2.2 baseline CNN, 2.3 transfer learning |
| `src/train.py` | 2.4–2.5 training/evaluation + 3.3–3.4 MLflow logging |
| `src/error_analysis.py`, `src/gradcam.py` | 2.6, bonus |
| `labs/keras_tuner_toy.py`, `src/tune.py` | 2.7 |
| `labs/mlflow_toy.py` | 3.1 / 3.2 |
| `scripts/run_experiments.py`, `scripts/compare_runs.py` | 3.5 |
| `scripts/register_model.py`, `scripts/predict_by_alias.py`, `scripts/export_champion.py` | 3.6 |
| `serving/` | 4.1 inference module + upload validation |
| `webapp/` | 4.2–4.4 Flask app, 5 database trail, 8 remote mode |
| `webapp/Dockerfile`, `docker-compose.yml` | 6.2, 6.3 |
| `api/` | 7 cloud model API |
| `.github/workflows/ci.yml`, `labs/gha_toy/` | 6.6 |
| `docs/REPORT_GUIDE.md` | every command + what to screenshot + draft explanations |
| `docs/DEPLOYMENT.md` | Task 7 step by step |

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # fill in DagsHub user/token
dvc pull                                                 # dataset from DagsHub storage
python scripts/audit_dataset.py && python scripts/make_splits.py
python scripts/run_experiments.py                        # 8 tracked runs
python scripts/register_model.py --run-id <id> --champion
python scripts/export_champion.py                        # -> models/champion/
docker compose up -d --build                             # http://localhost:5000
```

## Run the app without Docker
```bash
pip install -r requirements-web.txt
export DATABASE_URL=sqlite:///visionguard.db             # PowerShell: $env:DATABASE_URL="..."
flask --app webapp.app run                               # debug is OFF
```

## Tests
`pip install -r requirements-ci.txt && ruff check serving webapp api tests && pytest -q`
