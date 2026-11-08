# VisionGuard — Run book & report guide (android)

For every sub-task: **▶ what to run**, **📸 what to screenshot**, **✍ draft text** for the report.
The draft text explains the design choices made in this code — rewrite it in your own words and replace
every `<…>` with your real numbers after running. Screenshots must show your terminal prompt/username,
date-time and the project name `23L-0548_A2_Ops` (rule 11.1).

**Assumptions made** (state these in the report):
- Project name `23L-0548_A2_Ops` on GitHub, DagsHub, MLflow experiment, registry
  model (`23L-0548_A2_Ops_VisionGuard`). Docker/Azure require lowercase → `23l-0548_a2_ops`.
- Positive class = **AI-generated (label 1)**, Real = 0. Folder `AiArtData` → 1, `RealArt` → 0.
- Split 70/15/15 stratified, seed 42, after removing corrupt files and exact duplicates.
- Image size 224×224, images stretched (no crop/pad). Python 3.11, TensorFlow 2.16/2.17 (Keras 3), CPU.
- Database: PostgreSQL 16 in Compose (SQLite fallback for local runs/tests).
- Task 7 pathway C: model API on Azure Container Apps; Flask UI + DB on localhost (as Task 7 asks).

## Front matter (put at the top of the PDF)
Name: android · Project: 23L-0548_A2_Ops · GitHub / DagsHub / API URL ·
graded commit: `git rev-parse HEAD` · **AI-assistance disclosure** — say honestly which tools you used and
for what (e.g. "AI assistant was used to scaffold the code and draft explanations; I ran, debugged and verified
every step and wrote the final text").

---
## Chapter 1 — Data understanding & versioning

### 1.1 Environment (1)
▶
```bash
python3.11 -m venv .venv && source .venv/bin/activate     # Windows: py -3.11 -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
python scripts/env_check.py
```
📸 terminal output (versions + "GPUs visible to TF: 0/1").
✍ An isolated venv pins exact versions so the project does not break other projects and can be recreated
(`pip freeze` is also logged to every MLflow run). On native Windows TensorFlow ≥2.11 has no GPU support, so
CPU is expected; MobileNetV2 fine-tuning on ~1k images is feasible on CPU (minutes per epoch).

### Dataset download
```bash
# kaggle.json from kaggle.com → Settings → API → Create token, put in ~/.kaggle/
kaggle datasets download -d cashbowman/ai-generated-images-vs-real-images -p data --unzip
# make sure the layout is data/raw/AiArtData/... and data/raw/RealArt/... (move folders if nested twice)
```

### 1.2 Dataset audit (3)
▶ `python scripts/audit_dataset.py` → `reports/audit.csv`, `reports/audit_summary.md`
📸 the summary tables (counts per class; min/median/max of width, height, size; corrupt; duplicates).
✍ Every file is opened three times: header parse (size/mode/format), `verify()` (structure) and `load()`
(full decode — catches truncated downloads that `verify()` misses). The SHA-256 of the bytes identifies
exact duplicates; a duplicate inside the same class only inflates that class, but a duplicate that appears
in both train and test leaks the answer, and one appearing in both classes is a label conflict. Results:
`<n AI>` AI and `<n real>` real images, `<k>` unreadable, `<d>` duplicates → removed before splitting.

### 1.3 Visual EDA (1.5)
▶ `python scripts/eda.py` → `reports/figures/sample_grid.png`, `hist_width.png`, `hist_height.png`, `hist_file_size_kb.png`
✍ (adapt to what you see) *Sample grid*: AI images are mostly portraits/landscapes with smooth skin, glossy
lighting and saturated colours; real images are more varied, with noise and cluttered backgrounds.
*Width/height*: AI images cluster at generator sizes (512/768/1024 squares) while real photos spread widely
— a **shortcut risk**: a model could learn "square = AI". Resizing everything to 224×224 removes most of
this signal. *File size*: real photos are larger/heavier-tailed (log scale); compression level could also
be a shortcut — we never feed file size to the model.

### 1.4 DVC (1.5)
▶ **Toy part** (separate DagsHub repo `23L-0548_A2_Ops-dvc-toy`, connected to an empty GitHub repo):
```bash
mkdir dvc-toy && cd dvc-toy
git init && dvc init && git commit -m "init dvc"
printf "id,label\n1,real\n2,ai\n" > data.csv
dvc add data.csv                     # creates data.csv.dvc + adds data.csv to .gitignore
git add data.csv.dvc .gitignore && git commit -m "data v1"
printf "3,ai\n4,real\n" >> data.csv
dvc add data.csv && git commit -am "data v2"
git log --oneline
git checkout HEAD~1 -- data.csv.dvc && dvc checkout && cat data.csv     # back to v1 (2 rows)
git checkout HEAD -- data.csv.dvc && dvc checkout                      # forward to v2 again
# remote = DagsHub storage (S3-compatible)
dvc remote add -d origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<user>/23L-0548_A2_Ops-dvc-toy.s3
dvc remote modify origin --local access_key_id <token>
dvc remote modify origin --local secret_access_key <token>
git add .dvc/config && git commit -m "dvc remote"
git remote add origin https://dagshub.com/<user>/23L-0548_A2_Ops-dvc-toy.git && git push -u origin main
dvc push
rm data.csv && rm -rf .dvc/cache     # Windows: del data.csv & rmdir /s /q .dvc\cache
dvc pull && cat data.csv
```
**Apply** (main repo): `dvc add data/raw` → `git add data/raw.dvc data/.gitignore` → commit → `dvc push`.
The `--local` flag writes the token into `.dvc/config.local`, which is git-ignored.
📸 `git log`, the `cat` after checkout, `dvc push`, `dvc pull`, the DagsHub repo page showing DVC files.
✍ The `.dvc` pointer file is a small YAML with the **md5 hash, size and file count** of the tracked data and
its path. Git versions this pointer; the real bytes live in the DVC cache and remote, addressed by that hash.
Large binary data does not belong in Git because every version is copied into history forever (repo bloats,
clones get slow), Git can't diff binaries, and GitHub rejects files >100 MB. The DVC md5 is also logged as
the `dvc_data_version` tag on every MLflow run → each model points to the exact data version.

---
## Chapter 2 — Model development

▶ Splits: `python scripts/make_splits.py` → `data/splits/{train,val,test}.csv` (commit these).

### 2.1 Input pipeline (2)
✍ `src/data.py`: (1) `from_tensor_slices` over (path, label) from the split CSV; (2) `read_file` +
`decode_image(channels=3, expand_animations=False)` → every image becomes RGB, GIFs give their first frame,
greyscale is replicated to 3 channels; (3) `resize` to **224×224** bilinear — 224 is the native ImageNet
size of MobileNetV2/EfficientNetB0 (pretrained filters work best at their training scale) and keeps CPU
training feasible; the **aspect ratio is not preserved** — images are stretched/squeezed to a square.
We accepted that because cropping could cut off the evidence (faces, hands, textures) and padding adds
black bars that correlate with source; (4) `cache()` of uint8 tensors in RAM so decoding happens once;
(5) `shuffle` (train only, reshuffled each epoch); (6) `batch(32)`; (7) **augmentation only for train**
— horizontal flip, ±5% rotation, ±10% zoom, ±10% contrast — mild on purpose: strong blur/noise/JPEG
augmentation would destroy the very artefacts that distinguish AI images; validation/test must be the
untouched distribution so metrics are honest; (8) `prefetch(AUTOTUNE)` overlaps input prep with training.
📸 `outputs/<run>/augmented_batch.png`.

### 2.2 Baseline CNN (3)
▶ `python scripts/run_experiments.py R1` → `outputs/cnn_baseline/model_summary.txt`, `layer_table.csv`
✍ Layers: `Rescaling(1/255)` puts pixels in [0,1] inside the model (serving can't forget it);
4 × [Conv3×3 → BatchNorm → ReLU → MaxPool] with 32→64→128→256 filters: convolutions learn local texture
patterns, doubling filters as resolution halves keeps compute balanced, BatchNorm stabilises/accelerates
training (so the conv has no bias), MaxPool halves resolution and adds translation tolerance;
`GlobalAveragePooling` instead of Flatten → far fewer parameters, less overfitting on ~1k images;
`Dropout(0.3)` regularises the head; `Dense(1, sigmoid)` outputs P(AI). **Loss** binary cross-entropy
(the correct likelihood for one sigmoid output); **optimizer** Adam lr 1e-3 (adaptive, robust default);
**metrics** accuracy plus precision, recall and AUC because accuracy alone hides which error happens.
Paste `layer_table.csv` as the table of output shape + params per layer.

### 2.3 Transfer learning (3)
▶ `python scripts/run_experiments.py R2` (MobileNetV2) and `R7` (EfficientNetB0)
📸 `training_curves.png` (shows both stages, grey line = start of fine-tuning), `trainable_layers.json`.
✍ Stage 1: MobileNetV2 without its classifier (`include_top=False`), **all base layers frozen**, new head
GAP → Dropout → Dense(sigmoid), lr 1e-3 — only the head's ~1.3k params train. Stage 2: unfreeze only the
**last block** (`block_16_*` + `Conv_1`; BatchNorm stays frozen and the base runs with `training=False`
so its running statistics aren't destroyed by small batches), recompile with lr **1e-5** (100× smaller)
so the pretrained filters are nudged, not overwritten. **Preprocessing**: MobileNetV2 was trained on
inputs scaled to [-1, 1], so we use `Rescaling(1/127.5, offset=-1)` = `mobilenet_v2.preprocess_input`;
EfficientNetB0 in Keras contains its own normalisation and expects [0,255]. Feeding the wrong scale makes
the pretrained features meaningless.

### 2.4 Training discipline (1)
✍ `keras.utils.set_random_seed(42)` fixes Python, NumPy and TF seeds (shuffle, augmentation, init).
Callbacks: `EarlyStopping(val_loss, patience 5, restore_best_weights)`, `ModelCheckpoint(best.keras,
save_best_only on val_loss)` — stage 2 starts with the stage-1 best as threshold so a worse fine-tune never
overwrites it — and `ReduceLROnPlateau(factor 0.3)`. Best epoch = `<n>` (red dashed line in the curve,
also logged as `best_epoch` in MLflow). Remaining nondeterminism on GPU can be removed with `--deterministic`.

### 2.5 Evaluation (2)
✍ The test set is evaluated **once**, at the end of each run, with the checkpointed best model.
Threshold: chosen on **validation** by maximising **F0.5** (precision weighted 2× recall). **Which error is
worse?** Calling a *real photo fake* (false positive). A false accusation can discredit a journalist,
an artist or evidence, while a missed fake is the status quo without the tool; so we accept missing some
fakes to keep false accusations low, and the UI shows a caveat. Report both models:

| Model | Acc | Prec | Rec | F1 | ROC-AUC | FPR | thr |
|---|---|---|---|---|---|---|---|
| Baseline CNN | | | | | | | |
| MobileNetV2 (fine-tuned) | | | | | | | |
(Copy from `reports/run_comparison.csv`.) 📸 `confusion_matrix.png`, `roc_curve.png` for both.

### 2.6 Error analysis (1)
▶ `python -m src.error_analysis --run outputs/mnv2_ref` → `reports/figures/misclassified_grid.png` + table.
The script proposes a hypothesis from measurable cues (sharpness, saturation, size, aspect ratio) — check
each image and edit (blur, painting/illustration style, watermark/text, tiny upscaled image, studio lighting…).

### 2.7 Keras Tuner (2)
▶ `python labs/keras_tuner_toy.py` (Hyperband, Fashion-MNIST, units 32–512 & lr) → `reports/tuner_toy.csv`
▶ `python -m src.tune --trials 10 --epochs 6` (RandomSearch: lr ∈ {1e-3,3e-4,1e-4,3e-5}, dropout ∈ {0.2,0.3,0.5},
unfrozen layers ∈ {0,10,20,40}) → `reports/tuner_visionguard.csv`, `tuner_importance.json`
✍ "Which mattered most": the hyper-parameter with the largest spread of mean val_loss across its values —
typically the **learning rate** (too high + unfrozen layers wrecks pretrained features). Selecting the
single best of many trials by validation score **overfits the validation set**: with ~200 validation
images, differences of 1–2 % are noise, and the max of many noisy scores is optimistically biased. That's
why the test set is untouched during tuning and reported once.

### Bonus — Grad-CAM
▶ `python -m src.gradcam --run outputs/mnv2_ref` → `reports/figures/gradcam.png`; comment whether the model
looks at faces/skin texture/background (good) or borders/watermarks (shortcut).

---
## Chapter 3 — MLflow & DagsHub

### 3.1 MLflow locally (2)
▶ `python labs/mlflow_toy.py --lr 0.001` · `python labs/mlflow_toy.py --lr 0.01` · `mlflow ui` →
http://127.0.0.1:5000 → tick both runs → **Compare**.
📸 both run pages + the comparison (parallel-coordinates / metric chart).
✍ **Experiment** – a named folder grouping runs that attack the same problem. **Run** – one execution of
training code, with its inputs and outputs recorded. **Parameter** – an input setting fixed before the
run (lr, epochs). **Metric** – a numeric result, possibly logged per step (val_loss per epoch). **Artifact**
– any output file (plot, model, CSV). **Tag** – free-form key/value metadata used to label/filter runs
(git commit, author, data version).

### 3.2 DagsHub setup (2)
▶ Create the repo on DagsHub → **Connect a repository → GitHub** (mirror) → green **Remote** button →
**Experiments** tab gives `https://dagshub.com/<user>/23L-0548_A2_Ops.mlflow`. User settings → Tokens →
New token. Put URI, username, token in `.env` (`cp .env.example .env`). `python labs/mlflow_toy.py --remote`.
📸 DagsHub project page, toy run on DagsHub, `git check-ignore -v .env` output (proves it's ignored) and
`git status` not listing `.env`.
✍ MLflow reads `MLFLOW_TRACKING_URI/USERNAME/PASSWORD` from the environment; `python-dotenv` loads them from
`.env`. Credentials in code would be pushed to GitHub forever and anyone could write to our tracking server.

### 3.3 Instrumenting training (3)
▶ manual: `R2` · autolog: `R8` (`--autolog`)
📸 run page of both: Parameters, Metrics charts, Tags (`git_commit`, `dvc_data_version`, `seed`).
✍ **Manual** (`MlflowEpochLogger` callback + explicit calls): we choose exactly what is logged — every
param, per-epoch train/val metrics *including learning-rate*, continuous step numbering across both
stages, test metrics at the tuned threshold, custom tags and artifacts. More code, easy to forget things.
**Autolog** (`mlflow.tensorflow.autolog`): zero code; captures `fit()` arguments (epochs, batch size),
optimizer name/lr, per-epoch metrics, early-stopping info and model summary automatically. It misses what
happens outside `fit()`: test metrics, the chosen threshold, git/DVC tags, confusion matrix, split lists —
and each `fit()` call is treated separately (we put stage 2 in a nested child run). We disabled autolog's
model logging (`log_models=False`) to keep one consistent model format (pyfunc, below).

### 3.4 Artifacts (1.5)
✍ Each run has `plots/` (curves, confusion matrix, ROC, augmented batch), `splits/` (train/val/test CSV),
`info/` (requirements_frozen.txt = package list, model summary, layer table, history, test predictions,
meta.json), `keras_model/visionguard.keras`, and `model/` — an MLflow **pyfunc** wrapper (`src/vg_pyfunc.py`)
bundling the .keras file with meta.json (img size + threshold). 📸 Artifacts tab.

### 3.5 Comparison (2)
▶ `python scripts/run_experiments.py` then `python scripts/compare_runs.py` → table + `reports/run_comparison.png`
(also DagsHub Experiments → select runs → Compare).

| Run | One change vs reference R2 |
|---|---|
| R1 cnn_baseline | architecture (from scratch) |
| R2 mnv2_ref | reference: MobileNetV2, lr 1e-3, aug, 224, fine-tune last block |
| R3 mnv2_lr3e-4 | learning rate |
| R4 mnv2_noaug | no augmentation |
| R5 mnv2_frozen_only | frozen layers (no fine-tuning) |
| R6 mnv2_img160 | image size 160 |
| R7 effb0_ref | backbone EfficientNetB0 |
| R8 mnv2_ref_autolog | logging approach (should match R2) |

✍ Choose by **ROC-AUC** (threshold-independent ranking quality) first, then **false-positive rate / precision
at the tuned threshold** (our worse error), then F1; break near-ties (<0.01) in favour of the smaller/faster
model because it serves on CPU.

### 3.6 Registry (1.5)
▶ `python scripts/register_model.py --run-id <runner-up>` (v1) · `python scripts/register_model.py --run-id <best> --champion` (v2 @champion)
· `python scripts/predict_by_alias.py some.jpg` · then `python scripts/export_champion.py`
📸 DagsHub → Models (versions + alias) and the terminal output.
✍ The serving app never calls DagsHub per request: `export_champion.py` downloads the champion **once**
into `models/champion/`. Compose mounts that folder read-only into the web container; the cloud API image
copies it in at build time. Changing champion = re-run export (+ rebuild the API image).

---
## Chapter 4 — Flask web app
### 4.1 Inference module (2)
▶ `python -m serving.inference path/real.jpg path/ai.png`  📸 console output.
✍ Loading a .keras model takes seconds and ~hundreds of MB; per-request loading would make every request
slow and concurrent requests could run out of memory, so `Predictor` is created once when the app starts
(and warmed up with a dummy forward pass). Preprocessing must match training (**training/serving skew**):
any difference — RGB vs BGR, other resize method, crop vs stretch, double scaling — shifts the input
distribution and silently degrades accuracy. We share the logic: decode → RGB first frame → bilinear
stretch-resize → uint8 round → model does its own scaling. (EXIF rotation is deliberately *not* applied
because TF's decoder ignores it in training.)

### 4.2 Routes & validation (3)
Routes: `/` upload, `POST /predict` → result page, `/history`, `/health` (+ `/api/history` JSON).
Validation (`serving/validation.py`): extension allow-list → size limit (Flask `MAX_CONTENT_LENGTH` aborts
early + explicit check) → Pillow `verify()` + full `load()` (a renamed .txt or truncated file fails) →
real format allow-list (content sniffing, not trusting the extension) → decompression-bomb guard.
File names are sanitised (no `../`, no odd characters). The upload is read into memory and never saved.
📸 the three friendly error pages: `notes.txt` (415), a 10 MB image (413), `fake.jpg` = text renamed (422).

### 4.3 HTML/CSS (2.5)  📸 upload page, result page (verdict + confidence bar + caveat), history page.
### 4.4 Configuration (1)
✍ Every setting comes from env vars (`MODEL_PATH`, `THRESHOLD`, `DATABASE_URL`, `MAX_UPLOAD_MB`,
`INFERENCE_MODE`, `MODEL_API_URL/KEY`, `SECRET_KEY`) → same image runs in dev, compose and cloud.
Debug mode is off: the Werkzeug debugger allows **arbitrary code execution** from the browser and shows
stack traces/secrets; the auto-reloader also double-loads the model. In the container we use gunicorn (a
production WSGI server); locally `flask --app webapp.app run` (debug off by default).
📸 `curl http://localhost:5000/health`.

## Chapter 5 — Database trail (8)
✍ Table `predictions`: id, created_at (UTC), filename, label, confidence, prob_ai, threshold, model_name,
model_version, inference_mode (local/remote), latency_ms, image format/size, file size, client IP → answers
*what, when, which model, how confidently*. Written in `/predict` after every successful inference (rejected
uploads are not logged); `/history` shows the latest 100. SQLAlchemy makes the same code work on Postgres
and SQLite. 📸 history page + `docker compose exec db psql -U visionguard -d visionguard -c "select id,created_at,label,confidence,model_version from predictions order by id desc limit 5;"`

## Chapter 6 — Docker
### 6.1 Warm-up (2)
```bash
docker run -d --name web1 -p 8080:80 -v "$PWD/labs/www:/usr/share/nginx/html:ro" nginx:alpine
curl localhost:8080          # port mapping host 8080 -> container 80
docker logs web1
docker exec -it web1 sh      # shell inside; `ls /usr/share/nginx/html`, exit
docker ps -a && docker images
docker stop web1 && docker rm web1 && docker rmi nginx:alpine
```
(create `labs/www/index.html` with "hello android" first.) 📸 each command.
✍ **Image** – read-only template (filesystem + metadata) built from a Dockerfile. **Container** – a running
(or stopped) instance of an image with its own writable layer, process and network. **Layer** – each
Dockerfile instruction produces a cached, content-addressed filesystem diff; images share layers.
**Volume** – storage managed outside the container's writable layer, so data survives container removal.
**Port mapping** – forwarding a host port to a container port (`-p host:container`).

### 6.2 Dockerfile (3)
✍ See comments in `webapp/Dockerfile`: `python:3.11-slim` base; `requirements-web.txt` copied and installed
**before** the source → editing code reuses the cached dependency layer (rebuild in seconds instead of
minutes); `PIP_NO_CACHE_DIR`; non-root `appuser` (uid 10001); `HEALTHCHECK` hitting `/health`; `.dockerignore`
keeps `.env`, data, `.git`, outputs out of the context; secrets only via runtime env vars.
**TensorFlow choice**: `tensorflow-cpu` — the default `tensorflow` wheel bundles CUDA libraries (~2 GB+)
that are useless on a CPU server; one 224×224 forward pass takes ~30–80 ms on CPU.
📸 `docker compose exec web whoami` → `appuser`, and `id`.

### 6.3 Compose (4)
▶ `docker compose up -d --build` · `docker compose ps` (both **healthy**)
Proof DB port closed: `docker compose port db 5432` (prints nothing/error) and from host
`python -c "import socket; socket.create_connection(('localhost',5432),2)"` → ConnectionRefused; but
`docker compose exec web python -c "import socket; print(socket.create_connection(('db',5432),2))"` works
→ services talk by **service name** on the internal network.
✍ Model **mounted** (`./models/champion:/models/champion:ro`) rather than built in: the web image stays
generic and smaller, a new champion needs only `export_champion.py` + `docker compose restart web`, and the
model is versioned by MLflow/DVC, not by image tag. (For the cloud API we bake it in — no repo there.)
`.env` is git-ignored; `.env.example` documents the variables.

### 6.4 Image analysis (1.5)
```bash
docker build -f webapp/Dockerfile.naive -t vg-web:naive .
docker build -f webapp/Dockerfile -t vg-web:slim .
docker images | grep vg-web
docker history vg-web:slim --format "{{.Size}}\t{{.CreatedBy}}" | head
```
✍ Report both sizes (roughly ~5 GB vs ~1.6–2 GB). Largest layer = the `pip install` layer (TensorFlow),
then the base. Optimisations applied: slim base, CPU-only TF, no pip cache, `.dockerignore`.

### 6.5 Operations & troubleshooting (1.5)
```bash
docker compose stop && docker compose start && docker compose restart web
docker compose logs -f web    |   docker compose logs db
docker compose exec web sh    |   docker compose exec db psql -U visionguard
# break it: set POSTGRES_PASSWORD in .env to something else for web only, e.g.
#   DATABASE_URL uses the new password but the volume was initialised with the old one
docker compose up -d web && docker compose logs web
```
Expected log: `[db] not ready (OperationalError) ... password authentication failed for user "visionguard"`.
Diagnosis: the Postgres volume keeps the password from first initialisation; fix: restore the original
password (or `docker compose down -v` to re-init, which deletes data). Second failure: rename
`models/champion/visionguard.keras` → log shows `FileNotFoundError: Model file not found ... export_champion.py`.

### 6.6 GitHub Actions (2)
Toy: create a throwaway repo, copy the contents of `labs/gha_toy/` (incl. `.github/`), push → green;
change the assert to `== 6` → red; revert → green. Project: `.github/workflows/ci.yml` (ruff + pytest, then
Docker build); badge in README. ✍ Secrets live in **GitHub → Settings → Secrets and variables → Actions**,
referenced as `${{ secrets.NAME }}`; they are encrypted and masked in logs. In the workflow file they would be
public in the repo history forever.

## Chapter 7 — API deployment (15) → follow `docs/DEPLOYMENT.md`.
## Chapter 8 — Integration (15)
▶ In `.env`: `INFERENCE_MODE=remote`, `MODEL_API_URL=https://<host>`, `MODEL_API_KEY=<key>` →
`docker compose up -d` (recreates web; local model not needed) → `/health` shows `inference_mode: remote`.
Upload 5–6 images (real, AI, edge cases) → result pages say "Served by: cloud API" → history shows rows with
mode `remote` → Azure log stream shows the matching `predict` lines.
📸 every result, the history page, the psql query, the cloud logs side by side.
