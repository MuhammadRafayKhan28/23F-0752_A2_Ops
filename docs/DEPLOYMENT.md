# Task 7 — Deploying the model API to the cloud

**Pathway chosen: C — Azure (Azure for Students, no credit card needed with a university email).**
The model API (`api/`) runs as a container in **Azure Container Apps** (free monthly grant, HTTPS URL out
of the box). The Flask UI + PostgreSQL keep running locally in Docker Compose and call the API (Task 8).
An alternative with Google Cloud Run is at the end; both use the same image.

Why not Vercel for the model? Vercel functions are capped at ~250 MB unzipped — TensorFlow alone is ~600 MB —
and have no persistent disk. Why not a VM? It works (use `api/docker-compose.api.yml`), but a managed
container service handles HTTPS, restarts and the firewall for you.

| Compose element | Azure equivalent |
|---|---|
| `api` service image | Docker Hub image `docker.io/<user>/23f-0752_a2_ops-api` |
| container + `ports: 8000` | Container App with external ingress, target port 8000 |
| env vars / `.env` | Container App secrets → env var `API_KEY=secretref:api-key` |
| `restart: unless-stopped` + healthcheck | Container Apps restarts + health probes |

## 0. Prerequisites
- Champion exported: `python scripts/export_champion.py` → `models/champion/visionguard.keras` + `meta.json`
- Docker Desktop, a Docker Hub account, Azure CLI (`az`), Azure for Students activated at
  https://azure.microsoft.com/free/students
- A long random API key: `python -c "import secrets; print(secrets.token_urlsafe(32))"`

## 1. Build and test the image locally
```bash
docker build -f api/Dockerfile -t <dockerhub-user>/23f-0752_a2_ops-api:1 .
docker run --rm -p 8000:8000 -e API_KEY=<key> <dockerhub-user>/23f-0752_a2_ops-api:1
curl http://localhost:8000/health
curl -H "X-API-Key: <key>" -F "file=@samples/real1.jpg" http://localhost:8000/predict
```
📸 `docker images` showing the tag, `/health` JSON, one prediction.

## 2. Push to Docker Hub
```bash
docker login
docker push <dockerhub-user>/23f-0752_a2_ops-api:1
```
(Docker image names must be lowercase, hence `23f-0752_a2_ops-api`.)

## 3. Create the Azure resources
```bash
az login
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az group create --name rg-23f-0752-a2-ops --location eastus

az containerapp env create --name env-23f-0752-a2-ops --resource-group rg-23f-0752-a2-ops --location eastus
az containerapp create \
  --name vg-23f-0752-a2-ops \
  --resource-group rg-23f-0752-a2-ops \
  --environment env-23f-0752-a2-ops \
  --image docker.io/<dockerhub-user>/23f-0752_a2_ops-api:1 \
  --target-port 8000 --ingress external \
  --cpu 1.0 --memory 2.0Gi \
  --min-replicas 0 --max-replicas 1 \
  --secrets api-key=<key> \
  --env-vars API_KEY=secretref:api-key MAX_UPLOAD_MB=5
az containerapp show -n vg-23f-0752-a2-ops -g rg-23f-0752-a2-ops --query properties.configuration.ingress.fqdn -o tsv
```
(PowerShell: replace the trailing `\` with a backtick `` ` ``.)
The last command prints the host, e.g. `vg-23f-0752-a2-ops.<random>.eastus.azurecontainerapps.io`.
**API URL = `https://<that host>`** — put it in the report front matter.

Notes: `min-replicas 0` = scale to zero when idle (keeps you inside the free grant) — the first request
after idle takes ~20–40 s while the container starts and loads TensorFlow. `2.0Gi` memory is needed:
TensorFlow + the model use ~0.9–1.2 GB. The key is stored as a Container App *secret*, not as plain text.

📸 Azure portal → Container App overview (status Running, Application Url), Revisions, Log stream.

## 4. Test with Postman (from a network that is not yours — e.g. mobile hotspot)
1. New request → `GET https://<host>/health` → Send → 200 with model name/version.
2. New request → `POST https://<host>/predict`
   - Headers: `X-API-Key: <key>`
   - Body → form-data → key `file`, type **File**, choose an image → Send.
3. Repeat without the header → expect `401 {"error": "missing or invalid API key"}`.
4. Send a `.txt` renamed to `.jpg` → expect `422`.
5. Azure portal → Log stream shows a `predict {...}` line for each request (the "logged result").

📸 each Postman response with the URL bar visible, and the log stream.

## 5. Updating the model
`export_champion.py` → build tag `:2` → push → 
`az containerapp update -n vg-23f-0752-a2-ops -g rg-23f-0752-a2-ops --image docker.io/<user>/23f-0752_a2_ops-api:2`

## 6. Clean up after grading
`az group delete --name rg-23f-0752-a2-ops`

---
## Alternative: Google Cloud Run
```bash
gcloud auth login && gcloud config set project <project-id>
gcloud services enable run.googleapis.com
docker tag <user>/23f-0752_a2_ops-api:1 docker.io/<user>/23f-0752_a2_ops-api:1
gcloud run deploy vg-23f-0752-a2-ops --image docker.io/<user>/23f-0752_a2_ops-api:1 \
  --region us-central1 --port 8000 --memory 2Gi --cpu 1 --allow-unauthenticated \
  --set-env-vars API_KEY=<key>
```
## Alternative: any Ubuntu VM (Oracle Always Free / Azure B1s)
```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
git clone https://github.com/<user>/23F-0752_A2_Ops && cd 23F-0752_A2_Ops
# copy models/champion/ from your laptop with scp, then:
API_KEY=<key> sudo -E docker compose -f api/docker-compose.api.yml up -d --build
```
Open TCP 8000 in BOTH firewalls: the cloud network rules (OCI Security List / Azure NSG) and the VM's own
(`sudo ufw allow 8000/tcp`; on Oracle Ubuntu images also `sudo iptables -I INPUT 6 -p tcp --dport 8000 -j ACCEPT`).
