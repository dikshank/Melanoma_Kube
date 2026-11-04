# Melanoma on Minikube: run order

Commands are for **Windows cmd**, run from the **repo root** (`C:\...\Melanoma_Kube`).
In bash, replace `^` with `\` and `findstr` with `grep`.

The order matters because of dependencies:
**storage, then Airflow (Postgres first), then model, then backend, then frontend, then monitoring, then ngrok.**

---

## 0. Start minikube

```bat
minikube start --cpus=5 --memory=12g
kubectl apply -f k8s/namespace.yaml
```

## 1. Build the 3 images inside minikube

Build again any time you change code. `backend` and `airflow` changed in this update, so **rebuild both**.

```bat
minikube image build -t frontend-image:v1 frontend
minikube image build -t backend-image:v1  backend
minikube image build -t airflow-image:v1  airflow
minikube image ls | findstr image
```

## 2. Storage, plus loading your data and the B1 model into it

```bat
kubectl apply -f k8s/storage/data-pvc.yaml -f k8s/storage/logs-pvc.yaml -f k8s/storage/mlruns-pvc.yaml
kubectl apply -f k8s/storage/pvc-loader.yaml
kubectl wait --for=condition=Ready pod/pvc-loader -n melanoma-production --timeout=120s
```

Copy the dataset. Use **relative paths**: `kubectl cp` misreads `C:\...` (the `C:` looks like a pod name).

```bat
kubectl cp data/raw            melanoma-production/pvc-loader:/data/raw
kubectl cp data/processed      melanoma-production/pvc-loader:/data/processed
kubectl cp data/baseline_stats melanoma-production/pvc-loader:/data/baseline_stats
```

Copy **your B1 file**. Put it in the repo folder first, then copy it to exactly this name.
This step is optional: without it, step 4 registers a **fallback model** and the UI shows a yellow warning banner.

```bat
kubectl exec pvc-loader -n melanoma-production -- mkdir -p /data/seed_model
kubectl cp efficientnet_b1.pt melanoma-production/pvc-loader:/data/seed_model/efficientnet_b1.pt
```

Fix permissions (Airflow runs as uid 50000), check the result, and clean up:

```bat
kubectl exec pvc-loader -n melanoma-production -- chmod -R 777 /data /logs /mlruns
kubectl exec pvc-loader -n melanoma-production -- ls /data /data/seed_model
kubectl delete pod pvc-loader -n melanoma-production
```

## 3. Airflow

```bat
kubectl apply -f k8s/airflow/airflow-config.yaml -f k8s/airflow/airflow-secret.yaml
kubectl apply -f k8s/airflow/postgres.yaml
kubectl wait --for=condition=Available deploy/postgres -n melanoma-production --timeout=180s

kubectl apply -f k8s/airflow/airflow-init-job.yaml
kubectl wait --for=condition=complete job/airflow-init -n melanoma-production --timeout=300s

kubectl apply -f k8s/airflow/statsd-exporter.yaml
kubectl apply -f k8s/airflow/airflow-schedular.yaml
kubectl apply -f k8s/airflow/airflow-webserver.yaml
```

If you applied the **old** statsd file before, delete the old Service once:
`kubectl delete svc statsd-exporter-service -n melanoma-production`.

## 4. Register your EfficientNet-B1 as the Production model

```bat
kubectl apply -f k8s/model/register-b1-job.yaml
kubectl logs -f job/register-b1-model -n melanoma-production
```

You should see `Loaded ...`, `Sanity check OK: input 1x3x224x224 -> output (1, 2)`, the val/test metrics, and then
`✅ efficientnet_b1 registered as melanoma_classifier v1 and moved to Production`.

**No B1 file?** You'll instead see `⚠️ Weights file not found` followed by `efficientnet_b1_fallback registered ... [FALLBACK ...]`.
That's the ImageNet EfficientNet-B1 with a head trained in seconds on your train images. The app works, the model dot turns yellow, and a banner warns users.
To switch to your real model later, copy the `.pt` in (step 2), then delete and re-apply the Job. Within about 30 seconds the backend picks it up and the banner disappears.
The fallback never overwrites a real model. To keep the generated fallback in git, see `airflow/training/fallback/README.md`.

If it fails, read the error, fix it, then `kubectl delete job register-b1-model -n melanoma-production` and apply again.
Jobs are immutable, so they have to be deleted before they can be re-run.

## 5. Backend and frontend

```bat
kubectl apply -f k8s/backend/deployment.yaml
kubectl apply -f k8s/frontend/deployment.yaml
kubectl get pods -n melanoma-production -l app=backend -w
```

Backend pods show `0/1` until `/ready` finds the Production model. After that they turn `1/1`.
The frontend needs the `backend` **Service** to exist, otherwise nginx fails with `host not found in upstream "backend"`.

## 6. MLflow, Prometheus, Grafana

```bat
kubectl apply -f k8s/mlflow/mlflow.yaml

kubectl create configmap grafana-dash-melanoma -n melanoma-production ^
  --from-file=grafana/dashboards/melanoma_dashboards/ --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap grafana-dash-airflow -n melanoma-production ^
  --from-file=grafana/dashboards/airflow_dashboards/ --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/monitoring/prometheus.yaml
kubectl apply -f k8s/monitoring/grafana.yaml
```

## 7. ngrok (public URL)

**Revoke the old token** that was committed in `.env` and create a new one at dashboard.ngrok.com. Then:

```bat
kubectl create secret generic ngrok-secret -n melanoma-production --from-literal=NGROK_AUTHTOKEN=<new-token>
kubectl apply -f k8s/ngrok/ngrok.yaml
kubectl logs deploy/ngrok -n melanoma-production | findstr "url="
```

## 8. Open the UIs

Each command blocks, so run each one in its own terminal.

| What | Command | URL |
|---|---|---|
| App (without ngrok) | `kubectl port-forward svc/frontend 8081:80 -n melanoma-production` | http://localhost:8081 |
| Airflow | `kubectl port-forward svc/airflow-webserver 8080:8080 -n melanoma-production` | http://localhost:8080 (admin / admin) |
| MLflow | `kubectl port-forward svc/mlflow 5000:5000 -n melanoma-production` | http://localhost:5000 |
| Prometheus | `kubectl port-forward svc/prometheus 9090:9090 -n melanoma-production` | http://localhost:9090/targets |
| Grafana | `kubectl port-forward svc/grafana 3000:3000 -n melanoma-production` | http://localhost:3000 (admin / admin) |
| ngrok inspector | `kubectl port-forward svc/ngrok 4040:4040 -n melanoma-production` | http://localhost:4040 |

---

## After changing code

```bat
minikube image build -t backend-image:v1 backend
kubectl rollout restart deploy/backend-deployment -n melanoma-production
```

The tag stays `v1`, so Kubernetes can't see that anything changed. `rollout restart` creates new pods, and those pods pick up the rebuilt image.
Do the same for `frontend-deployment`, `airflow-schedular` and `airflow-webserver`.

## Quick troubleshooting

| Symptom | Usual cause | Check |
|---|---|---|
| Pod `Pending` | PVC missing or not enough CPU/RAM | `kubectl describe pod <name> -n melanoma-production` (Events at the bottom) |
| `ErrImagePull` | image not built inside minikube | `minikube image ls \| findstr <name>` |
| Backend `0/1` forever | no Production model | `kubectl logs job/register-b1-model -n melanoma-production` |
| Frontend `CrashLoopBackOff` | `backend` Service missing | `kubectl get svc -n melanoma-production` |
| Prometheus target DOWN | wrong Service name or port | Prometheus UI, then Status, then Targets |
| Scheduler `OOMKilled` (137) | B1 retraining at 224px is heavy | raise the limit in `airflow-schedular.yaml` or lower `batch_size` |
| `Permission denied` writing to a PVC | uid 50000 vs root | rerun the `chmod -R 777` step with the loader pod |
