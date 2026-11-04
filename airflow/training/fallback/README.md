# Fallback model

`efficientnet_b1_fallback.pt` goes in this folder (it is allowed by an exception in `.gitignore`).

It is EfficientNet-B1 with the original ImageNet weights plus a 2-class head trained on
`data/processed/v1` train images. `training/src/register_pretrained.py` registers it,
tagged `is_fallback=true`, **only** when your own `efficientnet_b1.pt` is missing and no
model is in Production yet. The UI then shows a yellow "fallback model" banner.

If this file is missing, the `register-b1-model` Job builds the fallback itself (a few
seconds) and saves it to `/opt/airflow/data/seed_model/efficientnet_b1_fallback.pt` on
data-pvc. To keep it, copy it out and commit it:

```bat
kubectl apply -f k8s/storage/pvc-loader.yaml
kubectl cp melanoma-production/pvc-loader:/data/seed_model/efficientnet_b1_fallback.pt airflow/training/fallback/efficientnet_b1_fallback.pt
kubectl delete pod pvc-loader -n melanoma-production
git add airflow/training/fallback/efficientnet_b1_fallback.pt
git commit -m "Add fallback model"
minikube image build -t airflow-image:v1 airflow
```
