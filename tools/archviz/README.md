# archviz: interactive architecture snapshot

One command turns the running cluster plus this repo into a single, offline, clickable `architecture.html`.

```bat
pip install pyyaml
python tools/archviz/snapshot.py
```

Run it from the repo root while minikube is up. Then double-click `architecture.html`.

**What's inside**
- **Map:** every workload, Service, volume and connection ("gate"). Drag to pan, scroll to zoom.
- **Rooms:** double-click a box to see its incoming gates, outgoing gates, pods, container, volumes, probes and isolation (namespace, Service type, NetworkPolicies, cgroup limits, user, RBAC, filesystem).
- **Gates:** click any connection to see what would travel through it, the hop-by-hop route (DNS → ClusterIP → kube-proxy → pod), and the file and line that create it.
- **Scenarios:** step-by-step animations ("user clicks Analyse", "feedback → retrain → hot reload", "a pod dies").
- **Interview:** questions, each with a "Show me" button.

**Deep links:** `architecture.html#room=postgres`, `#gate=L12`, `#flow=predict`

**Privacy:** Secret values are never written. Only Secret names and key names are kept. The file contains pod names and cluster IPs, so keep it out of git (it's in `.gitignore`).

**Options:** `--namespace`, `--out`, `--no-cluster` (build from `k8s/*.yaml` only, with no live pod data).

**How the gates are found:**

| Gate | Source |
|---|---|
| frontend → backend | `frontend/nginx.conf` `proxy_pass` lines, matched to the FastAPI routes in `backend/main.py` |
| `name:port` references (DB URL, StatsD host, Grafana datasource, ngrok target) | args, env, ConfigMaps and Secrets |
| Prometheus targets | its `prometheus.yml`, including `kubernetes_sd` pod discovery |
| volume gates | `volumeMounts`; probe gates come from probes; config gates come from `envFrom` and config volumes |
| a few runtime facts (Airflow's log server on :8793, port-forwards) | added explicitly and labelled as such |
