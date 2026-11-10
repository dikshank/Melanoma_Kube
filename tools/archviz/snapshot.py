#!/usr/bin/env python3
"""
snapshot.py - one-time, interactive architecture snapshot of Melanoma_Kube
==========================================================================

Run it ONCE while minikube is up. It collects:

  * from the cluster (kubectl):  every workload, pod, Service, PVC, ConfigMap,
    Secret (NAMES and KEY NAMES only - values are never written anywhere),
    ServiceAccount/RBAC, NetworkPolicy, probes, limits, mounts, IPs, the node,
    and the kube-system control-plane pods
  * from your project files:     k8s/*.yaml (which file defines what),
    frontend/nginx.conf (proxy routes), backend/main.py (API routes),
    airflow/dags/*.py (DAGs + tasks), Prometheus/Grafana config

...works out every "gate" (who talks to whom, on which port and protocol, and
what travels through it), and writes ONE self-contained file:

    architecture.html   <- double-click it any time. No server, no internet,
                           no minikube needed after this.

Usage (from the repo root, with minikube running):

    pip install pyyaml
    python tools/archviz/snapshot.py

Options:
    --namespace NS     default: melanoma-production
    --out FILE         default: architecture.html (in the repo root)
    --no-cluster       build from the YAML files only (nothing live: no pod
                       names, IPs or status). Useful when minikube is off.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # PyYAML: parses the k8s/*.yaml files
except ImportError:  # the script still works with the cluster alone
    yaml = None

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "viewer.html"


# ════════════════════════════════════════════════════════════════════════════
# 1. Collect
# ════════════════════════════════════════════════════════════════════════════

def kubectl(*args: str, quiet: bool = False):
    """Run kubectl and return parsed JSON, or None if it fails."""
    exe = shutil.which("kubectl")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    if out.returncode != 0:
        if quiet:
            return None
        print(f"  ! kubectl {' '.join(args)}: {out.stderr.strip()[:200]}")
        return None
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return out.stdout.strip()


def collect_cluster(ns: str) -> dict | None:
    """Everything we need from the API server, grouped by kind."""
    print(f"• Reading the cluster (namespace {ns}) ...")
    ver = kubectl("version", "-o", "json")
    if not isinstance(ver, dict):
        print("  ! kubectl can't reach a cluster - continuing from YAML files only")
        return None
    ctx = kubectl("config", "current-context")
    kinds = ("deployments,statefulsets,daemonsets,jobs,pods,services,endpoints,"
             "persistentvolumeclaims,configmaps,secrets,serviceaccounts,roles,"
             "rolebindings,networkpolicies,replicasets")
    lst = kubectl("get", kinds, "-n", ns, "-o", "json") or {"items": []}
    by_kind: dict[str, list] = {}
    for it in lst.get("items", []):
        if it.get("kind") == "Secret":
            # SECURITY: keep only key names. Decode values in memory once, only
            # to spot host:port references (e.g. the DB URL), never to store.
            it["_keys"] = sorted((it.get("data") or {}).keys())
            it["_text"] = " ".join(_b64(v) for v in (it.get("data") or {}).values())
            it.pop("data", None)
            it.pop("stringData", None)
            (it.get("metadata") or {}).pop("annotations", None)  # last-applied holds values
        by_kind.setdefault(it["kind"], []).append(it)
    nodes = kubectl("get", "nodes", "-o", "json") or {"items": []}
    sysp = kubectl("get", "pods", "-n", "kube-system", "-o", "json") or {"items": []}
    sc = kubectl("get", "storageclass", "-o", "json") or {"items": []}
    print(f"  found {sum(len(v) for v in by_kind.values())} objects")
    return {
        "context": ctx if isinstance(ctx, str) else "",
        "version": (ver.get("serverVersion") or {}).get("gitVersion", ""),
        "objects": by_kind,
        "nodes": nodes.get("items", []),
        "system_pods": sysp.get("items", []),
        "storageclasses": sc.get("items", []),
    }


def _b64(v: str) -> str:
    try:
        return base64.b64decode(v).decode("utf-8", "replace")
    except Exception:
        return ""


def collect_manifests(repo: Path) -> list[dict]:
    """Every object in k8s/**/*.yaml, tagged with the file (and line) it came from."""
    out = []
    if yaml is None:
        print("  ! PyYAML not installed (pip install pyyaml): 'defined in' links will be missing")
        return out
    for f in sorted((repo / "k8s").rglob("*.y*ml")):
        text = f.read_text(encoding="utf-8", errors="replace")
        try:
            docs = list(yaml.safe_load_all(text))
        except yaml.YAMLError as e:
            print(f"  ! could not parse {f}: {e}")
            continue
        for d in docs:
            if isinstance(d, dict) and d.get("kind"):
                name = (d.get("metadata") or {}).get("name", "")
                d["_file"] = f.relative_to(repo).as_posix()
                d["_line"] = _line_of(text, rf"^\s*name:\s*{re.escape(name)}\s*(#.*)?$")
                if d["kind"] == "Secret":   # same rule as the cluster: never keep values
                    vals = {**(d.get("stringData") or {}),
                            **{k: _b64(v) for k, v in (d.get("data") or {}).items()}}
                    d["_keys"] = sorted(vals.keys())
                    d["_text"] = " ".join(str(v) for v in vals.values())
                    d.pop("stringData", None)
                    d.pop("data", None)
                out.append(d)
    print(f"• Read {len(out)} objects from k8s/*.yaml")
    return out


def _line_of(text: str, pattern: str) -> int:
    for i, line in enumerate(text.splitlines(), 1):
        if re.search(pattern, line):
            return i
    return 0


def parse_nginx(repo: Path) -> list[dict]:
    """location blocks with proxy_pass, from frontend/nginx.conf."""
    f = repo / "frontend" / "nginx.conf"
    if not f.exists():
        return []
    routes, cur, cur_line = [], None, 0
    for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        m = re.match(r"\s*location\s+(\S+)\s*\{", line)
        if m:
            cur, cur_line = m.group(1), i
        m = re.search(r"proxy_pass\s+http://([\w.-]+):(\d+)", line)
        if m and cur:
            routes.append({"path": cur, "host": m.group(1), "port": int(m.group(2)),
                           "file": "frontend/nginx.conf", "line": cur_line})
    return routes


def parse_fastapi(repo: Path) -> list[dict]:
    """@app.get/@app.post routes from backend/main.py."""
    f = repo / "backend" / "main.py"
    if not f.exists():
        return []
    out = []
    for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
        m = re.match(r'\s*@app\.(get|post|put|delete)\(\s*"([^"]+)"', line)
        if m:
            out.append({"method": m.group(1).upper(), "path": m.group(2),
                        "file": "backend/main.py", "line": i})
    return out


def parse_dags(repo: Path) -> list[dict]:
    dags = []
    for f in sorted((repo / "airflow" / "dags").glob("*.py")):
        t = f.read_text(encoding="utf-8")
        dag_id = re.search(r'dag_id\s*=\s*"([^"]+)"', t)
        sched = re.search(r"schedule_interval\s*=\s*([^,\n]+)", t)
        tasks = re.findall(r'task_id\s*=\s*"([^"]+)"', t)
        chain = re.findall(r"^\s*(t\w+(?:\s*>>\s*t\w+)+)", t, re.M)
        if dag_id:
            dags.append({"dag_id": dag_id.group(1), "schedule": (sched.group(1).strip() if sched else "None"),
                         "tasks": tasks, "file": f"airflow/dags/{f.name}"})
    return dags


# ════════════════════════════════════════════════════════════════════════════
# 2. Model: components, services, volumes, configs
# ════════════════════════════════════════════════════════════════════════════

# What each component IS, in one or two sentences (things kubectl can't tell you).
DESCRIBE = {
    "frontend-deployment": "nginx serves the static UI (index.html, app.js, style.css) and reverse-proxies API paths to the backend, so the browser only ever talks to one origin.",
    "backend-deployment": "FastAPI inference service. Loads the Production model from the MLflow registry into memory at startup; /predict classifies an upload, /feedback records the true label; /ready hot-reloads newly promoted models.",
    "ngrok": "Tunnel agent: dials OUT to ngrok's cloud and forwards every public request to the frontend Service. No inbound port on your laptop.",
    "cloudflared": "Tunnel agent for your own domain: dials OUT to Cloudflare and forwards requests to the frontend Service.",
    "airflow-schedular": "Airflow scheduler with LocalExecutor: there are no worker pods, so every DAG task (data pipeline, python train.py) runs as a subprocess INSIDE this pod.",
    "airflow-webserver": "The Airflow UI (gunicorn). Clicking Trigger only inserts a dag_run row in Postgres; the scheduler does the work.",
    "airflow-init": "One-time Job: creates Airflow's tables in Postgres (airflow db migrate) and the admin user.",
    "postgres": "Airflow's metadata database (dag runs, task instances, users). Not ML data.",
    "statsd-exporter": "Bridge from push to pull: receives Airflow's StatsD packets over UDP and serves them as Prometheus metrics.",
    "prometheus": "Pull-based metrics store: scrapes /metrics endpoints every 15 s.",
    "grafana": "Dashboards. Its server queries Prometheus with PromQL.",
    "mlflow": "Read-only viewer (mlflow ui) over the MLflow files on mlruns-pvc. Nobody logs to it over the network.",
    "register-b1-model": "One-time Job: registers the first Production model (your EfficientNet-B1, or the ImageNet fallback) into the MLflow registry on mlruns-pvc.",
    "pvc-loader": "Temporary helper pod used with kubectl cp to put data into the PVCs.",
}

GROUP = [  # (substring in workload name, group)
    ("ngrok", "ingress"), ("cloudflared", "ingress"),
    ("frontend", "serving"), ("backend", "serving"),
    ("airflow", "ml"), ("postgres", "ml"), ("register", "ml"), ("mlflow", "ml"),
    ("prometheus", "observability"), ("grafana", "observability"), ("statsd", "observability"),
]

# Hand-tuned positions on the map (x, y) for known names; unknown ones are auto-placed.
LAYOUT = {
    "ext:internet": (40, 90), "ext:edge": (40, 300), "ext:you": (40, 780),
    "ngrok": (300, 270), "cloudflared": (300, 410), "register-b1-model": (300, 650), "pvc-loader": (300, 860),
    "frontend-deployment": (560, 90), "backend-deployment": (560, 340),
    "pvc:data-pvc": (860, 330), "pvc:logs-pvc": (860, 500), "pvc:mlruns-pvc": (860, 670),
    "prometheus": (1130, 90), "airflow-schedular": (1130, 470), "airflow-webserver": (1130, 700), "airflow-init": (1130, 900),
    "grafana": (1430, 90), "statsd-exporter": (1430, 300), "mlflow": (1430, 560), "postgres": (1430, 760),
    "pvc:postgres-data": (1430, 960), "pvc:prometheus-data": (1180, 260), "pvc:grafana-data": (1700, 120),
}


def group_of(name: str) -> str:
    for key, g in GROUP:
        if key in name:
            return g
    return "other"


def labels_match(selector: dict, labels: dict) -> bool:
    return bool(selector) and all(labels.get(k) == v for k, v in selector.items())


def build(cluster: dict | None, manifests: list[dict], repo: Path, ns: str) -> dict:
    live = cluster is not None
    objs = (cluster or {}).get("objects", {})

    def from_both(kind: str) -> list[dict]:
        """Live objects when the cluster is up; otherwise the YAML ones."""
        if live:
            return objs.get(kind, [])
        return [m for m in manifests if m["kind"] == kind]

    defined_in = {(m["kind"], (m.get("metadata") or {}).get("name")): (m["_file"], m.get("_line", 0)) for m in manifests}
    manifest_by = {(m["kind"], (m.get("metadata") or {}).get("name")): m for m in manifests}

    def where(kind, name):
        f = defined_in.get((kind, name))
        return {"file": f[0], "line": f[1]} if f else {"file": "(created from the command line)", "line": 0}

    pods = objs.get("Pod", [])
    endpoints = {e["metadata"]["name"]: e for e in objs.get("Endpoints", [])}

    # ── workloads ──────────────────────────────────────────────────────────
    comps: list[dict] = []
    wl_kinds = ["Deployment", "StatefulSet", "DaemonSet", "Job"]
    for kind in wl_kinds:
        for w in from_both(kind):
            name = w["metadata"]["name"]
            spec = w.get("spec", {})
            tpl = spec.get("template", {})
            pspec = tpl.get("spec", {})
            tlabels = (tpl.get("metadata") or {}).get("labels", {})
            sel = (spec.get("selector") or {}).get("matchLabels") or tlabels
            mypods = [p for p in pods if labels_match(sel, p["metadata"].get("labels", {}))
                      and _owned_by(p, kind, name)]
            comps.append(_component(kind, name, w, spec, pspec, tlabels, sel, mypods, where(kind, name), live))
    # bare pods (no owner), e.g. pvc-loader
    for p in pods:
        if not p["metadata"].get("ownerReferences"):
            name = p["metadata"]["name"]
            comps.append(_component("Pod", name, p, {}, p.get("spec", {}), p["metadata"].get("labels", {}),
                                    p["metadata"].get("labels", {}), [p], where("Pod", name), live))
    # Jobs that finished long ago can be gone from the cluster; keep them from YAML.
    have = {c["name"] for c in comps}
    for m in manifests:
        n = (m.get("metadata") or {}).get("name")
        if m["kind"] in wl_kinds and n not in have:
            spec = m.get("spec", {}); tpl = spec.get("template", {}); pspec = tpl.get("spec", {})
            tl = (tpl.get("metadata") or {}).get("labels", {})
            c = _component(m["kind"], n, m, spec, pspec, tl, tl, [], where(m["kind"], n), False)
            c["notes"].append("Not found in the cluster right now (finished Job cleaned up, or not applied): shown from its YAML.")
            comps.append(c)

    # ── services ───────────────────────────────────────────────────────────
    services = []
    for s in from_both("Service"):
        name = s["metadata"]["name"]
        sp = s.get("spec", {})
        sel = sp.get("selector") or {}
        targets = [c["id"] for c in comps if labels_match(sel, c["podLabels"])]
        ep = endpoints.get(name)
        ready, notready = [], []
        if ep:
            for sub in ep.get("subsets", []) or []:
                for a in sub.get("addresses", []) or []:
                    ready.append({"ip": a.get("ip"), "pod": (a.get("targetRef") or {}).get("name")})
                for a in sub.get("notReadyAddresses", []) or []:
                    notready.append({"ip": a.get("ip"), "pod": (a.get("targetRef") or {}).get("name")})
        svc = {
            "id": f"svc:{name}", "name": name, "type": sp.get("type", "ClusterIP"),
            "clusterIP": sp.get("clusterIP", "(assigned at apply time)") if live else "(assigned at apply time)",
            "dns": f"{name}.{ns}.svc.cluster.local",
            "ports": [{"name": p.get("name", ""), "port": p.get("port"), "targetPort": p.get("targetPort", p.get("port")),
                       "protocol": p.get("protocol", "TCP"), "nodePort": p.get("nodePort")} for p in sp.get("ports", [])],
            "selector": sel, "targets": targets, "ready": ready, "notReady": notready,
            "defined": where("Service", name),
        }
        services.append(svc)
        for t in targets:
            next(c for c in comps if c["id"] == t)["services"].append(svc["id"])

    # ── volumes ────────────────────────────────────────────────────────────
    pvcs = []
    for v in from_both("PersistentVolumeClaim"):
        name = v["metadata"]["name"]
        sp = v.get("spec", {})
        users = []
        for c in comps:
            for mnt in c["mounts"]:
                if mnt.get("pvc") == name:
                    users.append({"component": c["id"], "path": mnt["path"], "readOnly": mnt["readOnly"]})
        pvcs.append({
            "id": f"pvc:{name}", "name": name, "size": (sp.get("resources") or {}).get("requests", {}).get("storage", "?"),
            "access": sp.get("accessModes", []), "storageClass": sp.get("storageClassName", "(default)"),
            "phase": (v.get("status") or {}).get("phase", "not live") if live else "not live",
            "volume": sp.get("volumeName", ""), "users": users, "defined": where("PersistentVolumeClaim", name),
            "contents": PVC_CONTENTS.get(name, ""),
        })

    # ── configmaps / secrets (KEY NAMES only) ──────────────────────────────
    configs = []
    for kind in ("ConfigMap", "Secret"):
        for o in from_both(kind):
            name = o["metadata"]["name"]
            if name in ("kube-root-ca.crt",) or name.startswith("sh.helm"):
                continue
            keys = o.get("_keys") if kind == "Secret" else sorted((o.get("data") or {}).keys())
            if kind == "Secret" and not keys:
                keys = (manifest_by.get(("Secret", name)) or {}).get("_keys", [])
            used = [c["id"] for c in comps if name in c["configRefs"] or name in c["secretRefs"]]
            configs.append({"id": f"{'cm' if kind == 'ConfigMap' else 'secret'}:{name}", "kind": kind, "name": name,
                            "keys": keys or [], "usedBy": used, "defined": where(kind, name)})

    # ── RBAC / isolation context for the namespace ────────────────────────
    netpols = [n["metadata"]["name"] for n in from_both("NetworkPolicy")]
    roles = [{"name": r["metadata"]["name"], "rules": r.get("rules", [])} for r in from_both("Role")]
    bindings = [{"name": b["metadata"]["name"], "role": (b.get("roleRef") or {}).get("name"),
                 "subjects": [s.get("name") for s in b.get("subjects", []) or []]} for b in from_both("RoleBinding")]

    model = {
        "meta": _meta(cluster, ns, repo, live),
        "components": comps, "services": services, "pvcs": pvcs, "configs": configs,
        "networkPolicies": netpols, "roles": roles, "roleBindings": bindings,
        "controlPlane": _control_plane(cluster),
        "routes": {"nginx": parse_nginx(repo), "fastapi": parse_fastapi(repo)},
        "dags": parse_dags(repo),
    }
    model["externals"] = EXTERNALS
    for pv in pvcs:
        pv["tree"], pv["treeSource"] = pvc_tree(pv, {c["id"]: c for c in comps}, ns, live)
    model["links"] = discover_links(model, manifests, objs, live)
    for l in model["links"]:
        l["response"], l["oneWay"] = response_for(l)
    model["flows"] = build_flows(model)
    model["questions"] = QUESTIONS
    _layout(model)
    return model


# ── what's inside each volume ──────────────────────────────────────────────
# Live: `kubectl exec` into a running pod that mounts the PVC and list the files.
# Otherwise: the layout the code is expected to produce.
PVC_ANNOT = {
    "logs-pvc": {"pending_feedback": "← backend saves every /predict upload here",
                 "feedback_data": "← backend moves the image here on /feedback · retraining DAG reads it",
                 "feedback.jsonl": "← backend appends one line per feedback",
                 "retrain_needed.flag": "← backend writes it · retraining DAG reads + deletes it",
                 "backend.log": "← backend's log file",
                 "feedback_data_archive": "← retraining DAG archives used feedback here"},
    "data-pvc": {"raw": "← you loaded it (kubectl cp) · data pipeline reads it",
                 "processed": "← data pipeline writes · train.py reads · retraining adds feedback images",
                 "manifests": "← split task: train/val/test CSVs (filepath,label)",
                 "baseline_stats": "← baseline task writes · backend reads it for drift",
                 "seed_model": "← your .pt goes here · register job reads it (and saves the fallback here)"},
    "mlruns-pvc": {"models": "← model registry: versions + stage (Production/Archived) · backend reads",
                   "artifacts": "← the model itself + reports",
                   "pytorch_model": "← what the backend loads (mlflow.pytorch.load_model)",
                   "params": "← img_size, best_threshold, lr …",
                   "metrics": "← val_recall, val_f1 … per epoch",
                   "tags": "← model_name, is_fallback, weights_sha256 …",
                   ".trash": "← deleted runs"},
    "postgres-data": {"pgdata": "← Postgres's data directory (only the postgres pod writes it)"},
    "prometheus-data": {"wal": "← write-ahead log of incoming samples"},
}
PVC_EXPECTED = {
    "logs-pvc": ["backend.log", "feedback.jsonl", "retrain_needed.flag", "pending_feedback/img_<md5>_<timestamp>.jpg",
                 "feedback_data/benign/img_<md5>_<timestamp>.jpg", "feedback_data/malignant/img_<md5>_<timestamp>.jpg",
                 "feedback_data_archive/<YYYYmmdd_HHMMSS>/benign/"],
    "data-pvc": ["raw/dataset1/train/benign/<n>_0.jpg", "raw/dataset1/train/malignant/<n>_1.jpg",
                 "raw/dataset1/test/benign/<n>_0.jpg", "raw/dataset1/test/malignant/<n>_1.jpg",
                 "processed/v1/train/benign/", "processed/v1/train/malignant/", "processed/v1/val/", "processed/v1/test/",
                 "processed/v1/manifests/train_manifest.csv", "processed/v1/manifests/val_manifest.csv",
                 "processed/v1/manifests/test_manifest.csv", "baseline_stats/v1_stats.json",
                 "seed_model/efficientnet_b1.pt", "seed_model/efficientnet_b1_fallback.pt"],
    "mlruns-pvc": ["<experiment_id>/meta.yaml", "<experiment_id>/<run_id>/meta.yaml",
                   "<experiment_id>/<run_id>/params/img_size", "<experiment_id>/<run_id>/params/best_threshold",
                   "<experiment_id>/<run_id>/metrics/val_recall", "<experiment_id>/<run_id>/tags/model_name",
                   "<experiment_id>/<run_id>/tags/is_fallback",
                   "<experiment_id>/<run_id>/artifacts/pytorch_model/MLmodel",
                   "<experiment_id>/<run_id>/artifacts/pytorch_model/data/model.pth",
                   "models/melanoma_classifier/meta.yaml", "models/melanoma_classifier/version-1/meta.yaml", ".trash/"],
    "postgres-data": ["pgdata/base/", "pgdata/global/pg_filenode.map", "pgdata/pg_wal/", "pgdata/postgresql.conf", "pgdata/PG_VERSION"],
    "prometheus-data": ["wal/", "chunks_head/", "<block_ulid>/index", "<block_ulid>/chunks/"],
    "grafana-data": ["grafana.db", "plugins/", "csv/"],
}


def pvc_tree(pv: dict, comps: dict, ns: str, live: bool):
    """Return ([lines], source) describing the files in a PVC."""
    paths, source = None, ""
    if live:
        for u in pv["users"]:
            c = comps.get(u["component"])
            pod = next((p for p in (c or {}).get("pods", []) if p.get("phase") == "Running"), None)
            if not pod:
                continue
            # folders get a trailing "/", files don't (works with GNU find and busybox)
            out = kubectl("exec", "-n", ns, pod["name"], "-c", c["containers"][0]["name"], "--", "sh", "-c",
                          f"cd {u['path']} 2>/dev/null && (find . -mindepth 1 -maxdepth 6 -type d 2>/dev/null | sed 's|$|/|'; "
                          f"find . -mindepth 1 -maxdepth 6 ! -type d 2>/dev/null) | head -6000", quiet=True)
            if isinstance(out, str) and out.strip():
                paths = [l[2:] for l in out.splitlines() if l.startswith("./")]
                source = f"live: kubectl exec into {pod['name']} → ls {u['path']}"
                break
    if not paths:
        paths = PVC_EXPECTED.get(pv["name"], [])
        source = "expected layout (from the code), not a live listing"
    return _summarise(paths, PVC_ANNOT.get(pv["name"], {})), source


def _summarise(paths: list[str], annot: dict, max_lines: int = 70) -> list[dict]:
    """Folder tree with big folders collapsed: 'a.jpg, b.jpg … +160 more .jpg'."""
    tree: dict = {}
    for p in paths:
        node = tree
        parts = [x for x in p.split("/") if x]
        is_dir = p.endswith("/")
        for i, part in enumerate(parts):
            if i == len(parts) - 1 and not is_dir:
                node.setdefault(part, None)          # a file (unless already known as a folder)
            else:
                if not isinstance(node.get(part), dict):
                    node[part] = {}
                node = node[part]
    lines: list[dict] = []

    def walk(node: dict, depth: int):
        dirs = sorted(k for k, v in node.items() if isinstance(v, dict))
        files = sorted(k for k, v in node.items() if v is None)
        show_d = dirs if len(dirs) <= 6 else dirs[:3]
        for d in show_d:
            if len(lines) >= max_lines:
                return
            n_files = _count(node[d])
            lines.append({"d": depth, "t": "dir", "n": d + "/", "c": n_files, "a": annot.get(d, "")})
            walk(node[d], depth + 1)
        if len(dirs) > len(show_d):
            lines.append({"d": depth, "t": "more", "n": f"… +{len(dirs) - len(show_d)} more folders like these", "c": 0, "a": ""})
        if len(files) <= 5:
            for f in files:
                if len(lines) < max_lines:
                    lines.append({"d": depth, "t": "file", "n": f, "c": 0, "a": annot.get(f, "")})
        else:
            for f in files[:2]:
                lines.append({"d": depth, "t": "file", "n": f, "c": 0, "a": annot.get(f, "")})
            rest = files[2:]
            exts: dict = {}
            for f in rest:
                stem, _, e = f.rpartition(".")
                key = f".{e}" if stem and e.isalnum() and len(e) <= 4 else "files"
                exts[key] = exts.get(key, 0) + 1
            if len(exts) == 1 and "files" not in exts:
                label = f"… +{len(rest)} more {next(iter(exts))}"
            else:
                label = f"… +{len(rest)} more files"
            lines.append({"d": depth, "t": "more", "n": label, "c": 0, "a": ""})

    walk(tree, 0)
    if len(lines) >= max_lines:
        lines.append({"d": 0, "t": "more", "n": "… (truncated)", "c": 0, "a": ""})
    return lines


def _count(node) -> int:
    if node is None:
        return 1
    return sum(_count(v) for v in node.values())


PVC_CONTENTS = {
    "data-pvc": "raw/dataset1/{train,test}/{benign,malignant} · processed/v1 (+ manifests/*.csv) · baseline_stats/v1_stats.json · seed_model/",
    "logs-pvc": "backend.log · pending_feedback/ (every upload) · feedback_data/{benign,malignant}/ · feedback.jsonl · retrain_needed.flag · feedback_data_archive/",
    "mlruns-pvc": "MLflow file store: <experiment>/<run>/{params,metrics,tags,artifacts} · models/melanoma_classifier/version-N (stage: Production/Archived)",
    "postgres-data": "pgdata/: Airflow's metadata tables (dag_run, task_instance, job, ab_user, session ...)",
    "prometheus-data": "Prometheus TSDB blocks (7-day retention)",
    "grafana-data": "grafana.db: users, preferences, UI-edited dashboards",
}

EXTERNALS = [
    {"id": "ext:internet", "name": "User's browser", "kind": "External", "group": "external",
     "what": "Anyone opening the public URL. Speaks HTTPS to the tunnel edge only; never reaches your laptop's IP."},
    {"id": "ext:edge", "name": "Tunnel edge", "kind": "External", "group": "external",
     "what": "ngrok's cloud (or Cloudflare). Terminates TLS and forwards each request down the tunnel your agent pod opened outbound."},
    {"id": "ext:you", "name": "You (kubectl)", "kind": "External", "group": "external",
     "what": "Your terminal: kubectl talks to the API server (apply, logs, exec, cp); kubectl port-forward relays a local port to a Service."},
]


def _owned_by(pod, kind, name) -> bool:
    for o in pod["metadata"].get("ownerReferences", []) or []:
        if kind == "Deployment" and o["kind"] == "ReplicaSet" and o["name"].rsplit("-", 1)[0] == name:
            return True
        if o["kind"] == kind and o["name"] == name:
            return True
    return False


def _component(kind, name, obj, spec, pspec, tlabels, sel, mypods, defined, live) -> dict:
    containers = []
    mounts, cfg_refs, sec_refs = [], set(), set()
    vols = {v["name"]: v for v in pspec.get("volumes", []) or []}
    for c in pspec.get("containers", []) or []:
        envs = []
        for e in c.get("env", []) or []:
            vf = e.get("valueFrom") or {}
            if "secretKeyRef" in vf:
                sec_refs.add(vf["secretKeyRef"]["name"]); envs.append({"name": e["name"], "from": f"secret {vf['secretKeyRef']['name']}"})
            elif "configMapKeyRef" in vf:
                cfg_refs.add(vf["configMapKeyRef"]["name"]); envs.append({"name": e["name"], "from": f"configmap {vf['configMapKeyRef']['name']}"})
            else:
                envs.append({"name": e["name"], "value": str(e.get("value", ""))[:120]})
        env_from = []
        for ef in c.get("envFrom", []) or []:
            if "configMapRef" in ef:
                cfg_refs.add(ef["configMapRef"]["name"]); env_from.append(f"configmap {ef['configMapRef']['name']}")
            if "secretRef" in ef:
                sec_refs.add(ef["secretRef"]["name"]); env_from.append(f"secret {ef['secretRef']['name']}")
        for m in c.get("volumeMounts", []) or []:
            v = vols.get(m["name"], {})
            entry = {"name": m["name"], "path": m["mountPath"], "readOnly": bool(m.get("readOnly"))}
            if "persistentVolumeClaim" in v:
                entry["pvc"] = v["persistentVolumeClaim"]["claimName"]
            elif "configMap" in v:
                entry["configMap"] = v["configMap"]["name"]; cfg_refs.add(v["configMap"]["name"])
            elif "secret" in v:
                entry["secret"] = v["secret"]["secretName"]; sec_refs.add(v["secret"]["secretName"])
            elif "emptyDir" in v:
                entry["emptyDir"] = True
            if not m["mountPath"].startswith("/var/run/secrets/kubernetes.io"):
                mounts.append(entry)
        probes = {}
        for pk in ("startupProbe", "readinessProbe", "livenessProbe"):
            p = c.get(pk)
            if p:
                how = ("GET " + p["httpGet"].get("path", "/") + f" :{p['httpGet'].get('port')}") if "httpGet" in p else \
                      ("exec " + " ".join(p["exec"]["command"])) if "exec" in p else \
                      (f"tcp :{p['tcpSocket'].get('port')}" if "tcpSocket" in p else "?")
                probes[pk.replace("Probe", "")] = {"how": how, "period": p.get("periodSeconds", 10),
                                                   "timeout": p.get("timeoutSeconds", 1), "failureThreshold": p.get("failureThreshold", 3),
                                                   "initialDelay": p.get("initialDelaySeconds", 0)}
        containers.append({
            "name": c["name"], "image": c.get("image", ""), "pullPolicy": c.get("imagePullPolicy", ""),
            "command": c.get("command", []), "args": c.get("args", []),
            "ports": [{"port": p.get("containerPort"), "protocol": p.get("protocol", "TCP"), "name": p.get("name", "")} for p in c.get("ports", []) or []],
            "env": envs, "envFrom": env_from, "probes": probes,
            "resources": c.get("resources", {}), "securityContext": c.get("securityContext", {}),
        })
    pod_list = []
    for p in mypods:
        st = p.get("status", {})
        cs = (st.get("containerStatuses") or [{}])
        pod_list.append({"name": p["metadata"]["name"], "ip": st.get("podIP", ""), "phase": st.get("phase", ""),
                         "ready": all(x.get("ready") for x in cs if x), "restarts": sum(x.get("restartCount", 0) for x in cs if x),
                         "node": (p.get("spec") or {}).get("nodeName", ""), "started": st.get("startTime", "")})
    status = obj.get("status", {}) or {}
    return {
        "id": name if kind != "Pod" else name, "name": name, "kind": kind, "group": group_of(name),
        "what": DESCRIBE.get(name, ""),
        "replicas": spec.get("replicas", 1 if kind != "Job" else None),
        "readyReplicas": status.get("readyReplicas", 0) if live else None,
        "jobStatus": ({"succeeded": status.get("succeeded", 0), "failed": status.get("failed", 0)} if kind == "Job" and live else None),
        "strategy": (spec.get("strategy") or {}).get("type", "") or (spec.get("updateStrategy") or {}).get("type", ""),
        "rolling": (spec.get("strategy") or {}).get("rollingUpdate"),
        "podLabels": tlabels, "selector": sel,
        "serviceAccount": pspec.get("serviceAccountName", "default"),
        "automountToken": pspec.get("automountServiceAccountToken", True),
        "podSecurity": pspec.get("securityContext", {}), "hostNetwork": pspec.get("hostNetwork", False),
        "restartPolicy": pspec.get("restartPolicy", "Always"),
        "containers": containers, "mounts": mounts,
        "configRefs": sorted(cfg_refs), "secretRefs": sorted(sec_refs),
        "pods": pod_list, "services": [], "defined": defined, "notes": [],
    }


def _meta(cluster, ns, repo, live) -> dict:
    node = {}
    if cluster and cluster.get("nodes"):
        n = cluster["nodes"][0]
        ni = n["status"].get("nodeInfo", {})
        node = {"name": n["metadata"]["name"], "cpu": n["status"].get("capacity", {}).get("cpu"),
                "memory": n["status"].get("capacity", {}).get("memory"),
                "allocatableMemory": n["status"].get("allocatable", {}).get("memory"),
                "runtime": ni.get("containerRuntimeVersion"), "os": ni.get("osImage"),
                "kubelet": ni.get("kubeletVersion"),
                "ip": next((a["address"] for a in n["status"].get("addresses", []) if a["type"] == "InternalIP"), ""),
                "podCIDR": n["spec"].get("podCIDR", "")}
    commit = ""
    try:
        commit = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%h %s"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    return {"generated": dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"), "namespace": ns,
            "live": live, "context": (cluster or {}).get("context", ""), "k8sVersion": (cluster or {}).get("version", ""),
            "node": node, "repoCommit": commit}


def _control_plane(cluster) -> list[dict]:
    base = [
        ("kube-apiserver", "Front door. kubectl, controllers, kubelet and Prometheus discovery all talk to it; it alone reads/writes etcd."),
        ("etcd", "Key-value store of every object (your YAML = desired state, plus status). Images are NOT stored here."),
        ("kube-controller-manager", "Reconcile loops: Deployment → ReplicaSet → Pods, Job completion, Endpoints (which pod IPs are Ready behind each Service)."),
        ("kube-scheduler", "Binds each new pod to a node after checking its resource requests fit."),
        ("kubelet", "Node agent (not a pod): pulls images, mounts volumes, starts containers through the runtime, runs every probe, restarts crashed containers."),
        ("kube-proxy", "Writes iptables rules so ClusterIP:port → one Ready pod IP:targetPort."),
        ("coredns", "DNS: backend → backend.<ns>.svc.cluster.local → ClusterIP."),
        ("storage-provisioner", "minikube's hostPath provisioner: each PVC becomes a folder on the node."),
    ]
    found = {}
    for p in (cluster or {}).get("system_pods", []):
        n = p["metadata"]["name"]
        for key, _ in base:
            if n.startswith(key):
                found.setdefault(key, []).append({"pod": n, "ip": p["status"].get("podIP", ""),
                                                  "image": p["spec"]["containers"][0].get("image", "")})
    return [{"id": f"cp:{k}", "name": k, "what": w, "pods": found.get(k, [])} for k, w in base]


# ════════════════════════════════════════════════════════════════════════════
# 3. Gates (links): who talks to whom, how, and what travels
# ════════════════════════════════════════════════════════════════════════════

def discover_links(model, manifests, objs, live) -> list[dict]:
    comps = {c["id"]: c for c in model["components"]}
    svcs = {s["name"]: s for s in model["services"]}
    links: list[dict] = []

    def add(**kw):
        kw.setdefault("evidence", [])
        kw.setdefault("hops", [])
        kw["id"] = f"L{len(links) + 1}"
        links.append(kw)
        return kw

    def comp_named(sub):
        return next((c for c in model["components"] if sub in c["name"]), None)

    def svc_hops(src_name, svc, port, target_port=None):
        tgt = [comps[t] for t in svc["targets"]]
        ips = ", ".join(f"{e['pod'] or '?'} ({e['ip']})" for e in svc["ready"]) or \
              ", ".join(f"{p['name']} ({p['ip']})" for c in tgt for p in c["pods"]) or "the Ready pods"
        return [
            {"at": src_name, "text": f"opens a connection to the name `{svc['name']}:{port}`"},
            {"at": "CoreDNS", "text": f"{svc['dns']} → ClusterIP {svc['clusterIP']}"},
            {"at": "kube-proxy (iptables)", "text": f"ClusterIP:{port} is DNAT-ed to ONE Ready pod :{target_port or port}"},
            {"at": "pod", "text": f"candidates right now: {ips}"},
        ]

    # ── a) internet → edge → tunnel agent ─────────────────────────────────
    add(src="ext:internet", dst="ext:edge", kind="user", proto="HTTPS", port=443, label="HTTPS",
        payload=PAYLOAD["internet"], evidence=["Public URL from your ngrok / Cloudflare account"])
    for c in model["components"]:
        img = " ".join(x["image"] for x in c["containers"])
        if "ngrok" in img or "cloudflared" in img:
            add(src="ext:edge", dst=c["id"], kind="user", proto="tunnel (outbound TLS)", port=None, label="tunnel",
                payload=PAYLOAD["tunnel"], evidence=[f"{c['defined']['file']}: image {img}"],
                hops=[{"at": c["name"], "text": "pod dialled OUT to the provider at startup and keeps the connection open"},
                      {"at": "edge", "text": "each public request is pushed back down that same connection"}])

    # ── b) references to Service names found in args / env / config / secrets
    def texts_for(c):
        out = []
        for ct in c["containers"]:
            out += [("args", " ".join(map(str, ct["args"] + ct["command"])))]
            out += [(f"env {e['name']}", e.get("value", "")) for e in ct["env"]]
        for kind, refs in (("ConfigMap", c["configRefs"]), ("Secret", c["secretRefs"])):
            for r in refs:
                o = next((x for x in objs.get(kind, []) if x["metadata"]["name"] == r), None) or \
                    next((m for m in manifests if m["kind"] == kind and m["metadata"]["name"] == r), None)
                if not o:
                    continue
                if kind == "Secret":
                    out.append((f"secret {r}", o.get("_text", "")))
                else:
                    for k, v in (o.get("data") or {}).items():
                        out.append((f"configmap {r} → {k}", f"{k}={v}"))
        return out

    seen = set()
    for c in model["components"]:
        corpus = texts_for(c)
        for where_, text in corpus:
            for sname, svc in svcs.items():
                if svc["targets"] and c["id"] in svc["targets"]:
                    continue  # don't link to yourself
                pat = rf"(?<![\w.-]){re.escape(sname)}(?:\.[\w.-]+)?:(\d+)"
                for m in re.finditer(pat, text):
                    _ref_link(add, c, svc, int(m.group(1)), where_, seen, svc_hops)
                # KEY_HOST=<svc> + KEY_PORT=<n> style (Airflow StatsD)
                for m in re.finditer(rf"(\w+)_HOST=({re.escape(sname)})\b", text):
                    port_m = None
                    for w2, t2 in corpus:
                        port_m = re.search(rf"{m.group(1)}_PORT=(\d+)", t2) or port_m
                    if port_m:
                        _ref_link(add, c, svc, int(port_m.group(1)), where_, seen, svc_hops)

    # ── c) nginx reverse proxy routes ──────────────────────────────────────
    fe = next((c for c in model["components"] if "frontend" in c["name"]), None)
    routes = model["routes"]["nginx"]
    if fe and routes:
        by_host = {}
        for r in routes:
            by_host.setdefault((r["host"], r["port"]), []).append(r)
        for (host, port), rs in by_host.items():
            svc = svcs.get(host)
            if not svc:
                continue
            for t in svc["targets"]:
                paths = [r["path"] for r in rs]
                api = {r["path"]: r for r in model["routes"]["fastapi"]}
                add(src=fe["id"], dst=t, via=svc["id"], kind="user", proto="HTTP/1.1", port=port,
                    label=f"proxy_pass {' '.join(paths)}",
                    routes=[{"path": p, "method": (api.get(p) or {}).get("method", "?"),
                             "handler": f"{(api.get(p) or {}).get('file', '?')}:{(api.get(p) or {}).get('line', '?')}"} for p in paths],
                    payload=PAYLOAD["nginx_backend"],
                    evidence=[f"frontend/nginx.conf:{r['line']}  location {r['path']} → proxy_pass http://{host}:{port}" for r in rs]
                             + [f"{svc['defined']['file']}: Service {host}"],
                    hops=[{"at": fe["name"] + " (nginx)", "text": f"matches location, resolves `{host}` ONCE at startup"}] + svc_hops("nginx", svc, port)[1:])

    # ── d) Prometheus scrape config (static + kubernetes_sd) ──────────────
    prom = comp_named("prometheus")
    if prom:
        cfg_text = ""
        for r in prom["configRefs"]:
            o = next((x for x in objs.get("ConfigMap", []) if x["metadata"]["name"] == r), None) or \
                next((m for m in manifests if m["kind"] == "ConfigMap" and m["metadata"]["name"] == r), None)
            if o:
                cfg_text += "\n".join((o.get("data") or {}).values())
        pcfg = yaml.safe_load(cfg_text) if (yaml and cfg_text) else None
        for job in (pcfg or {}).get("scrape_configs", []) or []:
            for sc in job.get("static_configs", []) or []:
                for tgt in sc.get("targets", []):
                    host, _, port = tgt.partition(":")
                    if host in svcs:
                        _ref_link(add, prom, svcs[host], int(port or 80), f"prometheus.yml job {job['job_name']}", seen, svc_hops, scrape=True)
            if job.get("kubernetes_sd_configs"):
                keep = {}
                for rc in job.get("relabel_configs", []) or []:
                    if rc.get("action") == "keep" and rc.get("source_labels"):
                        keep[rc["source_labels"][0]] = str(rc.get("regex"))
                lab = keep.get("__meta_kubernetes_pod_label_app")
                port = keep.get("__meta_kubernetes_pod_container_port_number", "?")
                for c in model["components"]:
                    if lab and c["podLabels"].get("app") == lab:
                        ips = ", ".join(f"{p['name']} ({p['ip']})" for p in c["pods"]) or "every pod with that label"
                        add(src=prom["id"], dst=c["id"], via=None, kind="metric", proto="HTTP GET (pull)", port=int(port) if port.isdigit() else port,
                            label=f"scrape each pod :{port}{job.get('metrics_path', '/metrics')}", payload=PAYLOAD["scrape_backend"],
                            evidence=[f"{prom['defined']['file']}: job '{job['job_name']}' kubernetes_sd_configs role=pod, keep app={lab}, port {port}",
                                      "RBAC: ServiceAccount prometheus + Role allowing get/list/watch pods"],
                            hops=[{"at": "prometheus", "text": "asks the API server: list pods in the namespace (needs the Role)"},
                                  {"at": "relabel", "text": f"keeps pods labelled app={lab}, container port {port}"},
                                  {"at": "pod", "text": f"GET http://<podIP>:{port}/metrics directly, NO Service: {ips}"}])

    # ── e) volumes ─────────────────────────────────────────────────────────
    for p in model["pvcs"]:
        for u in p["users"]:
            c = comps[u["component"]]
            add(src=c["id"], dst=p["id"], kind="file", proto="filesystem (bind mount)", port=None,
                label=f"{'reads' if u['readOnly'] else 'reads/writes'} {u['path']}",
                mode="ro" if u["readOnly"] else "rw", path=u["path"],
                payload=FILE_PAYLOAD.get((c["name"], p["name"]), f"Files under {u['path']}: {p['contents']}"),
                evidence=[f"{c['defined']['file']}: volumeMount {u['path']} ← claimName {p['name']}{' (readOnly)' if u['readOnly'] else ''}"],
                hops=[{"at": c["name"], "text": f"open('{u['path']}/...') like a local folder"},
                      {"at": "kubelet", "text": f"bind-mounted the PVC's folder into the container at {u['path']}"},
                      {"at": "storage-provisioner", "text": f"{p['name']} is a hostPath folder on the node ({p['storageClass']}, {'/'.join(p['access'])})"}])

    # ── f) config injection ────────────────────────────────────────────────
    for cfg in model["configs"]:
        for u in cfg["usedBy"]:
            c = comps[u]
            how = "env vars (envFrom / valueFrom)"
            for mnt in c["mounts"]:
                if mnt.get("configMap") == cfg["name"] or mnt.get("secret") == cfg["name"]:
                    how = f"files mounted at {mnt['path']}"
            add(src=cfg["id"], dst=c["id"], kind="config", proto=how, port=None, label=cfg["kind"].lower(),
                payload=(f"Keys: {', '.join(cfg['keys']) or '(none)'}." +
                         (" Values are only base64-encoded in etcd; RBAC protects them." if cfg["kind"] == "Secret" else "")),
                evidence=[f"{c['defined']['file']}: references {cfg['kind']} {cfg['name']}"],
                hops=[{"at": "kubelet", "text": f"reads {cfg['kind']} {cfg['name']} from the API server (only for pods on its own node)"},
                      {"at": c["name"], "text": f"gets it as {how} when the container starts"}])

    # ── g) kubelet probes (the health inspector) ───────────────────────────
    for c in model["components"]:
        for ct in c["containers"]:
            for pk, p in ct["probes"].items():
                add(src="cp:kubelet", dst=c["id"], kind="probe", proto="probe", port=None,
                    label=f"{pk}: {p['how']}",
                    payload=PROBE_TEXT[pk].format(**p),
                    evidence=[f"{c['defined']['file']}: {pk}Probe {p['how']} every {p['period']}s, timeout {p['timeout']}s, failureThreshold {p['failureThreshold']}"],
                    hops=[{"at": "kubelet", "text": "runs ON the node, calls the pod IP directly (never the Service)"},
                          {"at": c["name"], "text": "answers; only the status code / exit code matters"}])

    # ── h) known behaviour that isn't in YAML (stated as such) ─────────────
    web, sch = comp_named("airflow-webserver"), comp_named("airflow-schedular")
    if web and sch:
        add(src=web["id"], dst=sch["id"], kind="http", proto="HTTP", port=8793, label="fetch task logs :8793",
            payload="When you open a task's log in the UI, the webserver GETs it from the scheduler's built-in log server (tasks ran there).",
            evidence=["Airflow default worker_log_server_port=8793; airflow-config sets HOSTNAME_CALLABLE=get_host_ip_address so the hostname is the pod IP",
                      f"{sch['defined']['file']}: containerPort 8793"],
            hops=[{"at": "airflow-webserver", "text": "reads the task's hostname (= scheduler pod IP) from Postgres"},
                  {"at": "airflow-schedular", "text": "GET http://<podIP>:8793/log/... (no Service involved)"}])
    for s in model["services"]:
        if s["name"] in PORT_FORWARD:
            for t in s["targets"]:
                add(src="ext:you", dst=t, via=s["id"], kind="control", proto="kubectl port-forward", port=s["ports"][0]["port"],
                    label=f"port-forward {PORT_FORWARD[s['name']]}",
                    payload=f"Your browser at http://localhost:{PORT_FORWARD[s['name']]} → tunnelled through the API server and kubelet to one pod behind Service {s['name']}.",
                    evidence=[f"kubectl port-forward svc/{s['name']} {PORT_FORWARD[s['name']]}:{s['ports'][0]['port']} -n {model['meta']['namespace']}"],
                    hops=[{"at": "you", "text": f"localhost:{PORT_FORWARD[s['name']]}"},
                          {"at": "kube-apiserver", "text": "authenticates you, opens a stream to the kubelet"},
                          {"at": "kubelet", "text": f"connects to the pod's port {s['ports'][0]['targetPort']}"}])
    add(src="ext:you", dst="cp:kube-apiserver", kind="control", proto="HTTPS (kubeconfig cert)", port=8443,
        label="kubectl apply / logs / exec / cp", payload="Every kubectl command is a REST call to the API server, authenticated with the client certificate in your kubeconfig.",
        evidence=[f"kubectl context: {model['meta'].get('context') or 'minikube'}"],
        hops=[{"at": "you", "text": "kubectl reads ~/.kube/config"}, {"at": "kube-apiserver", "text": "validates, stores in etcd"}])
    return links


PORT_FORWARD = {"frontend": 8081, "airflow-webserver": 8080, "mlflow": 5000, "grafana": 3000, "prometheus": 9090, "ngrok": 4040}


def _ref_link(add, c, svc, port, where_, seen, svc_hops, scrape=False):
    sp = next((p for p in svc["ports"] if p["port"] == port), svc["ports"][0] if svc["ports"] else {"port": port, "protocol": "TCP", "targetPort": port})
    for t in svc["targets"]:
        key = (c["id"], t, port)
        if key in seen:
            # same connection found in another place: just add the evidence
            continue
        seen.add(key)
        udp = sp.get("protocol") == "UDP"
        scrape = scrape or "prometheus" in c["name"]          # Prometheus only ever pulls
        agent = any(x in c["name"] for x in ("ngrok", "cloudflared"))
        kind = "user" if agent else "metric" if (scrape or udp) else ("db" if port == 5432 else "http")
        proto = "UDP (StatsD, fire-and-forget)" if udp else ("PostgreSQL wire protocol (TCP)" if port == 5432 else
                ("HTTP GET (pull)" if scrape else "HTTP"))
        payload = PAYLOAD.get((c["name"], svc["name"])) or PAYLOAD.get(("*", svc["name"])) or f"{proto} to {svc['name']}:{port}"
        add(src=c["id"], dst=t, via=svc["id"], kind=kind, proto=proto, port=port,
            label=(f"scrape :{port}" if scrape else f"http://{svc['name']}:{port}" if agent else f"{svc['name']}:{port}" + (" UDP" if udp else "")),
            payload=payload, evidence=[f"{c['defined']['file']} → {where_} mentions {svc['name']}:{port}", f"{svc['defined']['file']}: Service {svc['name']}"],
            hops=svc_hops(c["name"], svc, port, sp.get("targetPort")))


PAYLOAD = {
    "internet": "Plain HTTPS requests from anywhere on the internet: the UI files, POST /predict with an image, POST /feedback with JSON. TLS ends at the edge.",
    "tunnel": "The same HTTP requests, carried inside the long-lived outbound connection the agent pod opened. Nothing listens on your laptop.",
    ("ngrok", "frontend"): "Every public request, unchanged: GET / (UI files) and the API calls the page makes (POST /predict multipart image, POST /feedback JSON, GET /ready, /health, /metrics).",
    ("cloudflared", "frontend"): "Every public request for lesionscan.com, unchanged.",
    "nginx_backend": "Only API calls (static files are served by nginx itself):\n"
                     "POST /predict  multipart/form-data file=<jpeg/png>  →  200 {\"image_id\":\"img_9a4b..._1793...jpg\",\"label\":\"malignant\",\"confidence\":0.62,\"malignant_prob\":0.62,\"threshold_used\":0.1,\"is_fallback\":true,...}\n"
                     "POST /feedback {\"image_id\":...,\"predicted_label\":\"malignant\",\"true_label\":\"benign\"}  →  {\"received\":true}\n"
                     "GET /ready → 200 {model_loaded, model_name, model_version, is_fallback} or 503 · GET /health → {\"status\":\"ok\"} · GET /metrics → Prometheus text",
    ("*", "postgres"): "SQLAlchemy over psycopg2: INSERT/UPDATE dag_run, task_instance, job (scheduler heartbeats every 30 s), SELECT for the UI, user/session rows for logins. Connection string comes from the Secret (user/password never shown here).",
    ("*", "statsd-exporter"): "StatsD text lines over UDP, e.g.  airflow.scheduler_heartbeat:1|c   airflow.dag_processing.total_parse_time:2.3|g   airflow.ti.finish.<dag>.<task>.success:1|c. No reply: if the exporter is down, Airflow doesn't notice.",
    ("grafana", "prometheus"): "PromQL over HTTP, e.g. GET /api/v1/query_range?query=sum(rate(melanoma_prediction_total[5m]))*60 → JSON time series for the panels.",
    ("prometheus", "statsd-exporter"): "GET /metrics every 15 s → text exposition, e.g. airflow_scheduler_heartbeat 4321",
    "scrape_backend": "GET /metrics every 15 s on EACH backend pod → text exposition, e.g.\nmelanoma_prediction_total{label=\"malignant\"} 12\nmelanoma_request_latency_seconds_bucket{endpoint=\"/predict\",le=\"0.5\"} 40\nmelanoma_model_is_fallback 1",
}

FILE_PAYLOAD = {
    ("backend-deployment", "logs-pvc"): "WRITES: pending_feedback/img_<md5>_<ts>.jpg on every /predict · moves it to feedback_data/<true_label>/ on /feedback · appends feedback.jsonl · writes retrain_needed.flag (error > 10% after ≥ 10 feedbacks, or drift > 0.20) · backend.log",
    ("backend-deployment", "mlruns-pvc"): "READS (ro): models/melanoma_classifier/*/meta.yaml to find the Production version, then the run's params (img_size, best_threshold), tags (model_name, is_fallback) and artifacts/pytorch_model. At startup and on every /ready.",
    ("backend-deployment", "data-pvc"): "READS (ro): baseline_stats/v1_stats.json once at startup (pixel histogram for drift detection).",
    ("airflow-schedular", "logs-pvc"): "Retraining DAG: check_trigger READS retrain_needed.flag · prepare_feedback_data READS feedback_data/* · cleanup MOVES them to feedback_data_archive/ and DELETES the flag.",
    ("airflow-schedular", "data-pvc"): "Data pipeline DAG: reads raw/dataset1, writes processed/v1 + manifests + baseline_stats. Retraining DAG: copies feedback images into processed/v1/train and appends train_manifest.csv.",
    ("airflow-schedular", "mlruns-pvc"): "train.py (run inside this pod) WRITES runs: params, per-epoch metrics, artifacts, the model; registers a new version and promotes it to Production if val_recall beats the current one.",
    ("register-b1-model", "data-pvc"): "READS seed_model/efficientnet_b1.pt (missing → builds the fallback) and the val/test manifests to tune the threshold; WRITES seed_model/efficientnet_b1_fallback.pt.",
    ("register-b1-model", "mlruns-pvc"): "WRITES run + registered model version, then moves it to stage Production.",
    ("mlflow", "mlruns-pvc"): "READS everything to render experiments, runs and the registry in the UI.",
    ("postgres", "postgres-data"): "PGDATA: Postgres's own data files. Only this one pod may write them (Recreate strategy).",
}

PROBE_TEXT = {
    "startup": "Startup probe: {how} every {period}s, up to {failureThreshold} failures. Until it passes, liveness/readiness are paused, so a slow boot isn't killed.",
    "readiness": "Readiness probe: {how} every {period}s (timeout {timeout}s). Failing = pod removed from its Service's endpoints (gets no traffic) but NOT restarted.",
    "liveness": "Liveness probe: {how} every {period}s (timeout {timeout}s, {failureThreshold} failures). Failing = kubelet restarts the container.",
}


def response_for(l: dict):
    """What comes BACK through a gate (and whether anything does at all)."""
    src, dst, kind, proto = l["src"], l["dst"], l["kind"], l.get("proto", "")
    if "UDP" in proto:
        return ("Nothing comes back. UDP is fire-and-forget: Airflow sends the packet and moves on. "
                "If statsd-exporter is down, Airflow never notices (and is never slowed down)."), True
    if kind == "config":
        return "Nothing: it is injected once, when the container starts.", True
    if kind == "file":
        if l.get("mode") == "ro":
            return "The file contents flow INTO the pod (reads only; any write fails with EROFS).", False
        return "Reads return file contents; writes return success or an error (e.g. disk full). Same folder for every pod that mounts it.", False
    if kind == "probe":
        return ("HTTP 200-399 (or exit code 0) = success. Anything else, or no answer within the timeout, counts as a failure; "
                "enough failures in a row → readiness: pod removed from the Service / liveness: container restarted."), False
    if src == "ext:internet":
        return "HTTPS responses: the UI files (HTML/JS/CSS) and the backend's JSON, encrypted from the edge to the browser.", False
    if src == "ext:edge":
        return "The response goes back up the SAME outbound connection the agent opened, then out to the browser.", False
    if "ngrok" in src or "cloudflared" in src:
        return "nginx's answer: static files for page loads, or the backend's JSON for API calls (passed through unchanged).", False
    if "frontend" in src and "backend" in dst:
        return ("/predict → 200 {\"image_id\":\"img_9a4b..._1793...jpg\",\"label\":\"malignant\",\"confidence\":0.62,\"malignant_prob\":0.62,\"threshold_used\":0.1,\"is_fallback\":true,\"recommendation\":\"...\"} "
                "| 400 bad image | 503 model not loaded\n/feedback → 200 {\"received\":true,\"message\":\"Feedback recorded. Thank you.\"}\n"
                "/ready → 200 {model_loaded, model_name, model_version, is_fallback} or 503 · /health → {\"status\":\"ok\"} · /metrics → Prometheus text"), False
    if kind == "db":
        return "Result rows for SELECTs (dag runs, task states, users); OK / row counts for INSERT/UPDATE; errors such as 'too many clients' or a deadlock.", False
    if kind == "metric":
        return "200 OK with the metrics as plain text, e.g. melanoma_prediction_total{label=\"benign\"} 12. A failed or timed-out scrape marks the target DOWN.", False
    if "grafana" in src:
        return "JSON time series ({\"status\":\"success\",\"data\":{\"result\":[...]}}) that Grafana turns into panels.", False
    if "8793" in str(l.get("port")):
        return "The task's log text, which the UI shows under the task.", False
    if kind == "control":
        return "Whatever the UI answers (HTML, JSON), relayed back through the kubelet and API server to localhost.", False
    return "The callee's reply.", False


# ════════════════════════════════════════════════════════════════════════════
# 4. Flows (scenarios to play) and interview questions
# ════════════════════════════════════════════════════════════════════════════

def build_flows(model) -> list[dict]:
    L = model["links"]

    def find(src, dst, kind=None, contains=None):
        for l in L:
            s_ok = src in l["src"]
            d_ok = dst in l["dst"]
            if s_ok and d_ok and (kind is None or l["kind"] == kind) and (contains is None or contains in l["label"]):
                return l["id"]
        return None

    def step(text, link=None, node=None, reverse=False):
        return {"text": text, "link": link, "node": node, "reverse": reverse}

    flows = [
        {"id": "predict", "title": "A user clicks Analyse (POST /predict)", "steps": [
            step("The browser sends POST /predict with the image over HTTPS to the public URL.", find("ext:internet", "ext:edge")),
            step("The edge pushes it down the tunnel the agent pod opened outbound.", find("ext:edge", "ngrok") or find("ext:edge", "cloudflared")),
            step("The agent calls http://frontend:80. CoreDNS gives the ClusterIP; kube-proxy picks one of the nginx pods.", find("ngrok", "frontend") or find("cloudflared", "frontend")),
            step("nginx matches location /predict and proxy_passes to backend:8000, which kube-proxy sends to a READY backend pod.", find("frontend", "backend", "user")),
            step("The backend saves the upload to pending_feedback/ on logs-pvc (so feedback can find it later).", find("backend", "pvc:logs-pvc")),
            step("Inference happens in memory: resize to img_size (224), normalise, EfficientNet-B1 forward, softmax, compare with the tuned threshold, compute drift.", None, "backend-deployment"),
            step("The JSON answer (label, confidence, image_id, is_fallback) travels back the same way.", find("frontend", "backend", "user"), None, True),
            step("Later, Prometheus scrapes this pod's /metrics and sees the counters move.", find("prometheus", "backend")),
        ]},
        {"id": "loop", "title": "Feedback → retrain → hot reload (the MLOps loop)", "steps": [
            step("The user marks the true label: POST /feedback through nginx to a backend pod (maybe a different one than /predict).", find("frontend", "backend", "user")),
            step("The backend moves the image to feedback_data/<label>/ and appends feedback.jsonl on logs-pvc. If error > 10% after ≥ 10 feedbacks (or drift > 0.20) it writes retrain_needed.flag.", find("backend", "pvc:logs-pvc")),
            step("Every 30 minutes the retraining DAG runs in the scheduler pod. check_trigger reads the flag from the SAME folder (/opt/airflow/app_logs).", find("airflow-schedular", "pvc:logs-pvc")),
            step("Task states are written to Postgres the whole time.", find("airflow-schedular", "postgres", "db")),
            step("prepare_feedback_data copies the images into processed/v1/train and extends the manifest.", find("airflow-schedular", "pvc:data-pvc")),
            step("trigger_training runs python train.py inside the scheduler pod and logs the run to mlruns-pvc; if val_recall beats Production it promotes the new version.", find("airflow-schedular", "pvc:mlruns-pvc")),
            step("Airflow pushes task metrics over UDP to statsd-exporter.", find("airflow-schedular", "statsd")),
            step("Within 30 s, the kubelet's readiness probe calls /ready on each backend pod ...", find("cp:kubelet", "backend", "probe", "readiness")),
            step("... /ready runs check_and_reload(): the Production version changed, so it loads the new model from mlruns-pvc. No rebuild, no rollout.", find("backend", "pvc:mlruns-pvc")),
        ]},
        {"id": "poddies", "title": "A backend pod dies", "steps": [
            step("A backend pod crashes (or you kubectl delete it).", None, "backend-deployment"),
            step("Its readiness stops passing; the endpoints controller removes its IP from the backend Service.", find("cp:kubelet", "backend", "probe", "readiness")),
            step("kube-proxy updates iptables: nginx's next request goes to one of the remaining Ready pods. Users don't notice.", find("frontend", "backend", "user")),
            step("controller-manager sees 2/3 pods for the ReplicaSet and creates a new Pod object.", None, "cp:kube-controller-manager"),
            step("kube-scheduler binds it to the node; kubelet starts it and mounts the three PVCs.", None, "cp:kube-scheduler"),
            step("The new pod loads the model from mlruns-pvc at startup.", find("backend", "pvc:mlruns-pvc")),
            step("Readiness passes → its IP is added back to the Service. 3/3 again.", find("cp:kubelet", "backend", "probe", "readiness")),
        ]},
        {"id": "metrics", "title": "How a metric reaches Grafana", "steps": [
            step("Airflow pushes StatsD lines over UDP (fire-and-forget).", find("airflow-schedular", "statsd")),
            step("Prometheus pulls the converted metrics from statsd-exporter every 15 s.", find("prometheus", "statsd")),
            step("Prometheus also pulls /metrics from EACH backend pod, found through the API server (no Service).", find("prometheus", "backend")),
            step("Grafana's server runs PromQL against prometheus:9090 to draw the panels.", find("grafana", "prometheus")),
        ]},
        {"id": "trigger", "title": "You trigger a DAG in the Airflow UI", "steps": [
            step("kubectl port-forward relays localhost:8080 to an airflow-webserver pod.", find("ext:you", "airflow-webserver", "control")),
            step("Clicking Trigger only INSERTs a dag_run row in Postgres.", find("airflow-webserver", "postgres", "db")),
            step("The scheduler's loop sees the queued run and starts the tasks as subprocesses (LocalExecutor).", find("airflow-schedular", "postgres", "db")),
            step("The tasks read raw/ and write processed/ and baseline stats on data-pvc.", find("airflow-schedular", "pvc:data-pvc")),
            step("When you open a task log, the webserver fetches it from the scheduler pod on :8793.", find("airflow-webserver", "airflow-schedular")),
        ]},
        {"id": "bootstrap", "title": "First model gets into Production", "steps": [
            step("register-b1-model reads seed_model/ (your .pt, or nothing → fallback) and the val/test manifests.", find("register", "pvc:data-pvc")),
            step("It logs the run, registers version 1 and moves it to Production on mlruns-pvc.", find("register", "pvc:mlruns-pvc")),
            step("Backend pods were 0/1 (readiness 503). The next /ready call ...", find("cp:kubelet", "backend", "probe", "readiness")),
            step("... finds the Production version, loads it, and the pod becomes Ready.", find("backend", "pvc:mlruns-pvc")),
        ]},
        {"id": "apply", "title": "kubectl apply -f k8s/backend/deployment.yaml", "steps": [
            step("kubectl sends the Deployment + Service to the API server, which stores them in etcd.", find("ext:you", "cp:kube-apiserver")),
            step("controller-manager: Deployment → ReplicaSet → 3 Pod objects.", None, "cp:kube-controller-manager"),
            step("kube-scheduler binds each pod to the minikube node (checks requests fit).", None, "cp:kube-scheduler"),
            step("kubelet: image is local (IfNotPresent), mounts PVCs, injects config, starts the container.", find("backend", "pvc:logs-pvc")),
            step("Probes start; readiness 503 until a model loads.", find("cp:kubelet", "backend", "probe", "readiness")),
            step("Once Ready, the pod IP joins the backend Service; nginx can reach it.", find("frontend", "backend", "user")),
        ]},
    ]
    return flows


QUESTIONS = [
    {"q": "Walk me through what happens when a user clicks Analyse.", "show": {"flow": "predict"},
     "a": "Browser → HTTPS → tunnel edge → (outbound tunnel) → ngrok pod → Service frontend (CoreDNS + kube-proxy) → nginx pod → proxy_pass → Service backend → a READY FastAPI pod → saves the upload to logs-pvc → in-memory B1 inference → JSON back the same way."},
    {"q": "What's the difference between a Pod, a Deployment and a Service?", "show": {"room": "backend-deployment"},
     "a": "Pod = the running containers (here one uvicorn process). Deployment = 'keep 3 of these from backend-image:v1', via a ReplicaSet, and roll updates. Service = no code at all: a stable DNS name + virtual IP that selects pods by label and only includes Ready ones."},
    {"q": "Who actually receives traffic on port 8000?", "show": {"room": "backend-deployment"},
     "a": "Not the Service: it's an iptables rule. The FastAPI process inside one backend pod receives it. kube-proxy DNATs backend:8000 to <podIP>:8000 of a Ready pod. Prometheus and the kubelet skip the Service and hit pod IPs directly."},
    {"q": "Why does Prometheus scrape the pods and not the Service?", "show": {"flow": "metrics"},
     "a": "Each pod keeps its own counters. Through the Service each scrape would hit a random pod and the series would jump. kubernetes_sd_configs (role: pod) lists pods via the API server (hence the ServiceAccount + Role) and scrapes each one."},
    {"q": "How does a new model reach production without redeploying?", "show": {"flow": "loop"},
     "a": "train.py promotes a better version in the MLflow registry on mlruns-pvc. The readiness probe calls /ready every 30 s, which runs check_and_reload() and loads the new Production version into memory."},
    {"q": "What happens if a backend pod dies mid-traffic?", "show": {"flow": "poddies"},
     "a": "Its IP leaves the Service endpoints, kube-proxy routes to the other two, controller-manager creates a replacement, scheduler binds it, kubelet starts it, it loads the model, readiness passes, it rejoins. In-flight requests on the dead pod fail; new ones don't."},
    {"q": "Why does readiness return 503 when no model is loaded?", "show": {"flow": "bootstrap"},
     "a": "Probes only look at the status code. A 200 with model_loaded:false would put a useless pod behind the Service. 503 keeps it out until it can actually predict."},
    {"q": "How do the backend and Airflow share data?", "show": {"room": "pvc:logs-pvc"},
     "a": "Through PVCs mounted in both: logs-pvc is /app/logs in the backend and /opt/airflow/app_logs in the scheduler; mlruns-pvc is the model registry for both. They're ReadWriteOnce = one NODE, fine on 1-node minikube; multi-node needs RWX or S3/MinIO + an MLflow server."},
    {"q": "Why does the scheduler mount the volumes but the webserver doesn't?", "show": {"room": "airflow-schedular"},
     "a": "LocalExecutor: no worker pods, the scheduler forks every task (including train.py) inside its own pod. The webserver only shows the UI and writes dag_run rows to Postgres."},
    {"q": "What isolates one component from another here?", "show": {"room": "postgres"},
     "a": "Each pod has its own network, PID and mount namespaces and cgroup limits (requests/limits). Services are ClusterIP = unreachable from outside the cluster. Volumes are only visible where mounted, read-only where marked. BUT there are no NetworkPolicies, so any pod in the cluster could connect to postgres:5432: that's the gap to name."},
    {"q": "How do Airflow's metrics get to Grafana?", "show": {"flow": "metrics"},
     "a": "Push → bridge → pull: StatsD over UDP to statsd-exporter:8125, Prometheus scrapes :9102, Grafana runs PromQL against prometheus:9090."},
    {"q": "How is the app exposed to the internet?", "show": {"room": "ngrok"},
     "a": "A tunnel agent pod dials OUT to ngrok/Cloudflare and forwards requests to the frontend Service. No inbound port, works behind home NAT/CGNAT. LoadBalancer needs a cloud; NodePort on a laptop isn't internet-reachable."},
    {"q": "How are secrets handled?", "show": {"room": "secret:airflow-secret"},
     "a": "Kubernetes Secrets injected as env vars; they're base64 in etcd, not encrypted, so RBAC is the protection. The ngrok token is created on the command line and never committed. Production: External Secrets/Vault + etcd encryption."},
    {"q": "Why Recreate for Postgres but RollingUpdate for the backend?", "show": {"room": "postgres"},
     "a": "RollingUpdate keeps stateless pods available during updates. Anything owning an RWO volume or a lock (Postgres, scheduler, Prometheus, Grafana, MLflow) uses Recreate so two copies never write the same files."},
    {"q": "What does the kubelet do for your pods?", "show": {"room": "cp:kubelet"},
     "a": "Pulls the image (IfNotPresent → your local build), mounts PVCs, injects ConfigMaps/Secrets, starts containers, and runs every startup/readiness/liveness probe against the pod IP. Liveness failure → restart; readiness failure → out of the Service."},
    {"q": "What would you change for production?", "show": {"room": "pvc:mlruns-pvc"},
     "a": "S3/MinIO + MLflow tracking server instead of shared RWO files; KubernetesExecutor + GPU pool for training; HPA on the backend; Ingress/Gateway with TLS and auth; NetworkPolicies; immutable image tags in a registry with CI/CD; feedback counters in a shared store."},
]


# ════════════════════════════════════════════════════════════════════════════
# 5. Layout + write
# ════════════════════════════════════════════════════════════════════════════

def _layout(model):
    nodes = []
    for e in model["externals"]:
        nodes.append({"id": e["id"], "type": "external"})
    for c in model["components"]:
        nodes.append({"id": c["id"], "type": "component"})
    for p in model["pvcs"]:
        nodes.append({"id": p["id"], "type": "pvc"})
    auto_x, auto_y = 1730, 300
    pos = {}
    for n in nodes:
        key = n["id"] if n["id"].startswith(("ext:", "pvc:")) else n["id"]
        if key in LAYOUT:
            pos[n["id"]] = LAYOUT[key]
        else:
            pos[n["id"]] = (auto_x, auto_y); auto_y += 170
    for i, cp in enumerate(model["controlPlane"]):
        pos[cp["id"]] = (40 + i * 225, 1130)
    model["layout"] = {"positions": pos, "width": max(1960, auto_x + 260), "height": 1260}


def main():
    # Windows consoles default to cp1252; make the progress lines (•, ✅) printable everywhere
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="One-time interactive architecture snapshot")
    ap.add_argument("--namespace", default="melanoma-production")
    ap.add_argument("--repo", default=str(HERE.parents[1]))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-cluster", action="store_true")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    out = Path(a.out) if a.out else repo / "architecture.html"
    print(f"• Repo: {repo}")
    manifests = collect_manifests(repo)
    cluster = None if a.no_cluster else collect_cluster(a.namespace)
    model = build(cluster, manifests, repo, a.namespace)
    html = TEMPLATE.read_text(encoding="utf-8")
    data = json.dumps(model, ensure_ascii=False, default=str).replace("</", "<\\/")
    html = html.replace("/*__SNAPSHOT__*/null", data)
    out.write_text(html, encoding="utf-8")
    n_links = len(model["links"])
    print(f"• {len(model['components'])} components, {len(model['services'])} services, {len(model['pvcs'])} volumes, {n_links} gates")
    print(f"✅ Wrote {out}  ({out.stat().st_size // 1024} KB) - open it in your browser.")


if __name__ == "__main__":
    sys.exit(main())
