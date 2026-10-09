# Kubernetes manifests (optional, NOT tested)

**Status: written but not applied to any cluster.** kind is not installed on the build machine,
so these were not tested on kind. Only YAML validity was checked.

To try on kind:

```bash
docker build -t orbitdesk:latest .
kind create cluster --name orbitdesk
kind load docker-image orbitdesk:latest --name orbitdesk
# Redis + Chroma (not included here):
kubectl create deployment redis --image=redis:7-alpine && kubectl expose deployment redis --port 6379
kubectl create deployment chroma --image=chromadb/chroma:1.5.9 && kubectl expose deployment chroma --port 8000
kubectl create secret generic orbitdesk-secrets --from-env-file=.env
kubectl apply -f k8s/
kubectl port-forward svc/orbitdesk-api 8000:80   # http://localhost:8000/ui
```

Known gaps: SQLite on `emptyDir` is per pod (fine for a demo, not for multiple replicas: use Postgres);
the HPA needs metrics-server (`kind` does not ship it by default).
