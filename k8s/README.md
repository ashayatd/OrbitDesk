# Kubernetes manifests (optional, NOT tested)

**Status: written but not applied to any cluster.** kind is not installed on the build machine,
so these were not tested on kind. Only YAML validity was checked.

To try on kind:

```bash
docker build -t orbitdesk:latest .
kind create cluster --name orbitdesk
kind load docker-image orbitdesk:latest --name orbitdesk
# Redis + Postgres with pgvector (not included here; use a managed Postgres in real clusters):
kubectl create deployment redis --image=redis:7-alpine && kubectl expose deployment redis --port 6379
kubectl create deployment postgres --image=pgvector/pgvector:pg17 && kubectl set env deployment/postgres POSTGRES_USER=orbit POSTGRES_PASSWORD=orbit POSTGRES_DB=orbitdesk && kubectl expose deployment postgres --port 5432
kubectl create secret generic orbitdesk-secrets --from-env-file=.env   # include DATABASE_URL=postgresql+psycopg://orbit:orbit@postgres:5432/orbitdesk
kubectl apply -f k8s/
kubectl port-forward svc/orbitdesk-api 8000:80   # http://localhost:8000/ui
```

Known gaps: without DATABASE_URL each pod falls back to its own SQLite on `emptyDir` (not shared between replicas);
the HPA needs metrics-server (`kind` does not ship it by default).
