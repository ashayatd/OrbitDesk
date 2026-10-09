"""MLflow setup: local SQLite tracking DB under mlruns/ (gitignored).

MLflow 3.x put the plain ./mlruns file store into maintenance mode; a single
SQLite file is the supported local backend and is still just a file on disk.
View with:  uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
"""

from common import ROOT


def init_mlflow(experiment: str):
    import mlflow

    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlruns' / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(experiment)
    return mlflow
