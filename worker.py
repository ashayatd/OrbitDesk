"""RQ worker entrypoint: `python worker.py` (Linux / Docker only).

RQ's default Worker forks a child process per job (isolation: a crashing job can't
kill the worker). Windows has no fork(), so on Windows use QUEUE_MODE=inline or
run this in Docker (docker compose up worker).
"""

import sys

from rq import Worker

from app.config import get_settings
from app.infra.queue import get_queue
from app.infra.redis_client import get_redis
from app.log import configure_logging, get_logger


def main() -> None:
    configure_logging()
    if sys.platform == "win32":
        sys.exit("RQ workers need fork(): run `docker compose up worker`, or set QUEUE_MODE=inline.")
    log = get_logger("worker")
    # Warm the models once in the parent so forked children inherit them (copy-on-write).
    from app.retrieval.embeddings import get_embedder

    get_embedder()
    log.info("worker_start", redis=get_settings().redis_url)
    Worker([get_queue()], connection=get_redis()).work(with_scheduler=False)


if __name__ == "__main__":
    main()
