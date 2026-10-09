"""RQ queue.

QUEUE_MODE=rq      jobs go to Redis and a separate `worker.py` process (Linux/Docker) runs them.
QUEUE_MODE=inline  RQ's synchronous mode (is_async=False): the same enqueue call runs the
                   job immediately in this process. No fork, so it works on Windows and in tests.
"""

from rq import Queue
from rq.job import Job

from app.config import get_settings
from app.infra.redis_client import get_redis

INGEST_QUEUE = "ingest"


def get_queue(name: str = INGEST_QUEUE) -> Queue:
    return Queue(name, connection=get_redis(), is_async=get_settings().queue_mode != "inline")


def job_status(job_id: str) -> dict | None:
    try:
        job = Job.fetch(job_id, connection=get_redis())
    except Exception:  # rq raises NoSuchJobError
        return None
    status = job.get_status(refresh=True)
    return {
        "job_id": job.id,
        "status": str(status.value if hasattr(status, "value") else status),
        "result": job.return_value(),
        "error": job.latest_result().exc_string if status == "failed" and job.latest_result() else None,
    }
