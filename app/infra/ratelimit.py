"""Per-session rate limit: fixed 60-second window in Redis.

INCR is atomic, so concurrent requests from the same session (multiple tabs, API
replicas) all count against one shared counter. Fixed windows allow a burst of up
to 2x the limit at a window boundary; a sliding log (sorted set) would fix that at
the cost of O(limit) memory per session, which isn't worth it here.
"""

import time

import redis

from app.errors import RateLimitedError


def check_rate_limit(r: redis.Redis, session_id: str, limit_per_minute: int) -> int:
    """Count this request; raise RateLimitedError if over the limit. Returns the count."""
    window = int(time.time() // 60)
    key = f"rl:{session_id}:{window}"
    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, 61)
    count = int(pipe.execute()[0])
    if count > limit_per_minute:
        raise RateLimitedError(retry_after_s=60 - int(time.time() % 60))
    return count
