"""Redis distributed locks.

redis-py's Lock stores a random token under the key with SET NX PX (atomic
"create if absent, with TTL") and releases with a Lua script that deletes the key
only if it still holds *our* token. So:
- two workers can never hold the same lock (SET NX is atomic in Redis),
- a crashed worker cannot block forever (TTL),
- a slow worker whose lock expired cannot delete someone else's lock (token check).
"""

from collections.abc import Iterator
from contextlib import contextmanager, suppress

import redis


class LockNotAcquiredError(Exception):
    pass


@contextmanager
def distributed_lock(r: redis.Redis, name: str, ttl_s: float = 300, wait_s: float = 30) -> Iterator[None]:
    lock = r.lock(f"lock:{name}", timeout=ttl_s, blocking_timeout=wait_s)
    if not lock.acquire():
        raise LockNotAcquiredError(f"could not acquire lock:{name} within {wait_s}s")
    try:
        yield
    finally:
        # LockError = our TTL expired while we worked; someone else may own it now, so leave it.
        with suppress(redis.exceptions.LockError):
            lock.release()
