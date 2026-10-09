"""Global index version. Every ingest that changes chunks bumps it.

Cache keys embed the version, so bumping it invalidates every cached answer
at once (old keys simply stop being read and expire via TTL).
"""

import redis

KEY = "index:version"


def get_index_version(r: redis.Redis) -> int:
    v = r.get(KEY)
    return int(v) if v else 0


def bump_index_version(r: redis.Redis) -> int:
    return int(r.incr(KEY))
