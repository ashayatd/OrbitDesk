"""Redis connection factory.

REDIS_URL="fakeredis://" gives an in-process fake (used by tests and by the
no-Docker dev mode). Anything else is passed to redis-py.
"""

from functools import lru_cache

import redis

from app.config import get_settings


@lru_cache
def get_redis() -> redis.Redis:
    url = get_settings().redis_url
    if url.startswith("fakeredis://"):
        import fakeredis

        return fakeredis.FakeRedis()
    return redis.Redis.from_url(url, socket_timeout=5, socket_connect_timeout=3)


def redis_ok() -> bool:
    try:
        return bool(get_redis().ping())
    except redis.RedisError:
        return False
