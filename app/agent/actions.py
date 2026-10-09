"""Human-in-the-loop actions: the agent proposes, the user confirms, the server executes once.

1. create_ticket (tool) stores a *pending action* in Redis under a random
   confirmation_id (TTL 10 min) and returns it. Nothing is written yet.
2. POST /agent/confirm executes it. Exactly-once is enforced in two layers:
   a. Redis:  SET idem:{key} "in_progress" NX  -- only the first request gets past this;
              a duplicate sees "in_progress" (409) or the stored result (200, same ticket).
   b. SQLite: tickets.idempotency_key UNIQUE    -- even if Redis were flushed mid-flight.
   The idempotency key defaults to the confirmation_id, so a double click on the
   same Confirm button can never create two tickets.
"""

import json
import uuid
from dataclasses import asdict, dataclass

import redis

from app.db.repo import create_ticket

PENDING_TTL_S = 600
IDEM_TTL_S = 24 * 3600

TICKET_CATEGORIES = (
    "VPN",
    "Hardware",
    "Software Request",
    "Access Request",
    "Security",
    "MFA re-enrolment",
    "Password",
    "Other",
)
PRIORITIES = ("P1", "P2", "P3", "P4")


@dataclass
class PendingAction:
    confirmation_id: str
    kind: str  # "create_ticket"
    employee_id: str
    session_id: str
    payload: dict
    summary: str

    def to_dict(self) -> dict:
        return asdict(self)


class ConfirmError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        self.status, self.detail = status, detail
        super().__init__(detail)


def propose_ticket(r: redis.Redis, employee_id: str, session_id: str, payload: dict) -> PendingAction:
    cid = uuid.uuid4().hex
    summary = f"Create {payload['priority']} ticket [{payload['category']}]: {payload['title']}"
    action = PendingAction(cid, "create_ticket", employee_id, session_id, payload, summary)
    r.set(f"pending:{cid}", json.dumps(action.to_dict()), ex=PENDING_TTL_S)
    return action


def get_pending(r: redis.Redis, confirmation_id: str) -> PendingAction | None:
    raw = r.get(f"pending:{confirmation_id}")
    return PendingAction(**json.loads(raw)) if raw else None


def confirm_action(r: redis.Redis, confirmation_id: str, employee_id: str, idempotency_key: str | None = None) -> dict:
    key = f"idem:{idempotency_key or confirmation_id}"
    if not r.set(key, "in_progress", nx=True, ex=IDEM_TTL_S):
        stored = r.get(key)
        stored = stored.decode() if isinstance(stored, bytes) else stored
        if stored == "in_progress":
            raise ConfirmError(409, "this action is already being executed")
        return {**json.loads(stored), "status": "duplicate"}

    try:
        action = get_pending(r, confirmation_id)
        if action is None:
            raise ConfirmError(404, "unknown or expired confirmation_id")
        if action.employee_id != employee_id:
            raise ConfirmError(403, "this action belongs to a different employee")
        ticket, created = create_ticket(
            employee_id=action.employee_id, idempotency_key=key.removeprefix("idem:"), **action.payload
        )
        result = {"status": "created" if created else "duplicate", "ticket": ticket}
        r.set(key, json.dumps(result), ex=IDEM_TTL_S)
        r.delete(f"pending:{confirmation_id}")
        return result
    except Exception:
        r.delete(key)  # let the user retry after a failure
        raise
