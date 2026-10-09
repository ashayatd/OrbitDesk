"""Human-in-the-loop actions: the agent proposes, the user confirms, the database executes once.

1. create_ticket (tool) inserts a row into `pending_actions` (expires after 10 min) and returns
   its confirmation_id. No ticket exists yet.
2. POST /agent/confirm runs ONE transaction:
     SELECT ... FROM pending_actions WHERE confirmation_id = ? FOR UPDATE   -- serialises double clicks
     INSERT INTO tickets ... ON CONFLICT (idempotency_key) DO NOTHING RETURNING id
     UPDATE pending_actions SET status = 'executed', ticket_id = ...
   The idempotency key defaults to the confirmation_id. A second click waits for the row lock,
   then sees status 'executed' and gets the same ticket back with status "duplicate".
   The UNIQUE constraint on tickets.idempotency_key guarantees one ticket even if two
   different pending actions were confirmed with the same client-supplied key.
On SQLite, BEGIN IMMEDIATE (see app/db/database.py) plays the role of FOR UPDATE.
"""

import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

from sqlalchemy import insert, select, update

from app.db.database import db_session, init_db, pending_actions, tickets
from app.db.repo import insert_ticket_once, ticket_dict

PENDING_TTL = timedelta(minutes=10)

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


def propose_ticket(employee_id: str, session_id: str, payload: dict) -> PendingAction:
    init_db()
    summary = f"Create {payload['priority']} ticket [{payload['category']}]: {payload['title']}"
    action = PendingAction(uuid.uuid4().hex, "create_ticket", employee_id, session_id, payload, summary)
    with db_session() as conn:
        conn.execute(
            insert(pending_actions).values(
                confirmation_id=action.confirmation_id,
                kind=action.kind,
                employee_id=employee_id,
                session_id=session_id,
                payload=payload,
                summary=summary,
                expires_at=datetime.now() + PENDING_TTL,
            )
        )
    return action


def confirm_action(confirmation_id: str, employee_id: str, idempotency_key: str | None = None) -> dict:
    init_db()
    pa = pending_actions.c
    with db_session() as conn:
        row = conn.execute(
            select(pending_actions).where(pa.confirmation_id == confirmation_id).with_for_update()
        ).first()
        if row is None:
            raise ConfirmError(404, "unknown confirmation_id")
        if row.employee_id != employee_id:
            raise ConfirmError(403, "this action belongs to a different employee")
        if row.status == "executed":  # double click / retry: same ticket, no new row
            ticket = conn.execute(select(tickets).where(tickets.c.id == row.ticket_id)).first()
            return {"status": "duplicate", "ticket": ticket_dict(ticket)}
        if row.status != "pending" or row.expires_at < datetime.now():
            raise ConfirmError(410, "this action expired; ask the assistant again")

        ticket, created = insert_ticket_once(conn, employee_id, row.payload, idempotency_key or confirmation_id)
        conn.execute(
            update(pending_actions)
            .where(pa.confirmation_id == confirmation_id)
            .values(status="executed", ticket_id=ticket["id"])
        )
    return {"status": "created" if created else "duplicate", "ticket": ticket}
