"""Small dialect helpers so the same code runs on Postgres and SQLite."""

from sqlalchemy import Table
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.engine import Connection


def insert_ignore(conn: Connection, table: Table):
    """INSERT ... ON CONFLICT DO NOTHING (both Postgres and SQLite >= 3.24 support it)."""
    mod = postgresql if conn.dialect.name == "postgresql" else sqlite
    return mod.insert(table).on_conflict_do_nothing()


def insert_on_conflict_do_nothing(conn: Connection, table: Table, conflict_column: str):
    mod = postgresql if conn.dialect.name == "postgresql" else sqlite
    return mod.insert(table).on_conflict_do_nothing(index_elements=[conflict_column])


def upsert(conn: Connection, table: Table, conflict_column: str, values: dict, update: dict):
    """INSERT ... ON CONFLICT (col) DO UPDATE SET ... (Postgres and SQLite)."""
    mod = postgresql if conn.dialect.name == "postgresql" else sqlite
    return mod.insert(table).values(**values).on_conflict_do_update(index_elements=[conflict_column], set_=update)
