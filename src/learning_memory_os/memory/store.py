import psycopg
from psycopg.rows import dict_row


def connect(database_url: str) -> psycopg.Connection:
    return psycopg.connect(database_url, row_factory=dict_row, autocommit=False)


def vec_literal(v: list[float]) -> str:
    """Postgres pgvector accepts a string literal `'[0.1,0.2,...]'`."""
    return "[" + ",".join(f"{x:.7f}" for x in v) + "]"
