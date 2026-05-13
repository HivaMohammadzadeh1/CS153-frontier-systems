import os
import uuid
import pytest
import psycopg
from psycopg.rows import dict_row

DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://lmos:lmos_dev@localhost:5433/learning_memory_os",
)


@pytest.fixture
def db_conn():
    """Yields a psycopg connection. Each test runs in a transaction that is rolled back."""
    conn = psycopg.connect(DB_URL, row_factory=dict_row)
    conn.autocommit = False
    yield conn
    conn.rollback()
    conn.close()


@pytest.fixture
def fresh_student_id(db_conn):
    sid = f"test-{uuid.uuid4().hex[:8]}"
    with db_conn.cursor() as cur:
        cur.execute("INSERT INTO students (id) VALUES (%s)", (sid,))
    yield sid
