"""Auth: Argon2id hashing + user/session store."""

import uuid

import pytest

from learning_memory_os.auth import (
    AuthStore,
    hash_password,
    verify_password,
)


def test_hash_and_verify_password():
    h = hash_password("correct horse battery staple")
    assert h.startswith("$argon2id$")          # Argon2id, not bcrypt/plaintext
    assert verify_password(h, "correct horse battery staple") is True
    assert verify_password(h, "wrong password") is False


def test_create_user_and_login(db_conn):
    store = AuthStore(db_conn)
    uname = f"alice-{uuid.uuid4().hex[:6]}"
    user = store.create_user(uname, f"{uname}@example.com", "s3cret-pw")
    assert user["username"] == uname

    # a students row was created so per-user data can key on the username
    with db_conn.cursor() as cur:
        cur.execute("SELECT 1 FROM students WHERE id = %s", (uname,))
        assert cur.fetchone() is not None

    assert store.verify_login(uname, "s3cret-pw")["username"] == uname           # by username
    assert store.verify_login(f"{uname}@example.com", "s3cret-pw") is not None   # by email
    assert store.verify_login(uname, "bad") is None                              # wrong pw
    assert store.verify_login("nobody", "whatever") is None                      # unknown user


def test_duplicate_username_rejected(db_conn):
    store = AuthStore(db_conn)
    uname = f"bob-{uuid.uuid4().hex[:6]}"
    store.create_user(uname, f"{uname}@example.com", "pw")
    with pytest.raises(ValueError):
        store.create_user(uname, f"other-{uname}@example.com", "pw")


def test_session_lifecycle(db_conn):
    store = AuthStore(db_conn)
    uname = f"carol-{uuid.uuid4().hex[:6]}"
    user = store.create_user(uname, f"{uname}@example.com", "pw")
    token = store.create_session(user["id"], uname)
    assert store.username_for_session(token) == uname
    assert store.username_for_session("bogus-token") is None
    assert store.username_for_session(None) is None
    store.delete_session(token)
    assert store.username_for_session(token) is None


def test_expired_session_is_invalid(db_conn):
    store = AuthStore(db_conn)
    uname = f"dan-{uuid.uuid4().hex[:6]}"
    user = store.create_user(uname, f"{uname}@example.com", "pw")
    token = store.create_session(user["id"], uname, ttl_days=30)
    # force it expired
    with db_conn.cursor() as cur:
        cur.execute(
            "UPDATE sessions SET expires_at = now() - interval '1 hour' WHERE token = %s",
            (token,),
        )
    assert store.username_for_session(token) is None
