"""Authentication: Argon2id password hashing + server-side sessions.

Design (from research, OWASP/FastAPI-backed):
- Argon2id via argon2-cffi (secure defaults), the current best practice.
- Server-side sessions keyed by a random token held in an HttpOnly cookie — the
  credential never touches JavaScript.
- Login equalizes timing by verifying a dummy hash for unknown users, so response
  time can't be used to enumerate accounts.

A user's ``username`` is also their ``student_id``, isolating each tester's data.
"""

import secrets
from datetime import datetime, timedelta, timezone

import psycopg
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError

COOKIE_NAME = "memex_session"

_ph = PasswordHasher()
# Hashed once at import; used to spend the same CPU on logins for unknown users.
_DUMMY_HASH = _ph.hash("timing-equalization-placeholder")


def hash_password(password: str) -> str:
    return _ph.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _ph.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError):
        return False


class AuthStore:
    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    # ── users ──────────────────────────────────────────────────────────────
    def create_user(self, username: str, email: str, password: str) -> dict:
        """Create a user (and their student row). Raises ValueError on duplicate."""
        username = username.strip()
        email = email.strip().lower()
        if not username or not email or not password:
            raise ValueError("username, email and password are required")
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO users (username, email, password_hash)
                    VALUES (%s, %s, %s) RETURNING id::text, username, email
                    """,
                    (username, email, hash_password(password)),
                )
                user = cur.fetchone()
                # The username doubles as the student_id; ensure the student exists.
                cur.execute(
                    "INSERT INTO students (id) VALUES (%s) ON CONFLICT DO NOTHING",
                    (username,),
                )
            return user
        except psycopg.errors.UniqueViolation as e:
            self.conn.rollback()
            field = "email" if "email" in str(e) else "username"
            raise ValueError(f"That {field} is already taken") from e

    def get_user_by_login(self, login: str) -> dict | None:
        """Look a user up by username or email."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT id::text, username, email, password_hash
                FROM users WHERE username = %s OR email = %s
                """,
                (login.strip(), login.strip().lower()),
            )
            return cur.fetchone()

    def verify_login(self, login: str, password: str) -> dict | None:
        """Return the user on valid credentials, else None — constant-ish time."""
        user = self.get_user_by_login(login)
        if user is None:
            verify_password(_DUMMY_HASH, password)  # equalize timing vs. real users
            return None
        if not verify_password(user["password_hash"], password):
            return None
        return user

    # ── sessions ───────────────────────────────────────────────────────────
    def create_session(self, user_id: str, username: str, *, ttl_days: int = 30) -> str:
        token = secrets.token_urlsafe(32)
        expires = datetime.now(timezone.utc) + timedelta(days=ttl_days)
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (token, user_id, username, expires_at)
                VALUES (%s, %s::uuid, %s, %s)
                """,
                (token, user_id, username, expires),
            )
        return token

    def username_for_session(self, token: str | None) -> str | None:
        if not token:
            return None
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT username FROM sessions WHERE token = %s AND expires_at > now()",
                (token,),
            )
            row = cur.fetchone()
            return row["username"] if row else None

    def delete_session(self, token: str | None) -> None:
        if not token:
            return
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE token = %s", (token,))

    def session_user(self, token: str | None) -> dict | None:
        """Resolve a session token to {username, is_pro} in one query (or None)."""
        if not token:
            return None
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.username, u.is_pro
                FROM sessions s JOIN users u ON u.id = s.user_id
                WHERE s.token = %s AND s.expires_at > now()
                """,
                (token,),
            )
            return cur.fetchone()

    # ---- Entitlement (Pro) ----
    def is_pro(self, username: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT is_pro FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
            return bool(row and row["is_pro"])

    def set_pro(self, username: str, value: bool = True, *, stripe_customer_id: str | None = None) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET is_pro = %s, "
                "stripe_customer_id = COALESCE(%s, stripe_customer_id) WHERE username = %s",
                (value, stripe_customer_id, username),
            )
