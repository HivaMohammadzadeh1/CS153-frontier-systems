# Sign-in / Sign-up Auth Layer — Design Spec

Date: 2026-06-01
Status: Implemented
Research: deep-research workflow (OWASP / FastAPI sources, adversarially verified)

## Goal
Let external beta testers create accounts and use the platform, with each tester's
memory/history/traces isolated under their own identity.

## Decisions (research-backed)
- **Session-cookie auth, not JWT** — HttpOnly + SameSite=Lax + (configurable) Secure
  cookie; the credential never touches JS. Simplest + safest for our same-origin
  static frontend (browser attaches the cookie automatically).
- **Argon2id** via `argon2-cffi` (secure defaults). Avoid `passlib` (unmaintained,
  depends on `crypt` removed in Python 3.13).
- **Server-side sessions** (random `secrets.token_urlsafe` token in a `sessions`
  table) — no extra dependency, trivially revocable.
- **Timing equalization**: unknown-user logins verify a dummy hash so response time
  can't enumerate accounts.

## Components
- Migration `006_auth.sql`: `users` (id, **username**, email, password_hash) +
  `sessions` (token, user_id, username, expires_at). The username *is* the
  `student_id`, so all existing per-user data keys on it; signup also creates the
  `students` row.
- `auth.py`: `hash_password`/`verify_password` (Argon2id) + `AuthStore`
  (create_user, get_user_by_login, verify_login, create/validate/delete session).
- API: `POST /api/auth/{signup,login,logout}`, `GET /api/auth/me`; an
  `auth_gate` middleware requiring a valid session for all `/api/` routes except a
  public list (health, topics, info, routers, auth/*), and enforcing
  `/api/student/{id}` ownership (403 on mismatch). Write endpoints (chat, quiz,
  diagnostic, feedback, conversations) **override** `student_id` with the session
  identity — client-supplied ids are never trusted.
- Frontend: a sign-in / sign-up **gate** (shown when `/api/auth/me` is 401), which
  boots the app under the authenticated username; the settings popover shows the
  signed-in identity + a Log out button. Cookie auth = no token handling in JS.

## Config
`COOKIE_SECURE` (default false for local http testing; set true behind HTTPS),
`SESSION_TTL_DAYS` (default 30).

## Testing
Store tests (hash/verify, user create + duplicate, login by username/email, wrong
pw, unknown user, session create/validate/expire/delete) + API tests (signup→cookie,
me, gating 401, ownership 403, data routes scoped to caller). Existing API tests
updated to authenticate. Full suite green (138). Auth flow verified live
(gate → signup → app boots → session persists across reload; curl 401/200/403).

## Pitfalls handled
No tokens in localStorage; timing-equalized login; unique username/email; session
expiry + logout invalidation; ownership checks; secure-cookie configurable so local
testers on http aren't locked out.
