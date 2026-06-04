# Deploying Memex (Render + Stripe $5 one-time)

The app is a long-running FastAPI server (with SSE streaming) + a managed Postgres,
served over HTTPS. Render fits this; serverless hosts (Vercel) do not — their
function timeouts kill streaming and they don't host Postgres.

Billing is **off until Stripe env vars are set**: with them unset, every signed-in
user has full access (good for local dev). With them set, a user must pay the
one-time $5 (becomes `is_pro`) before using anything but the billing routes.

## 1. Stripe (one-time $5)
1. Create a Stripe account → **Developers → API keys** → copy the **Secret key**
   (`sk_test_…` first; swap to `sk_live_…` when ready).
2. **Products → add product** → "Memex access", **one-time** price **$5.00** →
   copy the **Price ID** (`price_…`).
3. **Developers → Webhooks → add endpoint**:
   - URL: `https://<your-app>/api/billing/webhook`
   - Event: `checkout.session.completed`
   - Copy the **Signing secret** (`whsec_…`).
4. Test card later: `4242 4242 4242 4242`, any future expiry/CVC.

> No-code alternative: skip the API keys and create a **Payment Link** for the $5
> product; set `LMOS_CHECKOUT_URL` to it. The webhook still grants access.

## 2. Database (free Neon)
Render allows only one free managed Postgres per account, so the app uses an
**external** Postgres via `DATABASE_URL`:
1. **neon.tech** → sign up → **New Project** → copy the **connection string**
   (`postgresql://…@…neon.tech/…?sslmode=require`).

## 3. Render
1. Push to GitHub (done).
2. Render → **New + → Blueprint** → select this repo (reads `render.yaml`: a Docker
   web service, no managed DB). Migrations run on every boot (idempotent).
3. In the service's **Environment** tab, set the `sync: false` vars:
   - `DATABASE_URL` → your Neon connection string
   - `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
   - `APP_BASE_URL` and `LMOS_PUBLIC_URL` → your URL (e.g. `https://memex.onrender.com`)
   - `STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID`, `STRIPE_WEBHOOK_SECRET`
4. Deploy. Health check: `GET /api/health`.

## 3. Seed the curriculum (one-time)
The tutor needs course content embedded into Postgres for retrieval:
```
# from a shell with the prod DATABASE_URL + OPENAI_API_KEY exported:
uv run python -m scripts.ingest_all
```
(`/api/topics` works without this, but chat answers will lack context until it runs.)

## 4. Onboard people
Share the URL. Flow: sign up → "Unlock for $5" → Stripe checkout → returns to the
app unlocked. Each user's memory/history/progress is isolated to their account.

## Notes
- `COOKIE_SECURE=true` is set in `render.yaml` (HTTPS). Keep it `false` only for
  local `http://` testing.
- Render free web spins down when idle (~30–60s cold start on first hit). For
  always-on, bump the plan. Free Postgres is free for 90 days, then a small fee;
  Fly.io is an alternative with an indefinite free allowance.
