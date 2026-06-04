# Deploying Memex on Vercel

The FastAPI app runs as a single Python serverless function (`api/index.py`,
served via `vercel.json`). Because serverless functions are slim and stateless,
two things differ from a normal server:

- **Database is external** — use a free **Neon** Postgres (set `DATABASE_URL`).
- **No boot step** — migrations are run **once, manually** (below), not on deploy.
- **Slim deps** — `api/requirements.txt` excludes the ML stack to stay under
  Vercel's 250 MB function limit (the API never imports it at runtime).

## 1. Neon Postgres
1. **neon.tech** → create a project → copy the **Pooled** connection string
   (`postgresql://…-pooler.…neon.tech/…?sslmode=require`).
2. Apply the schema once (from your machine):
   ```bash
   DATABASE_URL="<neon-url>" uv run python -m scripts.migrate
   ```

## 2. Vercel
1. **vercel.com** → **Add New → Project** → import `HivaMohammadzadeh1/CS153-frontier-systems`.
   It auto-detects the Python function + `vercel.json`. (No build command needed.)
2. **Settings → Environment Variables** — add:
   - `DATABASE_URL` = your Neon pooled URL
   - `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`
   - `COOKIE_SECURE` = `true`
   - `APP_BASE_URL` and `LMOS_PUBLIC_URL` = your Vercel URL (set after first deploy,
     e.g. `https://memex.vercel.app`)
   - `STRIPE_SECRET_KEY`, `STRIPE_PRICE_ID`, `STRIPE_WEBHOOK_SECRET` (later — billing
     stays off until these are set)
3. **Deploy.** Check `https://<your-app>.vercel.app/api/health` → `{"status":"ok"}`.

## 3. Seed the curriculum (once)
```bash
DATABASE_URL="<neon-url>" OPENAI_API_KEY="<key>" uv run python -m scripts.ingest_all
```

## 4. Stripe ($5 one-time)
Same as `DEPLOY.md`, but the webhook URL is
`https://<your-app>.vercel.app/api/billing/webhook`. Add the three `STRIPE_*` env
vars to turn on the paywall.

## Caveats (serverless)
- **Streaming**: Vercel may buffer the SSE response rather than stream tokens
  live, and long answers can hit the 60s function limit. If chat ever times out,
  point the frontend at the non-streaming `/api/chat` endpoint (it's already built)
  — ping me and it's a one-line change.
- **Cold starts**: first request after idle is slower.
- For a long-running server with native streaming, Render (`DEPLOY.md`) is better.
