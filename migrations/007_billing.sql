-- 007: one-time $5 access. A user is `paid` once their Stripe checkout completes.
-- Idempotent.

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS paid BOOLEAN NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS paid_at TIMESTAMPTZ;
