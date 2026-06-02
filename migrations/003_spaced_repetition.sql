-- 003: spaced-repetition scheduling on the mastery (student) tier.
-- Adds an SM-2-style review schedule so the tutor can resurface decaying
-- knowledge. Idempotent so it is safe to re-run.

ALTER TABLE mastery
    ADD COLUMN IF NOT EXISTS reps INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS interval_days REAL NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS next_review_at TIMESTAMPTZ DEFAULT (now() + interval '1 day');

-- Backfill existing rows: schedule the next review one interval after the last update.
UPDATE mastery
   SET next_review_at = last_updated + (interval_days || ' days')::interval
 WHERE next_review_at IS NULL;

CREATE INDEX IF NOT EXISTS mastery_due_idx ON mastery(student_id, next_review_at);
