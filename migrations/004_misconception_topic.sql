-- 004: associate misconceptions with a topic so the selector's misconception
-- boost can fire even when no specific concept_id is known (e.g. misconceptions
-- surfaced by the diagnostic flow). Idempotent.

ALTER TABLE misconceptions
    ADD COLUMN IF NOT EXISTS topic_id TEXT;

CREATE INDEX IF NOT EXISTS misconceptions_topic_idx ON misconceptions(student_id, topic_id);
