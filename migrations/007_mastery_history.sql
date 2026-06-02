-- Longitudinal mastery snapshots: one row per mastery update, so we can show
-- learning gains / interview-readiness improving over time (the differentiator
-- a stateless chat tutor structurally can't offer).
CREATE TABLE IF NOT EXISTS mastery_history (
    id BIGSERIAL PRIMARY KEY,
    student_id  TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    concept_id  UUID NOT NULL,
    score       DOUBLE PRECISION NOT NULL,
    confidence  DOUBLE PRECISION NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS mastery_history_student_idx
    ON mastery_history(student_id, occurred_at);
