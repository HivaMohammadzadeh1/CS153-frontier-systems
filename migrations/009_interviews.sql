-- AI Frontiers Lab: design-interview evaluations + persisted tutor reflections.
CREATE TABLE IF NOT EXISTS interview_evaluations (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id    TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    topic_id      TEXT,
    level         TEXT,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL,
    overall_score INT,
    evaluation    JSONB NOT NULL,
    occurred_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS interview_eval_student_idx
    ON interview_evaluations(student_id, occurred_at DESC);

CREATE TABLE IF NOT EXISTS tutor_reflections (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id  TEXT NOT NULL REFERENCES students(id) ON DELETE CASCADE,
    summary     TEXT NOT NULL,
    payload     JSONB,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS tutor_reflections_student_idx
    ON tutor_reflections(student_id, occurred_at DESC);
