-- 005: per-user "learning traces" — one row per chat turn, captured for later
-- fine-tuning. A superset of the synthetic Trajectory schema (student_state,
-- task, candidate_pool, selection) plus tutor/RL fields (reply, reward) so the
-- same rows serve both a router fine-tune and a tutor fine-tune. Idempotent.

CREATE TABLE IF NOT EXISTS learning_traces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id TEXT REFERENCES students(id) ON DELETE CASCADE,
    conversation_id UUID,
    turn_ordinal INTEGER,
    task_type TEXT NOT NULL DEFAULT 'explain',
    task_text TEXT NOT NULL,
    budget INTEGER NOT NULL DEFAULT 3000,
    student_state JSONB NOT NULL DEFAULT '{}'::jsonb,
    candidate_pool JSONB NOT NULL DEFAULT '[]'::jsonb,
    selected_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    dropped_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    reply TEXT,
    model TEXT,
    reward REAL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS learning_traces_student_idx
    ON learning_traces(student_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS learning_traces_turn_idx
    ON learning_traces(student_id, turn_ordinal);
