CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Semantic tier: stable course/topic facts
CREATE TABLE semantic_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,        -- concept | example | misconception | exercise | code_pattern | paper_claim
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX semantic_items_topic_idx ON semantic_items(topic_id);
CREATE INDEX semantic_items_type_idx ON semantic_items(artifact_type);
CREATE INDEX semantic_items_embedding_idx ON semantic_items USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Prerequisite graph (concept-level edges)
CREATE TABLE prerequisites (
    src UUID REFERENCES semantic_items(id) ON DELETE CASCADE,
    dst UUID REFERENCES semantic_items(id) ON DELETE CASCADE,
    PRIMARY KEY (src, dst)
);

-- Student tier: per-student mastery + misconceptions
CREATE TABLE students (
    id TEXT PRIMARY KEY,
    profile JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE mastery (
    student_id TEXT REFERENCES students(id) ON DELETE CASCADE,
    concept_id UUID REFERENCES semantic_items(id) ON DELETE CASCADE,
    score REAL NOT NULL,
    confidence REAL NOT NULL,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (student_id, concept_id)
);

CREATE TABLE misconceptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id TEXT REFERENCES students(id) ON DELETE CASCADE,
    concept_id UUID REFERENCES semantic_items(id),
    description TEXT NOT NULL,
    evidence TEXT,
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ
);
CREATE INDEX misconceptions_student_idx ON misconceptions(student_id);

-- Episodic tier: append-only event log
CREATE TABLE episodic_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id TEXT REFERENCES students(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,           -- session_start | question | tutor_reply | quiz_attempt | exercise_attempt
    payload JSONB NOT NULL,
    embedding vector(1536),
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX episodic_student_idx ON episodic_events(student_id, occurred_at DESC);
CREATE INDEX episodic_embedding_idx ON episodic_events USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Intervention tier: which tutoring strategy was tried, did it work
CREATE TABLE interventions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    student_id TEXT REFERENCES students(id) ON DELETE CASCADE,
    misconception_id UUID REFERENCES misconceptions(id) ON DELETE SET NULL,
    strategy TEXT NOT NULL,
    outcome TEXT,                       -- helped | partial | no_effect | regressed | unknown
    notes TEXT,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
