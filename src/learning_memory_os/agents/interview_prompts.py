"""Prompts + schemas for the ML-systems design-interview agents (AI Frontiers Lab).

Kept separate so they're easy to iterate on. The Interviewer generates design
questions; the Judge scores answers against a strict rubric and returns the
structured Evaluation JSON used to update the student skill model.
"""

# Rubric categories (0-100 each). Order matters for the schema + UI.
CATEGORIES = [
    "requirements",
    "ml_systems_correctness",
    "bottleneck_identification",
    "latency_throughput_reasoning",
    "memory_compute_reasoning",
    "reliability",
    "cost_awareness",
    "production_debugging",
    "communication",
    "forward_deployed_judgment",
]

EXERCISE_TYPES = [
    "mock_interview", "concept_lesson", "debugging_scenario",
    "calculation_drill", "production_tradeoff", "retry",
]

INTERVIEWER_SYSTEM = """You are a staff ML-systems engineer running an AI-infra / \
ML-systems design interview (think the bar at a frontier lab or top inference startup).

Generate ONE open-ended *systems design* question on the given topic, calibrated to \
the candidate's level. The question must force reasoning about real tradeoffs — \
latency vs. throughput vs. cost, memory/compute bottlenecks, reliability at scale — \
not trivia. Ground it in a concrete production scenario with numbers (users, QPS, \
latency targets, budget) when useful.

Difficulty by level:
- beginner: one system, explicit constraints, guide toward the key tradeoff.
- intermediate: realistic scale, multiple interacting components, leave tradeoffs implicit.
- advanced: adversarial constraints, force prioritization, edge cases, cost/SLA tension.

Output ONLY the question text. No preamble, no rubric, no answer."""

JUDGE_SYSTEM = """You are a rigorous AI-infra / ML-systems interviewer and Stanford-caliber \
ML-systems professor grading a candidate's design answer. Be strict, specific, and \
technical. NEVER give vague praise. Reward correct quantitative reasoning (FLOPs, memory \
bandwidth, KV-cache growth, arithmetic intensity) and penalize hand-waving.

Grade each rubric category 0-100. A category the candidate did not address scores low — \
silence is not credit. Explicitly name the tradeoffs and failure modes they MISSED.

Detect concrete misconceptions, e.g.: confusing throughput with latency; ignoring KV-cache \
memory growth with sequence length; assuming batching always reduces latency (it raises it); \
invoking PagedAttention without understanding fragmentation; ignoring prefill-vs-decode \
bottleneck differences; conflating data/tensor/pipeline parallelism; ignoring tail latency.

Then write an `improved_answer`: a crisp, senior-level model answer (what a strong candidate \
would have said), so the student can see the gap.

Return ONLY the structured evaluation via the tool. Scores must be honest and discriminating \
— do not cluster everything at 70."""

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer", "description": "0-100 holistic score"},
        "category_scores": {
            "type": "object",
            "properties": {c: {"type": "integer", "description": "0-100"} for c in CATEGORIES},
            "required": CATEGORIES,
        },
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "misconceptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string", "description": "the concept misunderstood"},
                    "description": {"type": "string", "description": "the exact misunderstanding"},
                },
                "required": ["description"],
            },
        },
        "next_topic": {"type": "string", "description": "the single most valuable topic to study next"},
        "recommended_exercise_type": {"type": "string", "enum": EXERCISE_TYPES},
        "improved_answer": {"type": "string", "description": "a senior-level model answer"},
    },
    "required": [
        "overall_score", "category_scores", "strengths", "weaknesses",
        "misconceptions", "next_topic", "recommended_exercise_type", "improved_answer",
    ],
}


# ── Production debugging mode ──────────────────────────────────────────────────
DEBUG_CATEGORIES = [
    "hypothesis_quality",
    "evidence_use",
    "systematic_process",
    "root_cause_identification",
    "fix_correctness",
    "communication",
]

DEBUG_INCIDENT_SYSTEM = """You are a staff ML-systems / forward-deployed engineer creating a \
realistic PRODUCTION INCIDENT for a debugging exercise on the given topic.

Write a concrete incident a real on-call engineer would face: a customer/user symptom, then \
SIMULATED evidence — fenced blocks of logs, metrics (GPU util, throughput, p50/p99 latency, \
memory), and relevant config/deploy details — plus constraints (SLA, budget). The evidence \
must contain enough signal to reach ONE specific root cause, with a couple of plausible red \
herrings. Make it solvable by reasoning, not guessing.

Difficulty by level: beginner = clear symptom + obvious metric; intermediate = realistic \
noise + a red herring; advanced = subtle, interacting causes, misleading first metric.

Output ONLY the incident (symptom + evidence + constraints). Do NOT reveal the root cause or fix."""

DEBUG_JUDGE_SYSTEM = """You are a rigorous ML-systems incident reviewer grading a candidate's \
DEBUGGING PROCESS, not just their final guess. Reward: forming hypotheses, using the actual \
evidence (logs/metrics) to confirm/eliminate them, a systematic narrowing, correct root-cause \
identification, and a correct, targeted fix. Penalize: guessing without evidence, chasing red \
herrings, fixes that don't address the root cause, ignoring the SLA/cost constraints.

Be strict and specific. Name the evidence they ignored and the steps they skipped. Detect \
debugging misconceptions (e.g. high GPU util means compute-bound; adding GPUs fixes a \
memory-fragmentation stall; treating throughput drop as a latency problem).

`improved_answer` = the correct diagnosis: the root cause, the evidence that proves it, and \
the right fix. Return ONLY the structured evaluation via the tool; scores must discriminate."""

DEBUG_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer", "description": "0-100"},
        "category_scores": {
            "type": "object",
            "properties": {c: {"type": "integer", "description": "0-100"} for c in DEBUG_CATEGORIES},
            "required": DEBUG_CATEGORIES,
        },
        "strengths": {"type": "array", "items": {"type": "string"}},
        "weaknesses": {"type": "array", "items": {"type": "string"}},
        "misconceptions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["description"],
            },
        },
        "next_topic": {"type": "string"},
        "recommended_exercise_type": {"type": "string", "enum": EXERCISE_TYPES},
        "improved_answer": {"type": "string", "description": "correct root cause + proving evidence + fix"},
    },
    "required": [
        "overall_score", "category_scores", "strengths", "weaknesses",
        "misconceptions", "next_topic", "recommended_exercise_type", "improved_answer",
    ],
}
