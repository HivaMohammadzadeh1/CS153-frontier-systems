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

# Hire/no-hire weights from a staff ML-infra interviewer (each sums to 1.0).
# Communication is a multiplier, not a driver — deliberately small.
CATEGORY_WEIGHTS = {
    "ml_infra": {
        "ml_systems_correctness": 0.15, "bottleneck_identification": 0.15,
        "memory_compute_reasoning": 0.15, "latency_throughput_reasoning": 0.15,
        "production_debugging": 0.12, "reliability": 0.10, "cost_awareness": 0.08,
        "requirements": 0.05, "forward_deployed_judgment": 0.03, "communication": 0.02,
    },
    "forward_deployed": {
        "ml_systems_correctness": 0.13, "bottleneck_identification": 0.13,
        "memory_compute_reasoning": 0.12, "latency_throughput_reasoning": 0.12,
        "production_debugging": 0.12, "reliability": 0.10, "cost_awareness": 0.08,
        "requirements": 0.08, "forward_deployed_judgment": 0.08, "communication": 0.04,
    },
}
# A weak score in any of these blocks "interview-ready" regardless of the average.
CRITICAL_CATEGORIES = [
    "bottleneck_identification", "memory_compute_reasoning",
    "latency_throughput_reasoning", "production_debugging",
]


def weighted_overall(category_scores: dict, role: str = "ml_infra") -> int:
    """Compute the overall score as the staff-interviewer weighted sum of category
    scores (more defensible than trusting the LLM's holistic number)."""
    w = CATEGORY_WEIGHTS.get(role, CATEGORY_WEIGHTS["ml_infra"])
    total = sum(w.values()) or 1.0
    return round(sum(float(category_scores.get(c, 0)) * wt for c, wt in w.items()) / total)


# Canonical, high-consequence ML-systems misconceptions (staff-interviewer list).
MISCONCEPTION_BANK = [
    "Confusing throughput with latency.",
    "Assuming batching always lowers latency (it raises per-request latency).",
    "Ignoring KV-cache memory growth with sequence length and batch size.",
    "Citing PagedAttention without understanding fragmentation / block-based KV memory.",
    "Failing to distinguish prefill (compute-bound) from decode (memory-bandwidth-bound) bottlenecks.",
    "Treating high GPU utilization as always good, without checking useful work, queueing, or memory stalls.",
    "Optimizing only average latency and ignoring p95/p99 tail latency.",
    "Assuming quantization only affects quality, not memory bandwidth, cost, and kernel behavior.",
    "Choosing data/tensor/pipeline parallelism without accounting for communication overhead.",
    "Proposing 'add more GPUs' before identifying the actual bottleneck.",
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

INTERVIEW_FOLLOWUP_SYSTEM = """You are the same staff ML-systems interviewer, now mid-interview. \
You have the original question and the candidate's answer(s) so far. Ask ONE sharp follow-up \
that a real interviewer would ask next — drilling into the WEAKEST or vaguest part of their \
latest answer, forcing a number, exposing a missed tradeoff, or pushing on an unstated \
assumption. Do not re-ask what they already covered. Do not coach or reveal the answer. \
Keep it to 1-3 sentences. Output ONLY the follow-up question."""

JUDGE_SYSTEM = """You are a rigorous AI-infra / ML-systems interviewer and Stanford-caliber \
ML-systems professor grading a candidate's design answer. Be strict, specific, and \
technical. NEVER give vague praise. Reward correct quantitative reasoning (FLOPs, memory \
bandwidth, KV-cache growth, arithmetic intensity) and penalize hand-waving.

Grade each rubric category 0-100. A category the candidate did not address scores low — \
silence is not credit. Explicitly name the tradeoffs and failure modes they MISSED.

Detect concrete misconceptions. Watch especially for these high-consequence ones: \
(1) confusing throughput with latency; (2) assuming batching always lowers latency; \
(3) ignoring KV-cache memory growth with sequence length and batch size; (4) citing \
PagedAttention without understanding fragmentation / block-based KV memory; (5) not \
distinguishing prefill (compute-bound) from decode (memory-bandwidth-bound) bottlenecks; \
(6) treating high GPU utilization as always good; (7) optimizing only average latency, \
ignoring p95/p99; (8) assuming quantization only affects quality, not bandwidth/cost/kernels; \
(9) picking parallelism without accounting for communication overhead; (10) saying "add more \
GPUs" before identifying the bottleneck.

Communication is a MULTIPLIER, not a driver: a great answer must be explainable, but polished \
prose cannot rescue weak technical reasoning. Score communication accordingly.

Calibrate the `overall_score` the way a real interviewer judges ONE response — by the quality \
of what they actually reasoned, NOT by how many of the 10 categories they happened to touch. A \
focused, correct answer that simply doesn't cover every dimension is still a strong answer:
- 85-100: unusually complete, staff-level — quantitative reasoning + tradeoffs + failure modes.
- 75-84 : strong; correct quantitative reasoning and clear tradeoffs, minor gaps.
- 65-74 : good answer that would survive follow-ups; right instincts, some depth missing.
- 55-64 : strong but partial — correct core (e.g. right bottleneck + rough math) missing dimensions.
- 40-54 : partial; notable gaps or one real error.
- below 40: mostly buzzwords, misses the main bottleneck, or makes a dangerous claim.
A correct, well-reasoned answer that doesn't enumerate every possible dimension should NOT score \
below 55. Reserve sub-40 for genuinely weak answers.

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


# ── Forward-deployed engineer mode ─────────────────────────────────────────────
# A customer-facing scenario ("our AI agent feels slow") graded on the 7 sub-skills
# that separate a forward-deployed engineer from a pure backend engineer.
FORWARD_CATEGORIES = [
    "problem_framing",          # turn a vague customer complaint into a measurable problem
    "metric_selection",         # ask for / choose the RIGHT signals (p99, tokens/s, util, queue)
    "bottleneck_localization",  # localize to prefill/decode/KV/batching/network/client
    "hypothesis_iteration",     # propose, test, and discard hypotheses with the evidence
    "solution_tradeoffs",       # fix that respects latency/quality/cost constraints
    "cost_business_awareness",  # tie the fix to $ / SLA / customer impact
    "customer_communication",   # explain it to a non-expert stakeholder without hand-waving
]

FORWARD_SCENARIO_SYSTEM = """You are a staff forward-deployed engineer at a frontier-model \
company, writing a realistic CUSTOMER SCENARIO for a training exercise on the given topic.

A customer (a non-expert, e.g. a PM or founder at the customer company) reports a vague, \
business-flavored problem about an AI product they built on your serving stack — e.g. "our \
support agent feels slow," "costs are exploding," "answers got worse last week," "it falls \
over at peak." Write the scenario as the customer would actually say it: a real complaint, \
some business context (scale, SLA, budget, what they tried), and a FEW concrete-but-incomplete \
facts. CRUCIALLY, withhold the precise metrics — a strong engineer must ask for the right ones. \
Embed a single coherent underlying root cause an ML-systems engineer could reach by asking the \
right questions, plus a plausible distractor the customer is fixated on.

Difficulty by level: beginner = one clear lever; intermediate = a red herring the customer \
believes; advanced = interacting causes + a business constraint that rules out the obvious fix.

Output ONLY the customer's message + business context. Do NOT reveal the root cause, the \
metrics to request, or the fix."""

FORWARD_JUDGE_SYSTEM = """You are a rigorous staff forward-deployed engineer grading how a \
candidate handled a customer's vague problem. Forward-deployed work is NOT just backend skill \
— it is turning ambiguity into a measured problem, asking for the RIGHT signals, localizing \
the bottleneck, iterating on hypotheses with evidence, proposing a fix that respects the \
customer's latency/quality/COST/SLA constraints, and explaining it to a non-expert without \
hand-waving or jargon-dumping.

Grade each of the 7 sub-skills 0-100. Reward: reframing the vague complaint into something \
measurable; asking for the specific metrics that would discriminate causes (p50/p99, tokens/s, \
GPU util vs useful work, queue depth, KV/memory, request mix, context lengths) BEFORE guessing; \
naming the real bottleneck and the customer's distractor; tying the fix to dollars and SLA; and \
clear stakeholder communication. Penalize: jumping to a fix before measuring, chasing the \
customer's pet theory, ignoring cost/SLA, and either talking down to or over the customer.

`improved_answer` = how a strong forward-deployed engineer would have handled it: the questions \
they'd ask, the metric that localizes the cause, the fix, its cost/SLA tradeoff, and the \
one-paragraph explanation they'd give the customer. Return ONLY the structured evaluation; \
scores must discriminate — do not cluster at 70."""

FORWARD_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer", "description": "0-100"},
        "category_scores": {
            "type": "object",
            "properties": {c: {"type": "integer", "description": "0-100"} for c in FORWARD_CATEGORIES},
            "required": FORWARD_CATEGORIES,
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
        "improved_answer": {"type": "string", "description": "how a strong forward-deployed engineer handles it"},
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

Prefer the canonical real-world incident patterns: (1) long-context p99 latency regression as \
conversations grow (prefill/decode + KV-cache); (2) high GPU utilization but low useful \
throughput (batching/memory stalls/queueing/stragglers); (3) OOM / capacity collapse during a \
traffic burst (KV-cache sizing, admission control, autoscaling, fallback); (4) quality \
regression after a model/quantization change (eval harness, rollback, A/B); (5) p99 regression \
after a kernel/driver/serving-engine upgrade (observability, canary, rollback).

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

Strong on-call engineers: form hypotheses before guessing; inspect the RIGHT metrics first; \
separate immediate mitigation from root-cause analysis; protect the customer experience first; \
and know when to roll back. Reward that discipline; penalize random fixes and metric tunnel-vision.

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
