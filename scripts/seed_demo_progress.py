"""Seed synthetic Postgres mastery + misconceptions for a demo student.

Populates the `mastery` and `misconceptions` tables so the Profile tab shows
realistic-looking topic-by-topic progress instead of all zeros. Designed to
pair with `seed_xtrace_memory.py` — together they make a coherent demo of a
student who's been working with the tutor for a while.

Run:
    uv run python scripts/seed_demo_progress.py           # seeds demo-user
    uv run python scripts/seed_demo_progress.py alice     # seeds 'alice'
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict

from learning_memory_os.config import get_settings
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.semantic import SemanticStore
from learning_memory_os.memory.student import StudentStore


# (topic_id, target_band, n_concepts_to_assess)
#   band: ("good", lo, hi), ("mid", lo, hi), ("low", lo, hi)
TOPIC_PLAN = [
    ("tokenization",            "good", 0.85, 0.95, 3),
    ("transformer_architecture","good", 0.78, 0.88, 5),
    ("attention_moe",           "mid",  0.55, 0.65, 4),
    ("resource_accounting",     "good", 0.80, 0.92, 4),
    ("data_parallelism",        "good", 0.82, 0.92, 3),
    ("sharded_training",        "mid",  0.50, 0.62, 3),
    ("gpu_kernels",             "low",  0.20, 0.35, 3),
    ("model_parallelism",       "mid",  0.45, 0.58, 3),
]

# Misconceptions written as if extracted from real tutor sessions.
MISCONCEPTIONS = [
    {
        "description": (
            "Believes INT8 quantization always hurts model quality. Hasn't internalized "
            "that per-channel weight quantization with percentile calibration achieves "
            "<0.5% degradation on modern LLMs."
        ),
        "evidence": "Said: 'I always thought INT8 significantly hurts quality, is that wrong?'",
    },
    {
        "description": (
            "Confuses inference latency and throughput. Treats them as the same metric "
            "instead of as a per-request vs per-second trade-off mediated by batch size."
        ),
        "evidence": "Asked which to optimize for a chat app without distinguishing them.",
    },
    {
        "description": (
            "Mixes up data parallelism gradient sync race conditions with optimizer "
            "state divergence — both produce diverging loss but require different fixes."
        ),
        "evidence": "Loss-diverges-after-epoch-3 question on PyTorch DDP setup.",
    },
]


def main() -> int:
    s = get_settings()
    student_id = sys.argv[1] if len(sys.argv) > 1 else "demo-user"
    random.seed(42)  # deterministic demo

    conn = connect(s.database_url)
    semantic = SemanticStore(conn)
    student = StudentStore(conn)

    student.ensure_student(student_id)
    conn.commit()

    print(f"Seeding mastery + misconceptions for '{student_id}'")

    summary = defaultdict(list)
    n_inserted = 0
    for topic_id, band, lo, hi, n in TOPIC_PLAN:
        try:
            items = semantic.by_topic(topic_id)
        except Exception as exc:
            print(f"  {topic_id}: error fetching items — {exc}")
            continue
        concepts = [
            it for it in items
            if (getattr(it.artifact_type, "value", str(it.artifact_type or "")).lower()
                == "concept")
        ]
        if not concepts:
            print(f"  {topic_id}: no concept-type items in DB, skipping")
            continue
        random.shuffle(concepts)
        chosen = concepts[: min(n, len(concepts))]
        scores: list[float] = []
        for c in chosen:
            score = round(random.uniform(lo, hi), 2)
            confidence = round(random.uniform(0.6, 0.85), 2)
            student.update_mastery(
                student_id=student_id,
                concept_id=c.id,
                score=score,
                confidence=confidence,
            )
            scores.append(score)
            n_inserted += 1
        conn.commit()
        mean = sum(scores) / len(scores)
        summary[band].append(f"{topic_id} ({len(scores)} concepts, mean {mean:.2f})")
        print(f"  {topic_id:<28s} {band:<5s}  {len(scores)} concepts  mean {mean:.2f}")

    print(f"\nMastery rows inserted: {n_inserted}")
    print("Topics by band:")
    for band, topics in summary.items():
        print(f"  {band}: {len(topics)} topics")
        for t in topics:
            print(f"    - {t}")

    print(f"\nSeeding {len(MISCONCEPTIONS)} misconceptions:")
    for m in MISCONCEPTIONS:
        student.record_misconception(
            student_id=student_id,
            concept_id=None,
            description=m["description"],
            evidence=m["evidence"],
        )
        conn.commit()
        print(f"  + {m['description'][:80]}...")

    conn.close()
    print("\nDone. Hard-reload the Profile tab to see populated metrics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
