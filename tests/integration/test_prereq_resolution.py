"""Integration test: prerequisite_titles get populated from the real DB."""

from pathlib import Path
from learning_memory_os.ingestion.topic_loader import (
    load_topics,
    resolve_prerequisite_titles,
)


def test_prereq_titles_for_scaling_laws(db_conn):
    topics = load_topics(Path("config/topics.yaml"))
    titles = resolve_prerequisite_titles(
        db_conn, topic_id="scaling_laws", topics=topics
    )
    # scaling_laws has prerequisites: [resource_accounting]
    # resource_accounting has 46 artifacts in the DB; the concept-typed ones should be plenty
    assert len(titles) >= 3, f"Expected >=3 prereq titles, got: {titles}"


def test_prereq_titles_empty_for_root_topic(db_conn):
    topics = load_topics(Path("config/topics.yaml"))
    # tokenization has prerequisites: []
    titles = resolve_prerequisite_titles(
        db_conn, topic_id="tokenization", topics=topics
    )
    assert titles == set()
