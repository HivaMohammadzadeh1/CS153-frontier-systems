"""Streamlit demo app for Learning Memory OS.

Run: uv run streamlit run scripts/app.py

Architecture note: the routing engine is called twice per turn — once inside
TutorAgent.answer() and once here to retrieve the RoutingDecision for display.
Both calls receive identical inputs so results are deterministic. A cleaner fix
would expose `decision` from AgentResponse, but that's deferred to keep the
agent API stable.
"""

from collections import Counter
from pathlib import Path

import streamlit as st

from learning_memory_os.config import get_settings
from learning_memory_os.llm import LLM
from learning_memory_os.embeddings import Embedder
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.semantic import SemanticStore
from learning_memory_os.memory.student import StudentStore
from learning_memory_os.memory.episodic import EpisodicStore
from learning_memory_os.selector.engine import RoutingEngine
from learning_memory_os.agents.tutor import TutorAgent
from learning_memory_os.logging_utils.interactions import InteractionLogger
from learning_memory_os.ingestion.topic_loader import load_topics, resolve_prerequisite_titles


st.set_page_config(page_title="Learning Memory OS", layout="wide")


# ---------------------------------------------------------------------------
# Cached singletons (one per server process, shared across reruns/sessions)
# ---------------------------------------------------------------------------


@st.cache_resource
def _settings():
    return get_settings()


@st.cache_resource
def _topics():
    return load_topics(Path("config/topics.yaml"))


@st.cache_resource
def _llm_and_embedder():
    s = _settings()
    return LLM(api_key=s.anthropic_api_key), Embedder(api_key=s.openai_api_key)


def _new_conn():
    """Open a short-lived DB connection (used in try/finally blocks)."""
    return connect(_settings().database_url)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[{"role": str, "content": str}]
    if "last_decision" not in st.session_state:
        st.session_state.last_decision = None  # dict filled after each turn
    if "reuse_counts" not in st.session_state:
        st.session_state.reuse_counts = Counter()
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False


# ---------------------------------------------------------------------------
# Left sidebar
# ---------------------------------------------------------------------------


def _render_left_sidebar(student_id_default: str = "demo-user"):
    st.sidebar.title("Learning Memory OS")
    st.sidebar.caption("CS 153 — context-routed tutor demo")

    student_id = st.sidebar.text_input("Student ID", value=student_id_default)

    topics = _topics()
    topic_options = ["(global vector search)"] + [t.id for t in topics]
    topic_choice = st.sidebar.selectbox("Topic focus", topic_options, index=0)
    topic_id = None if topic_choice == "(global vector search)" else topic_choice

    budget = st.sidebar.slider(
        "Token budget", min_value=1000, max_value=32000, value=3000, step=500
    )

    st.sidebar.divider()

    # Student mastery + misconceptions (short-lived connection, read-only)
    conn = _new_conn()
    try:
        student_store = StudentStore(conn)
        student_store.ensure_student(student_id)
        conn.commit()
        mastery = student_store.mastery_for(student_id)
        misconceptions = student_store.active_misconceptions(student_id)
    finally:
        conn.close()

    st.sidebar.subheader("Mastery state")
    if mastery:
        for m in mastery[:8]:
            st.sidebar.write(f"`{m.concept_id[:8]}` {m.score:.2f}")
    else:
        st.sidebar.caption("(no mastery recorded yet)")

    st.sidebar.subheader("Active misconceptions")
    if misconceptions:
        for m in misconceptions[:5]:
            desc = m["description"] or ""
            st.sidebar.write(f"- {desc[:80]}{'...' if len(desc) > 80 else ''}")
    else:
        st.sidebar.caption("(none)")

    st.sidebar.divider()

    # Clear chat — two-step confirm
    if not st.session_state.confirm_clear:
        if st.sidebar.button("Clear chat"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.sidebar.warning("This will clear all chat history. Confirm?")
        c1, c2 = st.sidebar.columns(2)
        if c1.button("Yes, clear"):
            st.session_state.messages = []
            st.session_state.last_decision = None
            st.session_state.reuse_counts = Counter()
            st.session_state.confirm_clear = False
            st.rerun()
        if c2.button("Cancel"):
            st.session_state.confirm_clear = False
            st.rerun()

    return student_id, topic_id, budget


# ---------------------------------------------------------------------------
# Right routing panel
# ---------------------------------------------------------------------------


def _render_routing_panel(col):
    """Render the observable routing panel in the right column."""
    col.subheader("Routing — last turn")

    d = st.session_state.last_decision
    if d is None:
        col.caption("Ask a question to see what the routing engine selected.")
        return

    n_sel = len(d["selected"])
    n_total = n_sel + len(d["dropped"])
    col.metric("Items selected", f"{n_sel} / {n_total}")
    col.metric("Tokens used", f"{d['tokens_used']} / {d['budget']}")

    col.markdown("**Selected items**")
    if not d["selected"]:
        col.caption("(no items selected — candidates list was empty)")
    for it in d["selected"]:
        score = d["scores"].get(it["id"])
        label = f"`{it['id'][:8]}` — {it['title'][:48]}"
        with col.expander(label):
            if score:
                col.write(
                    f"**total {score['total']:.3f}** = "
                    f"rel {score['relevance']:.2f} + "
                    f"rec {score['recency']:.2f} + "
                    f"misc {score['misconception']:.2f} + "
                    f"prereq {score['prerequisite']:.2f} + "
                    f"reuse {score['reuse']:.2f}"
                )
            col.caption(it["body"][:300] + ("..." if len(it["body"]) > 300 else ""))

    if d["dropped"]:
        col.markdown("**Dropped (over budget)**")
        dropped_sorted = sorted(
            d["dropped"],
            key=lambda x: d["scores"].get(x["id"], {}).get("total", 0.0),
            reverse=True,
        )[:5]
        for it in dropped_sorted:
            score = d["scores"].get(it["id"])
            label = f"`{it['id'][:8]}` — {it['title'][:48]}"
            with col.expander(label):
                if score:
                    col.write(
                        f"total **{score['total']:.3f}** — "
                        f"would cost {it['token_estimate']} tokens"
                    )
                col.caption(it["body"][:200] + ("..." if len(it["body"]) > 200 else ""))


# ---------------------------------------------------------------------------
# Turn handler
# ---------------------------------------------------------------------------


def _handle_turn(
    prompt: str, student_id: str, topic_id: str | None, budget: int
) -> str:
    """Run one tutor turn; update session_state.last_decision and reuse_counts."""
    llm, embedder = _llm_and_embedder()
    engine = RoutingEngine()
    settings = _settings()
    log_path = settings.log_dir / "interactions.jsonl"
    logger = InteractionLogger(path=log_path)

    conn = _new_conn()
    try:
        student_store = StudentStore(conn)
        student_store.ensure_student(student_id)
        semantic = SemanticStore(conn)
        episodic = EpisodicStore(conn)
        topics_cfg = _topics()

        # Retrieve candidates
        if topic_id:
            candidates = semantic.by_topic(topic_id)
        else:
            q_emb = embedder.embed_one(prompt)
            candidates = semantic.vector_search(query=q_emb, k=20)

        misconceptions_list = student_store.active_misconceptions(student_id)
        misconceptions = {m["id"] for m in misconceptions_list}

        prereq_titles: set[str] = set()
        if topic_id:
            prereq_titles = resolve_prerequisite_titles(
                conn, topic_id=topic_id, topics=topics_cfg
            )

        recent = episodic.recent(student_id, limit=10)
        recent_ids = {e.id for e in recent if e.id}

        # Run the tutor agent (internally calls routing engine + LLM)
        tutor = TutorAgent(llm=llm, engine=engine, embedder=embedder, logger=logger)
        response = tutor.answer(
            student_id=student_id,
            question=prompt,
            candidates=candidates,
            active_misconceptions=misconceptions,
            prerequisites=prereq_titles,
            recent_ids=recent_ids,
            reuse_counts=dict(st.session_state.reuse_counts),
            budget=budget,
        )

        # Re-run the routing engine with the same inputs to get the RoutingDecision
        # for display purposes. This is deterministic — results are identical to what
        # TutorAgent computed internally. The wart: we embed twice per turn.
        task_emb = embedder.embed_one(prompt)
        decision = engine.route(
            candidates=candidates,
            task_embedding=task_emb,
            active_misconceptions=misconceptions,
            prerequisites=prereq_titles,
            recent_ids=recent_ids,
            reuse_counts=dict(st.session_state.reuse_counts),
            budget=budget,
        )

        # Update reuse counts
        for it in response.selected_items:
            st.session_state.reuse_counts[it.id] += 1

        # Persist episodic events
        episodic.append(
            student_id=student_id,
            event_type="question",
            payload={
                "text": prompt,
                "topic_id": topic_id,
                "source": "streamlit_app",
            },
        )
        episodic.append(
            student_id=student_id,
            event_type="tutor_reply",
            payload={
                "text": response.text,
                "selected_ids": [it.id for it in response.selected_items],
                "tokens_used": response.tokens_used,
            },
        )
        conn.commit()

        # Serialize decision into session-state-friendly plain dicts
        st.session_state.last_decision = {
            "selected": [
                {
                    "id": it.id,
                    "title": it.title,
                    "body": it.body,
                    "token_estimate": it.token_estimate,
                }
                for it in decision.selected
            ],
            "dropped": [
                {
                    "id": it.id,
                    "title": it.title,
                    "body": it.body,
                    "token_estimate": it.token_estimate,
                }
                for it in decision.dropped
            ],
            "scores": {
                item_id: {
                    "relevance": s.relevance,
                    "recency": s.recency,
                    "misconception": s.misconception,
                    "prerequisite": s.prerequisite,
                    "reuse": s.reuse,
                    "total": s.total,
                }
                for item_id, s in decision.scores.items()
            },
            "budget": decision.budget,
            "tokens_used": decision.tokens_used,
        }

        return response.text

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    _init_session()

    # Header
    st.title("Learning Memory OS — tutor demo")
    st.caption(
        "CS 153 final project: context-routed tutor with observable routing decisions."
    )

    # Left sidebar (returns user controls)
    student_id, topic_id, budget = _render_left_sidebar()

    # Split into main chat pane (wider) and right routing panel
    main_col, right_col = st.columns([3, 2])

    with main_col:
        # Replay chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask the tutor a question...")
        if prompt:
            # Show user message immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display tutor response
            with st.chat_message("assistant"):
                with st.spinner("Selecting context and generating response..."):
                    reply = _handle_turn(prompt, student_id, topic_id, budget)
                st.markdown(reply)

            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

    with right_col:
        _render_routing_panel(right_col)


if __name__ == "__main__":
    main()
