"""Streamlit demo app for Learning Memory OS.

Run: uv run streamlit run scripts/app.py

Architecture note: the routing engine is called twice per turn — once inside
TutorAgent.answer() and once here to retrieve the RoutingDecision for display.
Both calls receive identical inputs so results are deterministic. A cleaner fix
would expose `decision` from AgentResponse, but that's deferred to keep the
agent API stable.
"""

import re
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
from learning_memory_os.eval.quiz import QuizQuestion, score_answer

# Optional mermaid renderer — fall back gracefully if unavailable
try:
    from streamlit_mermaid import st_mermaid
    _MERMAID_OK = True
except ImportError:
    _MERMAID_OK = False


st.set_page_config(page_title="Learning Memory OS", layout="wide")

# ---------------------------------------------------------------------------
# Quiz generation system prompt
# ---------------------------------------------------------------------------

QUIZ_GEN_SYSTEM = (
    "Generate ONE substantive quiz question about the given ML systems engineering topic. "
    "Output STRICT minified JSON on a single line with two keys: "
    'question (string), rubric (string describing what a correct answer must contain). '
    "Do NOT include any commentary, prose, code fences, or explanation outside the JSON object. "
    "Output ONLY the JSON object. "
    "All string values must be a single line. "
    "Inside string values, never use unescaped double quotes — use single quotes or 'these' for sub-quoting. "
    "Do not use smart/curly quotes; only straight ASCII quotes (\")."
)

# ---------------------------------------------------------------------------
# Starter prompts per topic
# ---------------------------------------------------------------------------

_STARTER_PROMPTS: dict[str, list[str]] = {
    "kv_cache": [
        "What is a KV cache and why does it exist in transformers?",
        "How does KV cache memory scale with batch size and sequence length?",
        "What are the main strategies for compressing or evicting KV cache entries?",
        "Draw me a diagram showing how KV cache fits into the decode loop.",
    ],
    "context_selection": [
        "What problem does context selection solve for RAG systems?",
        "How does relevance scoring differ from recency scoring in context selection?",
        "What happens when the context budget is exceeded?",
        "Show me the trade-offs between dense retrieval and BM25 for context selection.",
    ],
    "quantization": [
        "What is quantization and why does it matter for LLM inference?",
        "What's the difference between INT8 and INT4 quantization?",
        "How does GPTQ differ from AWQ?",
        "What accuracy loss should I expect from 4-bit quantization on a large model?",
    ],
    "agent_memory": [
        "What are the different tiers of memory an AI agent can use?",
        "How does episodic memory differ from semantic memory in an agent?",
        "What are common failure modes when agents lose context mid-task?",
        "Show me how memory is read and written during an agent turn.",
    ],
    "pretraining_data": [
        "What makes a high-quality pretraining dataset?",
        "How does data deduplication affect downstream model quality?",
        "What is the Chinchilla scaling law and what does it say about data?",
        "What ethical issues arise from pretraining data collection?",
    ],
    "scaling_laws": [
        "What do scaling laws predict about model performance?",
        "What is the Chinchilla result and why did it change how we train models?",
        "How do compute-optimal scaling laws differ from earlier power laws?",
        "What happens to scaling laws at very large scales — do they hold?",
    ],
    "transformer_architecture": [
        "Explain the transformer architecture in one paragraph.",
        "Why is multi-head attention better than single-head attention?",
        "How does positional encoding work and what are the alternatives?",
        "Draw a diagram of one transformer decoder block.",
    ],
    "attention_moe": [
        "What is Mixture-of-Experts (MoE) and how does it scale parameters efficiently?",
        "How does sparse attention differ from full attention?",
        "What are the trade-offs of routing in MoE models?",
        "How does FlashAttention speed up the attention computation?",
    ],
    "speculative_decoding": [
        "What is speculative decoding and why is it faster?",
        "What makes a good draft model for speculative decoding?",
        "What are the acceptance rate trade-offs in speculative decoding?",
        "Draw a sequence diagram of one speculative decoding step.",
    ],
    "continuous_batching": [
        "What problem does continuous batching solve over static batching?",
        "How does PagedAttention enable continuous batching?",
        "What is the throughput vs latency trade-off in continuous batching?",
        "How does vLLM implement continuous batching internally?",
    ],
}

_GENERIC_PROMPTS = [
    "Explain the core idea of this topic in plain English.",
    "What's the most common misconception about this topic?",
    "Give me a concrete example of how this is used in practice.",
    "What should I learn next after understanding this topic?",
]


def _starter_prompts_for(topic_id: str | None) -> list[str]:
    if topic_id and topic_id in _STARTER_PROMPTS:
        return _STARTER_PROMPTS[topic_id]
    return _GENERIC_PROMPTS


# ---------------------------------------------------------------------------
# Citation rendering
# ---------------------------------------------------------------------------

_HEX8_RE = re.compile(r"\[([0-9a-f]{8})\]", re.IGNORECASE)


def _render_citations(text: str, selected_items: list[dict]) -> str:
    """Replace [hex8id] citations with numbered [1] [2] ... and append a References section."""
    id_to_num: dict[str, int] = {}
    counter = [0]

    def _replace(match: re.Match) -> str:
        raw_id = match.group(1).lower()
        if raw_id not in id_to_num:
            counter[0] += 1
            id_to_num[raw_id] = counter[0]
        return f"[{id_to_num[raw_id]}]"

    rendered = _HEX8_RE.sub(_replace, text)

    if id_to_num and selected_items:
        # Build a lookup from first-8-chars to title
        item_titles: dict[str, str] = {}
        for it in selected_items:
            item_id = it.get("id", "")
            short = item_id[:8].lower() if item_id else ""
            if short:
                item_titles[short] = it.get("title", item_id[:8])

        refs_lines = ["\n\n---\n**References:**"]
        for raw_id, num in sorted(id_to_num.items(), key=lambda kv: kv[1]):
            title = item_titles.get(raw_id, raw_id)
            refs_lines.append(f"**[{num}]** {title}")
        rendered += "\n".join(refs_lines)

    return rendered


# ---------------------------------------------------------------------------
# Mermaid-aware renderer
# ---------------------------------------------------------------------------

_MERMAID_FENCE_RE = re.compile(
    r"```mermaid\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE
)


def _render_with_mermaid(text: str) -> None:
    """Render text that may contain ```mermaid blocks.

    Non-mermaid chunks go to st.markdown; mermaid blocks go to st_mermaid
    (or a plain code block if the library is unavailable).
    """
    parts = _MERMAID_FENCE_RE.split(text)
    # split() with one group: [before, diagram1, after_diagram1, diagram2, ...]
    # even indices = plain text, odd indices = mermaid diagram code
    for i, part in enumerate(parts):
        if not part:
            continue
        if i % 2 == 0:
            # Plain text chunk
            st.markdown(part)
        else:
            # Mermaid diagram
            if _MERMAID_OK:
                try:
                    st_mermaid(part.strip(), height=350)
                except Exception:
                    st.code(part.strip(), language="mermaid")
            else:
                st.code(part.strip(), language="mermaid")


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
    # Track which concept ids have been selected per topic this session
    if "seen_concepts_by_topic" not in st.session_state:
        st.session_state.seen_concepts_by_topic = {}  # topic_id -> set[concept_id]
    # Quiz state: keyed by message index
    if "quiz_state" not in st.session_state:
        st.session_state.quiz_state = {}  # msg_idx -> {"question":..., "rubric":..., "answer":..., "score":...}
    # Pending starter prompt (set when a chip is clicked)
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None


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
            st.session_state.seen_concepts_by_topic = {}
            st.session_state.quiz_state = {}
            st.session_state.pending_prompt = None
            st.rerun()
        if c2.button("Cancel"):
            st.session_state.confirm_clear = False
            st.rerun()

    return student_id, topic_id, budget


# ---------------------------------------------------------------------------
# Right routing panel
# ---------------------------------------------------------------------------


def _render_routing_panel(col, topic_id: str | None):
    """Render the routing panel in the right column."""

    d = st.session_state.last_decision

    # --- Concepts touched this turn (always visible) ---
    col.subheader("Concepts touched this turn")
    if d is None:
        col.caption("Ask a question to see what the routing engine selected.")
    else:
        if d["selected"]:
            for it in d["selected"]:
                col.write(f"- {it['title']}")
        else:
            col.caption("(no items selected — candidates list was empty)")

        # --- Progress stat ---
        col.subheader("Your progress")
        seen_ids = st.session_state.seen_concepts_by_topic.get(topic_id or "_global", set())
        n_seen = len(seen_ids)

        # Count total concept-type artifacts for this topic
        n_total = 0
        if topic_id:
            conn = _new_conn()
            try:
                semantic = SemanticStore(conn)
                all_items = semantic.by_topic(topic_id)
                n_total = sum(1 for it in all_items if it.artifact_type == "concept")
            finally:
                conn.close()

        topic_label = topic_id or "global"
        if n_total > 0:
            col.write(f"Topic: **{topic_label}**  •  Concepts covered: **{n_seen} / {n_total}**")
            col.progress(min(n_seen / n_total, 1.0))
        else:
            col.write(f"Topic: **{topic_label}**  •  Concepts covered: **{n_seen}**")

    # --- Collapsed diagnostics ---
    with col.expander("🔍 How this answer was built", expanded=False):
        if d is None:
            st.caption("No routing data yet.")
        else:
            n_sel = len(d["selected"])
            n_total_items = n_sel + len(d["dropped"])
            st.metric("Items selected", f"{n_sel} / {n_total_items}")
            st.metric("Tokens used", f"{d['tokens_used']} / {d['budget']}")

            st.markdown("**Selected items (with scores)**")
            if not d["selected"]:
                st.caption("(no items selected)")
            for it in d["selected"]:
                score = d["scores"].get(it["id"])
                label = f"`{it['id'][:8]}` — {it['title'][:48]}"
                with st.expander(label):
                    if score:
                        st.write(
                            f"**total {score['total']:.3f}** = "
                            f"rel {score['relevance']:.2f} + "
                            f"rec {score['recency']:.2f} + "
                            f"misc {score['misconception']:.2f} + "
                            f"prereq {score['prerequisite']:.2f} + "
                            f"reuse {score['reuse']:.2f}"
                        )
                    st.caption(it["body"][:300] + ("..." if len(it["body"]) > 300 else ""))

            if d["dropped"]:
                st.markdown("**Dropped (over budget)**")
                dropped_sorted = sorted(
                    d["dropped"],
                    key=lambda x: d["scores"].get(x["id"], {}).get("total", 0.0),
                    reverse=True,
                )[:5]
                for it in dropped_sorted:
                    score = d["scores"].get(it["id"])
                    label = f"`{it['id'][:8]}` — {it['title'][:48]}"
                    with st.expander(label):
                        if score:
                            st.write(
                                f"total **{score['total']:.3f}** — "
                                f"would cost {it['token_estimate']} tokens"
                            )
                        st.caption(it["body"][:200] + ("..." if len(it["body"]) > 200 else ""))


# ---------------------------------------------------------------------------
# Progress bar (shown below welcome / above first message)
# ---------------------------------------------------------------------------


def _render_progress_bar(topic_id: str | None):
    seen_ids = st.session_state.seen_concepts_by_topic.get(topic_id or "_global", set())
    n_seen = len(seen_ids)

    n_total = 0
    if topic_id:
        conn = _new_conn()
        try:
            semantic = SemanticStore(conn)
            all_items = semantic.by_topic(topic_id)
            n_total = sum(1 for it in all_items if it.artifact_type == "concept")
        finally:
            conn.close()

    if n_total > 0:
        fraction = min(n_seen / n_total, 1.0)
        st.progress(fraction)
        st.caption(f"Concepts touched in this topic: {n_seen} / {n_total}")
    elif n_seen > 0:
        st.caption(f"Concepts touched this session: {n_seen}")


# ---------------------------------------------------------------------------
# Quiz button + flow
# ---------------------------------------------------------------------------


def _render_quiz_for_message(msg_idx: int, topic_id: str | None, student_id: str):
    """Render the 'Test yourself' button and quiz flow for a given assistant message index."""
    quiz_key = str(msg_idx)
    state = st.session_state.quiz_state.get(quiz_key, {})

    if not state:
        if st.button("🎯 Test yourself", key=f"quiz_btn_{msg_idx}"):
            with st.spinner("Generating a quiz question..."):
                llm, _ = _llm_and_embedder()
                topic_label = topic_id or "ML systems engineering"
                try:
                    result = llm.complete_json(
                        system=QUIZ_GEN_SYSTEM,
                        user=f"Topic: {topic_label}",
                        max_tokens=256,
                    )
                    q_text = result.get("question", "")
                    rubric = result.get("rubric", "")
                except Exception as exc:
                    st.error(f"Could not generate quiz question: {exc}")
                    return

            if q_text:
                st.session_state.quiz_state[quiz_key] = {
                    "question": q_text,
                    "rubric": rubric,
                    "answer": None,
                    "score": None,
                    "rationale": None,
                }
                st.rerun()
        return

    # Question is ready
    st.info(f"**Quiz:** {state['question']}")

    if state.get("score") is not None:
        # Already scored
        score_val = state["score"]
        bar_val = int(score_val * 100)
        color = "green" if score_val >= 0.7 else ("orange" if score_val >= 0.4 else "red")
        st.markdown(
            f"**Score: {bar_val}/100** — <span style='color:{color}'>{state['rationale']}</span>",
            unsafe_allow_html=True,
        )
        if st.button("Try another question", key=f"quiz_retry_{msg_idx}"):
            del st.session_state.quiz_state[quiz_key]
            st.rerun()
    else:
        with st.form(key=f"quiz_form_{msg_idx}"):
            answer = st.text_area("Your answer:", key=f"quiz_answer_{msg_idx}")
            submitted = st.form_submit_button("Submit")

        if submitted and answer.strip():
            with st.spinner("Grading..."):
                llm, _ = _llm_and_embedder()
                qq = QuizQuestion(
                    question=state["question"],
                    rubric=state["rubric"],
                )
                try:
                    quiz_score = score_answer(question=qq, student_answer=answer, judge_llm=llm)
                except Exception as exc:
                    st.error(f"Grading failed: {exc}")
                    return

            st.session_state.quiz_state[quiz_key].update(
                {"answer": answer, "score": quiz_score.score, "rationale": quiz_score.rationale}
            )

            # Persist as episodic event
            conn = _new_conn()
            try:
                episodic = EpisodicStore(conn)
                episodic.append(
                    student_id=student_id,
                    event_type="quiz_attempt",
                    payload={
                        "question": state["question"],
                        "answer": answer,
                        "score": quiz_score.score,
                        "rationale": quiz_score.rationale,
                        "topic_id": topic_id,
                    },
                )
                conn.commit()
            finally:
                conn.close()

            st.rerun()


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

        # Track seen concepts for progress bar
        topic_key = topic_id or "_global"
        if topic_key not in st.session_state.seen_concepts_by_topic:
            st.session_state.seen_concepts_by_topic[topic_key] = set()
        for it in decision.selected:
            if getattr(it, "artifact_type", None) == "concept":
                st.session_state.seen_concepts_by_topic[topic_key].add(it.id)

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
# Welcome screen
# ---------------------------------------------------------------------------


def _render_welcome(topic_id: str | None):
    topic_label = topic_id if topic_id else "any ML systems topic"
    st.markdown(
        f"""
<div style="
    background: #f5f7fb;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
    border-left: 4px solid #5b6cff;
">
<h2 style="margin-top:0; color:#1f2235;">👋 Welcome — I'm your ML systems tutor.</h2>
<p style="color:#444; font-size:1.05rem;">
    Pick a topic on the left, then ask anything — or click one of the starter prompts below to dive in.
    I'll give you a concise answer, a diagram when helpful, and always invite you to go deeper.
</p>
<p style="color:#888; font-size:0.9rem; margin-bottom:0;">
    Currently focused on: <strong>{topic_label}</strong>
</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Starter prompt chips
    starters = _starter_prompts_for(topic_id)
    st.markdown("**Jump in with:**")
    cols = st.columns(len(starters))
    for i, (col, prompt) in enumerate(zip(cols, starters)):
        if col.button(prompt, key=f"starter_{i}"):
            st.session_state.pending_prompt = prompt
            st.rerun()


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
        # Welcome screen when chat is empty
        if not st.session_state.messages:
            _render_welcome(topic_id)
            _render_progress_bar(topic_id)

        # Replay chat history (with citation rendering + mermaid + quiz)
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    # Re-derive selected items from last_decision for citation rendering
                    # For replayed messages we only have the raw text; use stored decision
                    # for the most recent assistant turn, empty list for earlier turns.
                    selected_items = []
                    if idx == len(st.session_state.messages) - 1 and st.session_state.last_decision:
                        selected_items = st.session_state.last_decision.get("selected", [])
                    rendered = _render_citations(msg["content"], selected_items)
                    _render_with_mermaid(rendered)
                    # Quiz button below each assistant message
                    _render_quiz_for_message(idx, topic_id, student_id)
                else:
                    st.markdown(msg["content"])

        # Handle pending starter prompt (chip click)
        active_prompt: str | None = None
        if st.session_state.pending_prompt:
            active_prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None

        prompt = st.chat_input("Ask the tutor a question...")
        if active_prompt:
            prompt = active_prompt

        if prompt:
            # Show user message immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display tutor response
            with st.chat_message("assistant"):
                with st.spinner("Selecting context and generating response..."):
                    reply = _handle_turn(prompt, student_id, topic_id, budget)
                selected_items = st.session_state.last_decision.get("selected", []) if st.session_state.last_decision else []
                rendered_reply = _render_citations(reply, selected_items)
                _render_with_mermaid(rendered_reply)
                # Quiz button for this new message
                new_msg_idx = len(st.session_state.messages)  # will be the index after append
                _render_quiz_for_message(new_msg_idx, topic_id, student_id)

            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

    with right_col:
        _render_routing_panel(right_col, topic_id)


if __name__ == "__main__":
    main()
