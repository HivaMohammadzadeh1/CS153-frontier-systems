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


st.set_page_config(
    page_title="Learning Memory OS",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

DIAGNOSTIC_THRESHOLD = 0.6   # scores below this trigger the diagnostic flow
DIAGNOSTIC_MAX_TURNS = 3     # max back-and-forth turns in diagnostic loop

# ---------------------------------------------------------------------------
# Tool-use schemas
# ---------------------------------------------------------------------------

QUIZ_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "A substantive quiz question for the student."},
        "rubric": {"type": "string", "description": "What a correct answer to this question must contain."},
    },
    "required": ["question", "rubric"],
}

DIAGNOSTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {
            "type": "string",
            "description": "One sentence guessing what the student misunderstands.",
        },
        "follow_up_question": {
            "type": "string",
            "description": "A probing question that, if answered correctly, confirms or refutes the diagnosis.",
        },
    },
    "required": ["diagnosis", "follow_up_question"],
}

EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "confirmed_misconception": {"type": "string"},
        "explanation": {"type": "string"},
        "next_action": {
            "type": "string",
            "enum": ["explain", "re_test", "wrap_up"],
        },
        "next_message": {"type": "string"},
    },
    "required": ["confirmed_misconception", "explanation", "next_action", "next_message"],
}

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

        list_items = "".join(
            f"<li>[{num}] {item_titles.get(raw_id, raw_id)}</li>"
            for raw_id, num in sorted(id_to_num.items(), key=lambda kv: kv[1])
        )
        refs_html = (
            f'\n\n<details class="lm-refs">'
            f"<summary>Sources ▾</summary>"
            f"<ol>{list_items}</ol>"
            f"</details>"
        )
        rendered += refs_html

    return rendered


# ---------------------------------------------------------------------------
# Mermaid-aware renderer
# ---------------------------------------------------------------------------

_MERMAID_FENCE_RE = re.compile(
    r"```mermaid\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE
)


def _render_with_mermaid(text: str) -> None:
    """Render text that may contain ```mermaid blocks."""
    parts = _MERMAID_FENCE_RE.split(text)
    for i, part in enumerate(parts):
        if not part:
            continue
        if i % 2 == 0:
            st.markdown(part)
        else:
            if _MERMAID_OK:
                try:
                    st_mermaid(part.strip(), height=350)
                except Exception:
                    st.code(part.strip(), language="mermaid")
            else:
                st.code(part.strip(), language="mermaid")


# ---------------------------------------------------------------------------
# Cached singletons
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


@st.cache_resource
def _artifact_count() -> int:
    """Total number of semantic artifacts in the DB (fetched once at startup)."""
    conn = _new_conn()
    try:
        semantic = SemanticStore(conn)
        # Use a broad vector search with zero embedding to count artifacts
        # Fall back to counting topics * average
        topics = _topics()
        total = 0
        for t in topics:
            try:
                items = semantic.by_topic(t.id)
                total += len(items)
            except Exception:
                pass
        return total
    except Exception:
        return 0
    finally:
        conn.close()


def _new_conn():
    """Open a short-lived DB connection (used in try/finally blocks)."""
    return connect(_settings().database_url)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_decision" not in st.session_state:
        st.session_state.last_decision = None
    if "reuse_counts" not in st.session_state:
        st.session_state.reuse_counts = Counter()
    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False
    if "seen_concepts_by_topic" not in st.session_state:
        st.session_state.seen_concepts_by_topic = {}
    if "quiz_state" not in st.session_state:
        st.session_state.quiz_state = {}
    if "diagnostic" not in st.session_state:
        st.session_state.diagnostic = {}  # msg_idx -> diagnostic state dict
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "show_context_analysis" not in st.session_state:
        st.session_state.show_context_analysis = False


# ---------------------------------------------------------------------------
# Global CSS injection
# ---------------------------------------------------------------------------


def _inject_css():
    st.markdown(
        """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ---- Typography ---- */
html, body, [class*="css"], [data-testid="stMarkdownContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ---- Base ---- */
.stApp { background: #f7f8fb; }

/* ---- Rounded borders on containers ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    box-shadow: 0 1px 2px rgba(17, 24, 39, 0.05);
}

/* ---- Chat message bubbles ---- */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border-radius: 12px;
    padding: 10px 14px !important;
    margin-bottom: 10px;
    line-height: 1.55;
}
[data-testid="stChatMessage"][aria-label*="user"] {
    background: #f1f3f5 !important;
}
[data-testid="stChatMessage"][aria-label*="assistant"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb;
    box-shadow: 0 1px 2px rgba(17,24,39,0.05);
}

/* ---- Cards (quiz / diag) ---- */
.lm-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px 18px;
    margin: 12px 0;
    box-shadow: 0 1px 2px rgba(17, 24, 39, 0.05);
}
.lm-quiz-card  { border-left: 4px solid #5b6cff; }
.lm-diag-card  { border-left: 4px solid #f59e0b; }
.lm-card-header { font-weight: 600; font-size: 16px; color: #111827; margin-bottom: 8px; }
.lm-card-sub    { color: #6b7280; font-size: 13px; }

/* ---- Legacy card aliases (kept for any remaining inline refs) ---- */
.quiz-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #5b6cff;
    padding: 16px 18px;
    border-radius: 12px;
    margin: 12px 0;
    box-shadow: 0 1px 2px rgba(17, 24, 39, 0.05);
}
.diag-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-left: 4px solid #f59e0b;
    padding: 16px 18px;
    border-radius: 12px;
    margin: 12px 0;
    box-shadow: 0 1px 2px rgba(17, 24, 39, 0.05);
}

/* ---- Score colours (new classes) ---- */
.lm-score { font-size: 36px; font-weight: 700; font-feature-settings: 'tnum';
            line-height: 1; margin: 8px 0 4px; }
.lm-score--good { color: #16a34a; }
.lm-score--mid  { color: #d97706; }
.lm-score--bad  { color: #dc2626; }

/* ---- Legacy score aliases ---- */
.score-good { color: #16a34a; font-weight: 700; font-size: 36px; font-feature-settings: 'tnum'; }
.score-mid  { color: #d97706; font-weight: 700; font-size: 36px; font-feature-settings: 'tnum'; }
.score-bad  { color: #dc2626; font-weight: 700; font-size: 36px; font-feature-settings: 'tnum'; }

/* ---- Muted / italic helpers ---- */
.muted     { color: #6b7280; font-size: 0.92em; font-style: italic; }
.lm-muted  { color: #6b7280; font-style: italic; font-size: 0.92em; }
.ref-list  { font-size: 0.85em; color: #4b5563; }

/* ---- References disclosure ---- */
.lm-refs { font-size: 0.88em; color: #4b5563; margin-top: 12px; }
.lm-refs summary { cursor: pointer; color: #6b7280; font-weight: 500; user-select: none; }
.lm-refs ol { margin: 6px 0 0 18px; padding: 0; }

/* ---- Hero header ---- */
.lm-hero { padding: 8px 0 16px; }
.lm-title { font-size: 28px; font-weight: 700; color: #111827; margin: 0; line-height: 1.2; }
.lm-subtitle { font-size: 14px; color: #6b7280; margin-top: 4px; line-height: 1.4; }
.lm-stats { display: inline-flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.lm-stat-pill { background: #eef2ff; color: #3730a3; padding: 4px 10px; border-radius: 999px;
                font-size: 12px; font-weight: 500; }

/* ---- Legacy hero aliases ---- */
.hero-title    { font-size: 28px; font-weight: 700; color: #111827; margin-bottom: 2px; }
.hero-subtitle { font-size: 14px; color: #6b7280; margin-bottom: 8px; line-height: 1.4; }
.hero-stats    { font-size: 0.85rem; color: #9ca3af; margin-bottom: 0; }

/* ---- Suggested follow-up chips ---- */
.followup-label {
    font-size: 0.8rem;
    color: #6b7280;
    margin-bottom: 4px;
}

/* ---- Progress badge row ---- */
.badge {
    display: inline-block;
    background: #eef2ff;
    color: #3730a3;
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.75rem;
    margin-right: 4px;
    font-weight: 500;
}
.badge-warn {
    background: #fff7ed;
    color: #b45309;
}

/* ---- Progress bars — thinner ---- */
[data-testid="stProgress"] > div > div > div > div {
    height: 6px !important;
}

/* ---- Sidebar section headers ---- */
[data-testid="stSidebar"] h3 {
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6b7280 !important;
    font-weight: 600 !important;
    margin-top: 18px !important;
    margin-bottom: 6px !important;
}

/* ---- Chat input ---- */
[data-testid="stChatInput"] textarea {
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
    font-family: 'Inter', sans-serif !important;
}

/* ---- Buttons ---- */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    border: 1px solid #e5e7eb !important;
    background: #ffffff !important;
    color: #111827 !important;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    border-color: #5b6cff !important;
    color: #5b6cff !important;
}
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------


def _render_hero():
    n_topics = len(_topics())
    n_artifacts = _artifact_count()
    artifact_str = f"{n_artifacts} artifacts" if n_artifacts else "artifacts"
    st.markdown(
        f"""
<div class="lm-hero">
  <h1 class="lm-title">Learning Memory OS</h1>
  <p class="lm-subtitle">Context-routed tutor for ML systems engineers</p>
  <div class="lm-stats">
    <span class="lm-stat-pill">{n_topics} topics</span>
    <span class="lm-stat-pill">{artifact_str}</span>
    <span class="lm-stat-pill">CS153</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.divider()


# ---------------------------------------------------------------------------
# Left sidebar
# ---------------------------------------------------------------------------


def _render_left_sidebar(student_id_default: str = "demo-user"):
    st.sidebar.markdown("### Learning Memory OS")
    st.sidebar.caption("CS 153 — context-routed tutor demo")
    st.sidebar.divider()

    student_id = st.sidebar.text_input("Student ID", value=student_id_default)

    topics = _topics()
    topic_options = ["(global vector search)"] + [t.id for t in topics]
    topic_choice = st.sidebar.selectbox("Topic focus", topic_options, index=0)
    topic_id = None if topic_choice == "(global vector search)" else topic_choice

    budget = st.sidebar.slider(
        "Token budget", min_value=1000, max_value=32000, value=3000, step=500
    )

    st.sidebar.divider()

    # Student mastery + misconceptions
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

    st.sidebar.divider()

    st.sidebar.subheader("Active misconceptions")
    if misconceptions:
        for m in misconceptions[:5]:
            desc = m["description"] or ""
            st.sidebar.write(f"- {desc[:80]}{'...' if len(desc) > 80 else ''}")
    else:
        st.sidebar.caption("(none detected yet)")

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
            st.session_state.diagnostic = {}
            st.session_state.pending_prompt = None
            st.session_state.show_context_analysis = False
            st.rerun()
        if c2.button("Cancel"):
            st.session_state.confirm_clear = False
            st.rerun()

    return student_id, topic_id, budget


# ---------------------------------------------------------------------------
# Right pane — compact stats + toggle
# ---------------------------------------------------------------------------


def _render_right_pane(col, topic_id: str | None, student_id: str):
    """Minimal right pane: gauge + progress + badges + toggle."""
    d = st.session_state.last_decision

    with col:
        # --- Token usage "gauge" (horizontal bar + label) ---
        st.markdown("**Token usage**")
        if d:
            tokens_used = d["tokens_used"]
            budget = d["budget"]
            frac = min(tokens_used / max(budget, 1), 1.0)
            st.progress(frac)
            pct = int(frac * 100)
            color = "#16a34a" if pct < 60 else ("#d97706" if pct < 85 else "#dc2626")
            st.markdown(
                f"<span style='color:{color}; font-size:0.85rem; font-family:Inter,sans-serif;'>"
                f"{tokens_used:,} / {budget:,} tokens ({pct}%)</span>",
                unsafe_allow_html=True,
            )
        else:
            st.progress(0.0)
            st.caption("Ask a question to see usage.")

        st.divider()

        # --- Concept progress ---
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

        st.markdown("**Concepts covered**")
        if n_total > 0:
            frac_c = min(n_seen / n_total, 1.0)
            st.progress(frac_c)
            st.caption(f"{n_seen} / {n_total} in {topic_id or 'session'}")
        else:
            st.caption(f"{n_seen} touched this session")

        st.divider()

        # --- Badges ---
        conn = _new_conn()
        try:
            student_store = StudentStore(conn)
            misconceptions = student_store.active_misconceptions(student_id)
            mastery = student_store.mastery_for(student_id)
        finally:
            conn.close()

        n_mastered = sum(1 for m in mastery if m.score >= 0.8)
        n_misc = len(misconceptions)
        badge_html = (
            f'<span class="badge">{n_mastered} mastered</span>'
            f'<span class="badge badge-warn">{n_misc} misconception{"s" if n_misc != 1 else ""}</span>'
        )
        st.markdown(badge_html, unsafe_allow_html=True)

        st.divider()

        # --- Show context analysis toggle ---
        if d:
            toggle_label = "Hide context analysis" if st.session_state.show_context_analysis else "Show context analysis"
            if st.button(toggle_label, key="toggle_context"):
                st.session_state.show_context_analysis = not st.session_state.show_context_analysis


# ---------------------------------------------------------------------------
# Context analysis expander (full routing diagnostics)
# ---------------------------------------------------------------------------


def _render_context_analysis():
    """Full routing diagnostics, shown only when user requests it."""
    d = st.session_state.last_decision
    if not d or not st.session_state.show_context_analysis:
        return

    with st.expander("Context analysis — how this answer was built", expanded=True):
        n_sel = len(d["selected"])
        n_total_items = n_sel + len(d["dropped"])
        c1, c2 = st.columns(2)
        c1.metric("Items selected", f"{n_sel} / {n_total_items}")
        c2.metric("Tokens used", f"{d['tokens_used']} / {d['budget']}")

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
# Diagnostic chat state machine
# ---------------------------------------------------------------------------


def _generate_diagnostic_question(
    original_question: str,
    rubric: str,
    student_answer: str,
    score: float,
) -> dict:
    """Call LLM to generate the initial diagnosis + follow-up question."""
    llm, _ = _llm_and_embedder()
    system = (
        "You are an expert ML systems tutor diagnosing a student misconception. "
        "The student just answered a quiz question poorly. "
        "Your job is to identify what they likely misunderstand and ask a targeted probing question."
    )
    user = (
        f"ORIGINAL QUESTION: {original_question}\n\n"
        f"RUBRIC: {rubric}\n\n"
        f"STUDENT ANSWER: {student_answer}\n\n"
        f"SCORE: {score:.2f} / 1.0\n\n"
        "Based on the student's answer, diagnose the most likely misconception and craft a follow-up question."
    )
    return llm.complete_with_schema(
        system=system,
        user=user,
        schema=DIAGNOSTIC_SCHEMA,
        tool_name="submit_diagnosis",
        tool_description="Submit the diagnosis and follow-up question.",
    )


def _generate_explanation(
    original_question: str,
    diagnosis: str,
    follow_up_question: str,
    follow_up_answer: str,
) -> dict:
    """Given the student's follow-up answer, generate an explanation or wrap-up."""
    llm, _ = _llm_and_embedder()
    system = (
        "You are an expert ML systems tutor. A student answered a diagnostic follow-up question. "
        "Evaluate their answer to confirm or refute the initial diagnosis. "
        "Then decide: explain the correct mental model, request a re-test, or wrap up if they got it."
    )
    user = (
        f"ORIGINAL QUIZ QUESTION: {original_question}\n\n"
        f"INITIAL DIAGNOSIS: {diagnosis}\n\n"
        f"DIAGNOSTIC FOLLOW-UP: {follow_up_question}\n\n"
        f"STUDENT'S FOLLOW-UP ANSWER: {follow_up_answer}\n\n"
        "Respond with a confirmed misconception label, explanation, next_action (explain/re_test/wrap_up), "
        "and a student-facing next_message."
    )
    return llm.complete_with_schema(
        system=system,
        user=user,
        schema=EXPLAIN_SCHEMA,
        tool_name="submit_explanation",
        tool_description="Submit the confirmed misconception, explanation, and next action.",
    )


def _generate_retest_question(topic_id: str | None, confirmed_misconception: str) -> dict:
    """Generate a targeted re-test question focused on the confirmed misconception."""
    llm, _ = _llm_and_embedder()
    topic_label = topic_id or "ML systems engineering"
    return llm.complete_with_schema(
        system=(
            "Generate ONE targeted quiz question that directly tests whether the student "
            f"has overcome this misconception: {confirmed_misconception}\n"
            f"Topic area: {topic_label}"
        ),
        user=f"Create a question that reveals whether the student now understands: {confirmed_misconception}",
        schema=QUIZ_QUESTION_SCHEMA,
        tool_name="submit_retest_question",
        tool_description="Submit the re-test question and rubric.",
    )


def _record_misconception_to_db(
    student_id: str,
    confirmed_misconception: str,
    original_question: str,
    topic_id: str | None,
) -> str | None:
    """Persist the misconception to the database and return its ID."""
    conn = _new_conn()
    try:
        student_store = StudentStore(conn)
        misconception_id = student_store.record_misconception(
            student_id,
            concept_id=None,
            description=confirmed_misconception,
            evidence=original_question,
        )
        conn.commit()
        return misconception_id
    except Exception as exc:
        st.warning(f"Could not persist misconception: {exc}")
        return None
    finally:
        conn.close()


def _render_diagnostic_flow(
    quiz_key: str,
    msg_idx: int,
    topic_id: str | None,
    student_id: str,
):
    """Render the full diagnostic chat state machine for a low-scoring quiz attempt."""
    state = st.session_state.quiz_state[quiz_key]
    diag_key = quiz_key  # reuse same key for diagnostic state
    diag = st.session_state.diagnostic.get(diag_key)

    # Initialise diagnostic state if not yet started
    if diag is None:
        diag = {
            "phase": "init",
            "turns": 0,
            "diagnosis": None,
            "follow_up_question": None,
            "confirmed_misconception": None,
            "explanation": None,
            "next_action": None,
            "next_message": None,
            "retest_question": None,
            "retest_rubric": None,
            "retest_answer": None,
            "retest_score": None,
            "misconception_id": None,
        }
        st.session_state.diagnostic[diag_key] = diag

    # === PHASE: init — generate first diagnostic question ===
    if diag["phase"] == "init":
        st.markdown(
            """<div class="lm-card lm-diag-card">
<div class="lm-card-header">🧭 Diagnosing</div>
<span class="lm-muted">Your score was below the threshold. Let me ask you a follow-up question to pinpoint the gap.</span>
</div>""",
            unsafe_allow_html=True,
        )
        with st.spinner("Generating diagnostic question..."):
            try:
                result = _generate_diagnostic_question(
                    original_question=state["question"],
                    rubric=state["rubric"],
                    student_answer=state["answer"] or "",
                    score=state["score"],
                )
                diag["diagnosis"] = result["diagnosis"]
                diag["follow_up_question"] = result["follow_up_question"]
                diag["phase"] = "asking"
                st.session_state.diagnostic[diag_key] = diag
                st.rerun()
            except Exception as exc:
                st.error(f"Could not generate diagnostic question: {exc}")
                return

    # === PHASE: asking — show follow-up question, await student answer ===
    if diag["phase"] == "asking":
        st.markdown(
            f"""<div class="lm-card lm-diag-card">
<div class="lm-card-header">🧭 Diagnosing</div>
<span class="lm-muted">Possible gap: <em>{diag["diagnosis"]}</em></span><br><br>
<strong style="font-size:1.05em; color:#111827;">{diag["follow_up_question"]}</strong>
</div>""",
            unsafe_allow_html=True,
        )
        with st.form(key=f"diag_form_{msg_idx}"):
            followup_ans = st.text_area(
                "Your response:", key=f"diag_answer_{msg_idx}", height=100
            )
            submitted = st.form_submit_button("Submit response")

        if submitted and followup_ans.strip():
            diag["turns"] += 1
            with st.spinner("Analysing your response..."):
                try:
                    result = _generate_explanation(
                        original_question=state["question"],
                        diagnosis=diag["diagnosis"],
                        follow_up_question=diag["follow_up_question"],
                        follow_up_answer=followup_ans.strip(),
                    )
                    diag["confirmed_misconception"] = result["confirmed_misconception"]
                    diag["explanation"] = result["explanation"]
                    diag["next_action"] = result["next_action"]
                    diag["next_message"] = result["next_message"]
                    diag["phase"] = "explaining"
                    st.session_state.diagnostic[diag_key] = diag
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not analyse response: {exc}")
                    return

    # === PHASE: explaining — show the explanation and act on next_action ===
    if diag["phase"] == "explaining":
        next_action = diag.get("next_action", "wrap_up")

        st.markdown(
            f"""<div class="lm-card lm-diag-card">
<div class="lm-card-header">🧭 Diagnosing</div>
<span class="lm-muted">Confirmed gap: <em>{diag["confirmed_misconception"]}</em></span><br><br>
<span style="color:#111827;">{diag["explanation"]}</span><br><br>
<em class="lm-muted">{diag["next_message"]}</em>
</div>""",
            unsafe_allow_html=True,
        )

        if next_action == "wrap_up" or diag["turns"] >= DIAGNOSTIC_MAX_TURNS:
            # Record the misconception and mark done
            if diag["misconception_id"] is None:
                misc_id = _record_misconception_to_db(
                    student_id=student_id,
                    confirmed_misconception=diag["confirmed_misconception"],
                    original_question=state["question"],
                    topic_id=topic_id,
                )
                diag["misconception_id"] = misc_id
            diag["phase"] = "done"
            st.session_state.diagnostic[diag_key] = diag
            st.success(
                "Misconception logged. It will now appear in your sidebar and influence future question selection."
            )

        elif next_action == "re_test":
            if diag["retest_question"] is None:
                with st.spinner("Generating a targeted re-test question..."):
                    try:
                        rq = _generate_retest_question(
                            topic_id=topic_id,
                            confirmed_misconception=diag["confirmed_misconception"],
                        )
                        diag["retest_question"] = rq["question"]
                        diag["retest_rubric"] = rq["rubric"]
                        diag["phase"] = "retesting"
                        st.session_state.diagnostic[diag_key] = diag
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not generate re-test: {exc}")
                        return
            else:
                diag["phase"] = "retesting"
                st.session_state.diagnostic[diag_key] = diag
                st.rerun()

        elif next_action == "explain":
            # Another turn of explanation — prompt student for confirmation
            if diag["turns"] < DIAGNOSTIC_MAX_TURNS:
                with st.form(key=f"diag_confirm_form_{msg_idx}"):
                    confirm_ans = st.text_area(
                        "Does this make sense? Tell me in your own words or ask a follow-up:",
                        key=f"diag_confirm_{msg_idx}",
                        height=80,
                    )
                    confirm_submitted = st.form_submit_button("Continue")
                if confirm_submitted and confirm_ans.strip():
                    diag["turns"] += 1
                    # Treat this as another asking turn
                    diag["follow_up_question"] = confirm_ans.strip()
                    diag["phase"] = "asking"
                    st.session_state.diagnostic[diag_key] = diag
                    st.rerun()
            else:
                if diag["misconception_id"] is None:
                    misc_id = _record_misconception_to_db(
                        student_id=student_id,
                        confirmed_misconception=diag["confirmed_misconception"],
                        original_question=state["question"],
                        topic_id=topic_id,
                    )
                    diag["misconception_id"] = misc_id
                diag["phase"] = "done"
                st.session_state.diagnostic[diag_key] = diag

    # === PHASE: retesting — show the re-test question ===
    if diag["phase"] == "retesting":
        st.markdown(
            f"""<div class="lm-card lm-quiz-card">
<div class="lm-card-header">🎯 Re-test</div>
<span class="lm-card-sub">Let's see if the concept clicks now</span><br><br>
<span style="font-size:1.05em; color:#111827;">{diag["retest_question"]}</span>
</div>""",
            unsafe_allow_html=True,
        )

        if diag.get("retest_score") is None:
            with st.form(key=f"retest_form_{msg_idx}"):
                retest_ans = st.text_area(
                    "Your answer:", key=f"retest_answer_{msg_idx}", height=100
                )
                retest_submitted = st.form_submit_button("Submit re-test answer")

            if retest_submitted and retest_ans.strip():
                with st.spinner("Grading re-test..."):
                    llm, _ = _llm_and_embedder()
                    qq = QuizQuestion(
                        question=diag["retest_question"],
                        rubric=diag["retest_rubric"],
                    )
                    try:
                        quiz_score = score_answer(
                            question=qq,
                            student_answer=retest_ans.strip(),
                            judge_llm=llm,
                        )
                        diag["retest_answer"] = retest_ans.strip()
                        diag["retest_score"] = quiz_score.score
                        st.session_state.diagnostic[diag_key] = diag
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Re-test grading failed: {exc}")
                        return
        else:
            # Re-test has been scored
            rs = diag["retest_score"]
            if rs >= DIAGNOSTIC_THRESHOLD:
                score_cls = "lm-score lm-score--good"
                score_label = "Great improvement!"
            else:
                score_cls = "lm-score lm-score--bad"
                score_label = "Still needs work"
            bar_val = int(rs * 100)
            st.markdown(
                f'<span class="{score_cls}">{bar_val}/100</span>'
                f' <span class="lm-muted">— {score_label}</span>',
                unsafe_allow_html=True,
            )
            # Log misconception regardless of re-test result
            if diag["misconception_id"] is None:
                misc_id = _record_misconception_to_db(
                    student_id=student_id,
                    confirmed_misconception=diag["confirmed_misconception"],
                    original_question=state["question"],
                    topic_id=topic_id,
                )
                diag["misconception_id"] = misc_id
            if rs >= DIAGNOSTIC_THRESHOLD:
                st.success("Misconception logged and you demonstrated improvement. Keep it up!")
            else:
                st.warning(
                    "Misconception logged for future sessions. This concept will be revisited proactively."
                )
            diag["phase"] = "done"
            st.session_state.diagnostic[diag_key] = diag

    # === PHASE: done ===
    if diag["phase"] == "done":
        st.markdown(
            '<div class="lm-card lm-diag-card"><span class="lm-muted">Diagnostic session complete. '
            "The misconception has been saved and will influence future tutoring.</span></div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Quiz button + flow (redesigned)
# ---------------------------------------------------------------------------


def _render_quiz_for_message(msg_idx: int, topic_id: str | None, student_id: str):
    """Render the 'Test yourself' button and quiz flow for a given assistant message index."""
    quiz_key = str(msg_idx)
    state = st.session_state.quiz_state.get(quiz_key, {})

    if not state:
        if st.button("Test yourself", key=f"quiz_btn_{msg_idx}"):
            with st.spinner("Generating a quiz question..."):
                llm, _ = _llm_and_embedder()
                topic_label = topic_id or "ML systems engineering"
                try:
                    data = llm.complete_with_schema(
                        system="Generate ONE substantive quiz question on the given ML systems engineering topic.",
                        user=f"TOPIC: {topic_label}",
                        schema=QUIZ_QUESTION_SCHEMA,
                        tool_name="submit_quiz_question",
                        tool_description="Submit the generated quiz question and its rubric.",
                    )
                    q_text = data["question"]
                    rubric = data["rubric"]
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

    # Quiz card header
    st.markdown(
        f"""<div class="lm-card lm-quiz-card">
<div class="lm-card-header">🎯 Challenge</div>
<span style="font-size:1.05em; color:#111827;">{state["question"]}</span>
</div>""",
        unsafe_allow_html=True,
    )

    if state.get("score") is not None:
        # Already scored — render score with colour
        score_val = state["score"]
        bar_val = int(score_val * 100)
        if score_val >= 0.8:
            score_cls = "lm-score lm-score--good"
        elif score_val >= DIAGNOSTIC_THRESHOLD:
            score_cls = "lm-score lm-score--mid"
        else:
            score_cls = "lm-score lm-score--bad"

        st.markdown(
            f'<span class="{score_cls}">{bar_val}/100</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span class="lm-muted">{state["rationale"]}</span>',
            unsafe_allow_html=True,
        )

        # Diagnostic flow for low scores
        if score_val < DIAGNOSTIC_THRESHOLD:
            _render_diagnostic_flow(quiz_key, msg_idx, topic_id, student_id)

        if st.button("Try another question", key=f"quiz_retry_{msg_idx}"):
            del st.session_state.quiz_state[quiz_key]
            if quiz_key in st.session_state.diagnostic:
                del st.session_state.diagnostic[quiz_key]
            st.rerun()
    else:
        # Answer form
        with st.form(key=f"quiz_form_{msg_idx}"):
            answer = st.text_area("Your answer:", key=f"quiz_answer_{msg_idx}", height=100)
            submitted = st.form_submit_button("Submit answer")

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

        for it in response.selected_items:
            st.session_state.reuse_counts[it.id] += 1

        topic_key = topic_id or "_global"
        if topic_key not in st.session_state.seen_concepts_by_topic:
            st.session_state.seen_concepts_by_topic[topic_key] = set()
        for it in decision.selected:
            if getattr(it, "artifact_type", None) == "concept":
                st.session_state.seen_concepts_by_topic[topic_key].add(it.id)

        episodic.append(
            student_id=student_id,
            event_type="question",
            payload={"text": prompt, "topic_id": topic_id, "source": "streamlit_app"},
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
<h2 style="margin-top:0; color:#1f2235;">Welcome — I'm your ML systems tutor.</h2>
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

    starters = _starter_prompts_for(topic_id)
    st.markdown("**Jump in with:**")
    cols = st.columns(len(starters))
    for i, (col, prompt) in enumerate(zip(cols, starters)):
        if col.button(prompt, key=f"starter_{i}"):
            st.session_state.pending_prompt = prompt
            st.rerun()


# ---------------------------------------------------------------------------
# Suggested follow-ups (extracted from last assistant message)
# ---------------------------------------------------------------------------


_FOLLOWUP_RE = re.compile(
    r"(?:go deeper|explore|dive into|cover|discuss|learn about|want me to cover)\s+([^?.\n]{8,60})\??",
    re.IGNORECASE,
)


def _extract_followups(text: str) -> list[str]:
    """Extract 2-3 suggested follow-up topics from the tutor's closing line."""
    matches = _FOLLOWUP_RE.findall(text)
    # Capitalise, deduplicate, trim
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        m = m.strip().rstrip(".,;:")
        key = m.lower()
        if key not in seen and len(m) > 6:
            seen.add(key)
            result.append(m[0].upper() + m[1:])
        if len(result) >= 3:
            break
    return result


def _render_suggested_followups(topic_id: str | None):
    """Show suggested follow-up chips based on the last assistant message."""
    msgs = st.session_state.messages
    if not msgs:
        return
    last_asst = next(
        (m["content"] for m in reversed(msgs) if m["role"] == "assistant"), None
    )
    if not last_asst:
        return
    followups = _extract_followups(last_asst)
    if not followups:
        return
    st.markdown('<div class="followup-label">Suggested follow-ups:</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(followups), 3))
    for i, (col, fu) in enumerate(zip(cols, followups)):
        label = fu[:50] + ("..." if len(fu) > 50 else "")
        if col.button(label, key=f"followup_{i}"):
            st.session_state.pending_prompt = fu
            st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    _init_session()
    _inject_css()
    _render_hero()

    # Left sidebar
    student_id, topic_id, budget = _render_left_sidebar()

    # Three-zone layout: main chat (3) + right drawer (1)
    main_col, right_col = st.columns([3, 1])

    with main_col:
        # Welcome screen when chat is empty
        if not st.session_state.messages:
            _render_welcome(topic_id)
        else:
            # Suggested follow-ups instead of starter chips once chat has messages
            _render_suggested_followups(topic_id)

        # Replay chat history
        for idx, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    selected_items = []
                    if idx == len(st.session_state.messages) - 1 and st.session_state.last_decision:
                        selected_items = st.session_state.last_decision.get("selected", [])
                    rendered = _render_citations(msg["content"], selected_items)
                    _render_with_mermaid(rendered)
                    # Quiz button below each assistant message
                    _render_quiz_for_message(idx, topic_id, student_id)
                else:
                    st.markdown(msg["content"])

        # Handle pending prompt (chip click)
        active_prompt: str | None = None
        if st.session_state.pending_prompt:
            active_prompt = st.session_state.pending_prompt
            st.session_state.pending_prompt = None

        prompt = st.chat_input("Ask the tutor a question...")
        if active_prompt:
            prompt = active_prompt

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Selecting context and generating response..."):
                    reply = _handle_turn(prompt, student_id, topic_id, budget)
                selected_items = (
                    st.session_state.last_decision.get("selected", [])
                    if st.session_state.last_decision
                    else []
                )
                rendered_reply = _render_citations(reply, selected_items)
                _render_with_mermaid(rendered_reply)
                new_msg_idx = len(st.session_state.messages)
                _render_quiz_for_message(new_msg_idx, topic_id, student_id)

            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

        # Context analysis expander (below chat, full width)
        _render_context_analysis()

    with right_col:
        _render_right_pane(right_col, topic_id, student_id)


if __name__ == "__main__":
    main()
