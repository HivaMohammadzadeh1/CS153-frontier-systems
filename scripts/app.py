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
from learning_memory_os.agents.tutor import TUTOR_SYSTEM
from learning_memory_os.logging_utils.interactions import InteractionLogger
from learning_memory_os.ingestion.topic_loader import load_topics, resolve_prerequisite_titles
from learning_memory_os.eval.quiz import QuizQuestion, score_answer
from learning_memory_os.memory.xtrace import XTraceClient, xtrace_to_memory_item
from learning_memory_os.agents.topic_inference import (
    TopicCentroid,
    TopicCentroids,
    infer_topic,
)

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
def _xtrace_client() -> XTraceClient | None:
    s = _settings()
    if not s.xtrace_api_key or not s.xtrace_org_id:
        return None
    return XTraceClient(
        api_key=s.xtrace_api_key,
        org_id=s.xtrace_org_id,
        base_url=s.xtrace_base_url,
    )


@st.cache_resource
def _topic_centroids() -> TopicCentroids:
    """Build per-topic centroids from each topic's existing semantic-item embeddings."""
    topics = _topics()
    conn = _new_conn()
    centroids: list[TopicCentroid] = []
    try:
        semantic = SemanticStore(conn)
        for t in topics:
            try:
                items = semantic.by_topic(t.id)
            except Exception:
                items = []
            vectors = [it.embedding for it in items if it.embedding]
            if not vectors:
                continue
            dim = len(vectors[0])
            mean = [0.0] * dim
            for v in vectors:
                for i in range(dim):
                    mean[i] += v[i]
            mean = [x / len(vectors) for x in mean]
            import math as _m
            norm = _m.sqrt(sum(x * x for x in mean))
            if norm > 0:
                mean = [x / norm for x in mean]
            centroids.append(TopicCentroid(topic_id=t.id, vector=mean))
    finally:
        conn.close()
    return TopicCentroids(centroids)


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
    if "last_inferred_topic" not in st.session_state:
        st.session_state.last_inferred_topic = None
    if "chat_session_id" not in st.session_state:
        import uuid as _uuid
        st.session_state.chat_session_id = f"chat_{_uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Global CSS injection
# ---------------------------------------------------------------------------


def _inject_css():
    st.html(
        """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ============================================================
   Design tokens
   ============================================================ */
:root {
    --bg:           #fafbfd;
    --surface:      #ffffff;
    --surface-2:    #f4f5f9;
    --border:       #e6e8ef;
    --border-soft:  #eef0f5;
    --text:         #0f172a;
    --text-muted:   #5a6275;
    --text-soft:    #8a91a3;
    --accent:       #6366f1;
    --accent-soft:  #eef0ff;
    --accent-text:  #3730a3;
    --success:      #15803d;
    --success-soft: #e7f7ec;
    --warn:         #b45309;
    --warn-soft:    #fff5e0;
    --danger:       #b91c1c;
    --danger-soft:  #fee2e2;
    --shadow-sm:    0 1px 2px rgba(15, 23, 42, 0.04);
    --shadow-md:    0 2px 6px rgba(15, 23, 42, 0.06);
    --radius-sm:    8px;
    --radius:       12px;
    --radius-lg:    16px;
}

/* ============================================================
   Base
   ============================================================ */
html, body, [class*="css"], [data-testid="stMarkdownContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text);
}
.stApp { background: var(--bg); }

/* Tighten the top padding of the main content */
.main .block-container { padding-top: 1.5rem; }

/* ============================================================
   Tabs — pill-style with indigo accent
   ============================================================ */
[data-testid="stTabs"] button[role="tab"] {
    font-weight: 500;
    color: var(--text-muted);
    padding: 10px 22px;
    border-radius: 999px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.15s ease;
    margin-right: 6px;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: var(--text);
    background: var(--surface-2) !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--accent-text) !important;
    border-color: var(--border) !important;
    box-shadow: var(--shadow-sm);
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid var(--border-soft);
    padding-bottom: 8px;
    margin-bottom: 18px;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: transparent !important;
}

/* ============================================================
   Bordered containers — consistent surface treatment
   ============================================================ */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm);
    background: var(--surface);
}

/* ============================================================
   Chat message bubbles
   ============================================================ */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border-radius: var(--radius);
    padding: 12px 16px !important;
    margin-bottom: 12px;
    line-height: 1.6;
}
[data-testid="stChatMessage"][aria-label*="user"] {
    background: var(--surface-2) !important;
}
[data-testid="stChatMessage"][aria-label*="assistant"] {
    background: var(--surface) !important;
    border: 1px solid var(--border);
    box-shadow: var(--shadow-sm);
}

/* ============================================================
   Sidebar — denser, more refined
   ============================================================ */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border-soft);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: var(--text);
}
[data-testid="stSidebar"] h3 {
    font-size: 10.5px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-soft) !important;
    font-weight: 600 !important;
    margin: 22px 0 8px 0 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    margin-bottom: 4px;
    font-size: 0.92rem;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stSlider {
    font-size: 0.9rem;
}
[data-testid="stSidebar"] hr { margin: 14px 0 !important; }

/* ============================================================
   Chips / pills
   ============================================================ */
.lm-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent-text);
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 6px;
    margin-bottom: 4px;
}
.lm-chip--neutral { background: var(--surface-2); color: var(--text-muted); }
.lm-chip--success { background: var(--success-soft); color: var(--success); }
.lm-chip--warn    { background: var(--warn-soft);    color: var(--warn); }
.lm-chip--danger  { background: var(--danger-soft);  color: var(--danger); }
.lm-chip-dot {
    width: 6px; height: 6px; border-radius: 999px; background: currentColor;
    display: inline-block;
}

/* ============================================================
   Mini progress bar (mastery)
   ============================================================ */
.lm-meter {
    background: var(--surface-2);
    border-radius: 999px;
    height: 6px;
    overflow: hidden;
    margin: 2px 0 8px 0;
}
.lm-meter-fill { height: 100%; border-radius: 999px; }
.lm-meter-fill--good { background: var(--success); }
.lm-meter-fill--mid  { background: var(--warn); }
.lm-meter-fill--low  { background: #d97706; }

.lm-mastery-row {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.82rem; color: var(--text-muted); margin-bottom: 2px;
}
.lm-mastery-row strong { color: var(--text); font-weight: 500; }

/* ============================================================
   Hero
   ============================================================ */
.lm-hero {
    display: flex; align-items: center; justify-content: space-between;
    padding: 4px 0 16px 0;
    margin-bottom: 6px;
    border-bottom: 1px solid var(--border-soft);
}
.lm-hero-brand { display: flex; align-items: center; gap: 12px; }
.lm-hero-mark {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, var(--accent) 0%, #8b5cf6 100%);
    display: inline-flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 700; font-size: 18px;
    box-shadow: var(--shadow-md);
}
.lm-hero-text { line-height: 1.2; }
.lm-hero-name { font-size: 1.05rem; font-weight: 600; color: var(--text); }
.lm-hero-sub  { font-size: 0.82rem; color: var(--text-soft); }
.lm-hero-stats { display: inline-flex; gap: 6px; }

/* ============================================================
   Cards (quiz / diag) — kept, palette aligned
   ============================================================ */
.lm-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 18px;
    margin: 12px 0;
    box-shadow: var(--shadow-sm);
}
.lm-quiz-card  { border-left: 4px solid var(--accent); }
.lm-diag-card  { border-left: 4px solid var(--warn); }
.lm-card-header { font-weight: 600; font-size: 16px; color: var(--text); margin-bottom: 8px; }
.lm-card-sub    { color: var(--text-muted); font-size: 13px; }

.quiz-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    padding: 16px 18px;
    border-radius: var(--radius);
    margin: 12px 0;
    box-shadow: var(--shadow-sm);
}
.diag-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 4px solid var(--warn);
    padding: 16px 18px;
    border-radius: var(--radius);
    margin: 12px 0;
    box-shadow: var(--shadow-sm);
}

/* ============================================================
   Score colors
   ============================================================ */
.lm-score { font-size: 36px; font-weight: 700; font-feature-settings: 'tnum';
            line-height: 1; margin: 8px 0 4px; }
.lm-score--good { color: var(--success); }
.lm-score--mid  { color: var(--warn); }
.lm-score--bad  { color: var(--danger); }
.score-good { color: var(--success); font-weight: 700; font-size: 36px; font-feature-settings: 'tnum'; }
.score-mid  { color: var(--warn);    font-weight: 700; font-size: 36px; font-feature-settings: 'tnum'; }
.score-bad  { color: var(--danger);  font-weight: 700; font-size: 36px; font-feature-settings: 'tnum'; }

/* ============================================================
   Muted helpers, references
   ============================================================ */
.muted     { color: var(--text-muted); font-size: 0.92em; font-style: italic; }
.lm-muted  { color: var(--text-muted); font-style: italic; font-size: 0.92em; }
.ref-list  { font-size: 0.85em; color: var(--text-muted); }
.lm-refs   { font-size: 0.88em; color: var(--text-muted); margin-top: 12px; }
.lm-refs summary { cursor: pointer; color: var(--text-soft); font-weight: 500; user-select: none; }
.lm-refs ol { margin: 6px 0 0 18px; padding: 0; }

/* ============================================================
   Legacy badge classes — palette aligned
   ============================================================ */
.badge {
    display: inline-block;
    background: var(--accent-soft);
    color: var(--accent-text);
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 0.75rem;
    margin-right: 4px;
    font-weight: 500;
}
.badge-warn { background: var(--warn-soft); color: var(--warn); }

/* ============================================================
   Suggested follow-up label
   ============================================================ */
.followup-label {
    font-size: 0.78rem;
    color: var(--text-soft);
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 6px 0;
}

/* ============================================================
   Progress bars — thinner, indigo
   ============================================================ */
[data-testid="stProgress"] > div > div > div > div {
    height: 6px !important;
    background: var(--accent) !important;
}
[data-testid="stProgress"] > div > div > div {
    background: var(--surface-2) !important;
    border-radius: 999px;
}

/* ============================================================
   Section labels (small caps, used in main area headers)
   ============================================================ */
.lm-section-label {
    font-size: 0.78rem; color: var(--text-soft);
    font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 24px 0 10px 0;
}

/* ============================================================
   Form inputs — chat input, text input
   ============================================================ */
[data-testid="stChatInput"] textarea {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    font-family: 'Inter', sans-serif !important;
    background: var(--surface) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
.stTextInput input {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
}
.stSelectbox [data-baseweb="select"] > div {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
}

/* ============================================================
   Buttons — secondary by default, primary indigo on hover
   ============================================================ */
.stButton > button {
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    transition: all 0.15s ease;
    box-shadow: var(--shadow-sm);
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-soft) !important;
}

/* Primary buttons used as the active tab indicator */
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #fff !important;
    border-color: var(--accent) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #5054e0 !important;
    border-color: #5054e0 !important;
    color: #fff !important;
}

/* Metric — tighter typography */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 14px 16px;
    box-shadow: var(--shadow-sm);
}
[data-testid="stMetricLabel"] {
    color: var(--text-soft) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 600 !important;
}

/* Divider line color */
hr { border-color: var(--border-soft) !important; }
</style>
"""
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
  <div class="lm-hero-brand">
    <span class="lm-hero-mark">M</span>
    <div class="lm-hero-text">
      <div class="lm-hero-name">Memex</div>
      <div class="lm-hero-sub">Context-routed ML systems tutor</div>
    </div>
  </div>
  <div class="lm-hero-stats">
    <span class="lm-chip lm-chip--neutral">{n_topics} topics</span>
    <span class="lm-chip lm-chip--neutral">{artifact_str}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Left sidebar
# ---------------------------------------------------------------------------


def _render_left_sidebar(student_id_default: str = "demo-user"):
    student_id = st.sidebar.text_input("Student ID", value=student_id_default, label_visibility="collapsed", placeholder="Student ID")

    if "topic_choice" not in st.session_state:
        st.session_state.topic_choice = "(auto)"
    topic_id = None if st.session_state.topic_choice == "(auto)" else st.session_state.topic_choice

    # Token budget no longer surfaced in the UI; read the fixed default from settings.
    budget = _settings().default_token_budget

    xtrace = _xtrace_client()
    if xtrace is None:
        st.sidebar.markdown(
            "<div style='margin-top:8px; padding:8px 12px; background:var(--surface-2); "
            "border-radius:8px; font-size:0.78rem; color:var(--text-muted);'>"
            "Long-term memory off — set <code>XTRACE_API_KEY</code> to enable."
            "</div>",
            unsafe_allow_html=True,
        )

    # Pull student data once.
    conn = _new_conn()
    try:
        student_store = StudentStore(conn)
        student_store.ensure_student(student_id)
        conn.commit()
        mastery = student_store.mastery_for(student_id)
        misconceptions = student_store.active_misconceptions(student_id)
        # Map concept ids → titles so the sidebar shows readable names, not UUIDs.
        concept_titles: dict[str, str] = {}
        if mastery:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id::text AS id, title FROM semantic_items WHERE id::text = ANY(%s)",
                    ([m.concept_id for m in mastery],),
                )
                concept_titles = {r["id"]: r["title"] or "(untitled)" for r in cur.fetchall()}
    finally:
        conn.close()

    # --- Mastery ---
    st.sidebar.markdown("### Mastery")
    if mastery:
        top_mastery = sorted(mastery, key=lambda m: m.score, reverse=True)[:6]
        for m in top_mastery:
            score = m.score
            fill_class = (
                "lm-meter-fill--good" if score >= 0.7
                else "lm-meter-fill--mid" if score >= 0.4
                else "lm-meter-fill--low"
            )
            pct = int(score * 100)
            title = concept_titles.get(m.concept_id, m.concept_id[:10])
            label = title if len(title) <= 22 else title[:20] + "…"
            st.sidebar.markdown(
                f"<div class='lm-mastery-row'>"
                f"<strong title='{title}'>{label}</strong>"
                f"<span>{pct}%</span>"
                f"</div>"
                f"<div class='lm-meter'>"
                f"<div class='lm-meter-fill {fill_class}' style='width:{pct}%'></div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.markdown(
            "<div style='font-size:0.85rem; color:var(--text-soft); padding:4px 0;'>"
            "No mastery yet — ask a question and take a quiz."
            "</div>",
            unsafe_allow_html=True,
        )

    # --- Misconceptions ---
    st.sidebar.markdown("### Misconceptions")
    if misconceptions:
        for m in misconceptions[:5]:
            desc = (m["description"] or "")[:90]
            st.sidebar.markdown(
                f"<div style='display:flex; gap:8px; align-items:flex-start; "
                f"padding:6px 10px; margin-bottom:6px; background:var(--danger-soft); "
                f"border-radius:8px; font-size:0.82rem; color:var(--danger); line-height:1.4;'>"
                f"<span style='flex-shrink:0; width:6px; height:6px; border-radius:50%; "
                f"background:var(--danger); margin-top:6px;'></span>"
                f"<span>{desc}{'...' if len(m['description'] or '') > 90 else ''}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.markdown(
            "<div style='font-size:0.85rem; color:var(--text-soft); padding:4px 0;'>"
            "None detected."
            "</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.divider()

    # --- Clear chat — two-step confirm ---
    if not st.session_state.confirm_clear:
        if st.sidebar.button("Clear chat", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.sidebar.markdown(
            "<div style='padding:8px 12px; background:var(--warn-soft); "
            "border-radius:8px; font-size:0.82rem; color:var(--warn); margin-bottom:8px;'>"
            "Clear all chat history?"
            "</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.sidebar.columns(2)
        if c1.button("Yes", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_decision = None
            st.session_state.reuse_counts = Counter()
            st.session_state.confirm_clear = False
            st.session_state.seen_concepts_by_topic = {}
            st.session_state.quiz_state = {}
            st.session_state.diagnostic = {}
            st.session_state.pending_prompt = None
            st.session_state.show_context_analysis = False
            st.session_state.last_inferred_topic = None
            import uuid as _uuid
            st.session_state.chat_session_id = f"chat_{_uuid.uuid4().hex[:12]}"
            st.rerun()
        if c2.button("Cancel", use_container_width=True):
            st.session_state.confirm_clear = False
            st.rerun()

    return student_id, topic_id, budget


# ---------------------------------------------------------------------------
# Right pane — compact stats + toggle
# ---------------------------------------------------------------------------


def _render_right_pane(col, topic_id: str | None, student_id: str):
    """Compact right rail: concept progress, status badges, debug toggle."""
    d = st.session_state.last_decision

    with col:
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

        st.markdown(
            "<div class='lm-section-label'>Concepts covered</div>",
            unsafe_allow_html=True,
        )
        if n_total > 0:
            frac_c = min(n_seen / n_total, 1.0)
            pct_c = int(frac_c * 100)
            st.markdown(
                f"<div style='font-size:1.3rem; font-weight:600; color:var(--text); margin-bottom:2px;'>"
                f"{n_seen}<span style='font-size:0.8rem; color:var(--text-soft); font-weight:400;'> / {n_total}</span>"
                f"</div>"
                f"<div class='lm-meter'>"
                f"<div class='lm-meter-fill lm-meter-fill--good' style='width:{pct_c}%'></div>"
                f"</div>"
                f"<div style='font-size:0.78rem; color:var(--text-soft);'>in {_topic_title(topic_id) if topic_id else 'session'}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='font-size:1.3rem; font-weight:600; color:var(--text);'>{n_seen}</div>"
                f"<div style='font-size:0.78rem; color:var(--text-soft);'>touched this session</div>",
                unsafe_allow_html=True,
            )

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
        st.markdown(
            "<div class='lm-section-label'>Status</div>"
            f"<div style='display:flex; flex-direction:column; gap:6px;'>"
            f"<span class='lm-chip lm-chip--success'><span class='lm-chip-dot'></span>{n_mastered} mastered</span>"
            f"<span class='lm-chip lm-chip--danger'><span class='lm-chip-dot'></span>"
            f"{n_misc} misconception{'s' if n_misc != 1 else ''}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # --- Context analysis toggle ---
        if d:
            st.markdown("<div style='margin-top:18px;'></div>", unsafe_allow_html=True)
            toggle_label = "Hide context analysis" if st.session_state.show_context_analysis else "Show context analysis"
            if st.button(toggle_label, key="toggle_context", use_container_width=True):
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
        # Mirror to XTrace so long-term memory captures the misconception too.
        xtrace = _xtrace_client()
        if xtrace is not None:
            topic_label = _topic_title(topic_id) if topic_id else "(unknown topic)"
            xtrace.ingest_fact(
                student_id=student_id,
                text=f"Misconception identified in {topic_label}: {confirmed_misconception}",
                conv_id=st.session_state.get("chat_session_id"),
            )
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
                        # Mirror retest outcome to XTrace.
                        xtrace = _xtrace_client()
                        if xtrace is not None:
                            topic_label = _topic_title(topic_id) if topic_id else "(unspecified)"
                            outcome = (
                                "successfully corrected"
                                if quiz_score.score >= DIAGNOSTIC_THRESHOLD
                                else "still has the misconception"
                            )
                            xtrace.ingest_fact(
                                student_id=student_id,
                                text=(
                                    f"After a misconception correction on {topic_label}, "
                                    f"the student {outcome} "
                                    f"(retest score {int(quiz_score.score*100)}%)."
                                ),
                                conv_id=st.session_state.get("chat_session_id"),
                            )
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

            # Mirror quiz outcome to XTrace as a learning signal.
            xtrace = _xtrace_client()
            if xtrace is not None:
                outcome = (
                    "demonstrated solid understanding"
                    if quiz_score.score >= 0.7
                    else "got partial credit"
                    if quiz_score.score >= 0.4
                    else "struggled"
                )
                topic_label = _topic_title(topic_id) if topic_id else "(unspecified topic)"
                xtrace.ingest_fact(
                    student_id=student_id,
                    text=(
                        f"On a quiz about {topic_label}, the student {outcome} "
                        f"(score {int(quiz_score.score*100)}%). "
                        f"Question: {state['question'][:160]}"
                    ),
                    conv_id=st.session_state.get("chat_session_id"),
                )

            st.rerun()


# ---------------------------------------------------------------------------
# Turn handler
# ---------------------------------------------------------------------------


def _handle_turn(
    prompt: str, student_id: str, topic_id: str | None, budget: int
):
    """Run one tutor turn as a generator of text deltas.

    Phase 1 (before first yield): topic inference, candidate selection, prompt
    building. The Streamlit caller can show a 'Selecting context...' placeholder
    while this is running.

    Phase 2: yields text chunks streamed from the LLM. The caller appends them
    to a placeholder for a typing-style effect.

    Phase 3 (after stream loop exits): episodic logging, XTrace ingest, and
    last_decision update. Runs once the generator is fully consumed.
    """
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

        # Conversation-level topic inference when no manual topic is pinned.
        if topic_id is None:
            try:
                centroids = _topic_centroids()
                result = infer_topic(
                    conversation=list(st.session_state.messages),
                    centroids=centroids,
                    embed_fn=embedder.embed_one,
                )
                st.session_state.last_inferred_topic = {
                    "topic_id": result.topic_id,
                    "confidence": result.confidence,
                    "decision": result.decision,
                }
                if result.decision in ("auto", "inferred") and result.topic_id:
                    topic_id = result.topic_id
            except Exception:
                st.session_state.last_inferred_topic = {
                    "topic_id": None,
                    "confidence": 0.0,
                    "decision": "ask",
                }

        if topic_id:
            candidates = semantic.by_topic(topic_id)
        else:
            q_emb = embedder.embed_one(prompt)
            candidates = semantic.vector_search(query=q_emb, k=20)

        # 5th memory tier: XTrace long-horizon recall.
        xtrace = _xtrace_client()
        if xtrace is not None:
            for hit in xtrace.recall(student_id=student_id, query=prompt, k=5):
                candidates.append(xtrace_to_memory_item(hit))

        misconceptions_list = student_store.active_misconceptions(student_id)
        misconceptions = {m["id"] for m in misconceptions_list}

        prereq_titles: set[str] = set()
        if topic_id:
            prereq_titles = resolve_prerequisite_titles(
                conn, topic_id=topic_id, topics=topics_cfg
            )

        recent = episodic.recent(student_id, limit=10)
        recent_ids = {e.id for e in recent if e.id}

        # Build the prompt using the routing engine (same as TutorAgent.answer).
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
        logger.log({
            "event": "routing_decision",
            "agent": "tutor",
            "student_id": student_id,
            "task": prompt,
            "selected_ids": [it.id for it in decision.selected],
            "dropped_ids": [it.id for it in decision.dropped],
            "tokens_used": decision.tokens_used,
            "budget": decision.budget,
        })

        context_block = "\n\n".join(
            f"[{it.id}] {it.title}\n{it.body}" for it in decision.selected
        )
        user_prompt = (
            f"CONTEXT ITEMS:\n{context_block}\n\nSTUDENT QUESTION:\n{prompt}"
        )

        # Phase 2: stream the LLM response.
        chunks: list[str] = []
        for delta in llm.stream(
            system=TUTOR_SYSTEM, user=user_prompt, max_tokens=1024
        ):
            chunks.append(delta)
            yield delta
        full_text = "".join(chunks)

        # Phase 3: post-processing. Runs after the generator is fully consumed.
        logger.log({
            "event": "tutor_reply",
            "agent": "tutor",
            "student_id": student_id,
            "text": full_text,
        })

        for it in decision.selected:
            st.session_state.reuse_counts[it.id] += 1

        topic_key = topic_id or "_global"
        if topic_key not in st.session_state.seen_concepts_by_topic:
            st.session_state.seen_concepts_by_topic[topic_key] = set()
        for it in decision.selected:
            if getattr(it.artifact_type, "value", None) == "concept":
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
                "text": full_text,
                "selected_ids": [it.id for it in decision.selected],
                "tokens_used": decision.tokens_used,
            },
        )
        conn.commit()

        # Long-horizon memory write: ingest the user's turn under this chat's
        # stable conv_id so XTrace groups all turns into one Episode.
        xtrace = _xtrace_client()
        if xtrace is not None:
            xtrace.ingest_fact(
                student_id=student_id,
                text=prompt,
                conv_id=st.session_state.chat_session_id,
            )

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

    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------


def _topic_title(topic_id: str | None) -> str:
    """Human-readable title for a topic id. Falls back to title-cased id."""
    if not topic_id:
        return "Auto"
    for t in _topics():
        if t.id == topic_id:
            return t.title
    return topic_id.replace("_", " ").title()


def _format_topic_option(opt: str) -> str:
    """Selectbox format_func: titles for real topics, friendly label for auto."""
    if opt == "(auto)":
        return "Auto-detect topic"
    return _topic_title(opt)


def _topic_status_chip_html() -> str:
    """Small colored pill summarizing current topic / inference state.

    Green when a manual topic is pinned, blue for auto-detected, amber for
    inferred-with-medium-confidence, grey when the model can't decide.
    """
    if st.session_state.topic_choice != "(auto)":
        title = _topic_title(st.session_state.topic_choice)
        return (
            f"<span style='display:inline-block; padding:4px 10px; border-radius:999px; "
            f"background:#e7f0ff; color:#1f4ec7; font-size:0.85rem; font-weight:500;'>"
            f"Focused · {title}</span>"
        )

    inferred = st.session_state.get("last_inferred_topic")
    if not inferred or not inferred.get("topic_id"):
        return (
            "<span style='display:inline-block; padding:4px 10px; border-radius:999px; "
            "background:#f1f1f4; color:#666; font-size:0.85rem;'>"
            "Topic will be inferred from your message</span>"
        )
    decision = inferred.get("decision", "ask")
    title = _topic_title(inferred["topic_id"])
    conf = inferred["confidence"]
    if decision == "auto":
        bg, fg, label = "#e6f7ee", "#1a7f4a", f"Auto · {title} · {conf:.0%}"
    elif decision == "inferred":
        bg, fg, label = "#fff5e0", "#a05a00", f"Inferred · {title} · {conf:.0%}"
    else:
        bg, fg, label = "#f1f1f4", "#666", f"Uncertain · {conf:.0%} — I'll ask"
    return (
        f"<span style='display:inline-block; padding:4px 10px; border-radius:999px; "
        f"background:{bg}; color:{fg}; font-size:0.85rem; font-weight:500;'>{label}</span>"
    )


def _render_topic_header(*, expanded: bool) -> str | None:
    """In-content topic chooser; lives above the chat on every turn.

    Two layouts: expanded welcome card (no messages yet) and slim bar
    (during conversation). Writes ``st.session_state.topic_choice`` and
    returns the resolved topic_id (None == auto-detect at turn time).
    """
    topics = _topics()
    topic_options = ["(auto)"] + [t.id for t in topics]

    if expanded:
        with st.container(border=True):
            st.markdown(
                "<div style='padding:8px 4px 0 4px;'>"
                "<h2 style='margin:0 0 10px 0; color:var(--text); font-weight:600; font-size:1.6rem;'>"
                "Welcome — I'm Memex."
                "</h2>"
                "<p style='color:var(--text-muted); font-size:1.0rem; margin:0 0 18px 0; line-height:1.55;'>"
                "Your ML systems tutor. Pick a topic, or leave it on <strong>(auto)</strong> "
                "and I'll figure it out from what you ask. I keep answers concise, add a "
                "diagram when it helps, and remember what you've worked on across sessions."
                "</p>"
                "</div>",
                unsafe_allow_html=True,
            )
            st.selectbox(
                "Topic",
                topic_options,
                key="topic_choice",
                label_visibility="collapsed",
                format_func=_format_topic_option,
            )
            st.markdown(_topic_status_chip_html(), unsafe_allow_html=True)

        topic_id_for_starters = (
            None if st.session_state.topic_choice == "(auto)" else st.session_state.topic_choice
        )
        starters = _starter_prompts_for(topic_id_for_starters)
        if starters:
            st.markdown(
                "<div class='lm-section-label' style='margin:18px 0 8px 0;'>Jump in with</div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(len(starters))
            for i, (col, prompt) in enumerate(zip(cols, starters)):
                if col.button(prompt, key=f"starter_{i}"):
                    st.session_state.pending_prompt = prompt
                    st.rerun()
    else:
        with st.container(border=True):
            col_select, col_chip = st.columns([1, 2], vertical_alignment="center")
            with col_select:
                st.selectbox(
                    "Topic",
                    topic_options,
                    key="topic_choice",
                    label_visibility="collapsed",
                    format_func=_format_topic_option,
                )
            with col_chip:
                st.markdown(_topic_status_chip_html(), unsafe_allow_html=True)

    return None if st.session_state.topic_choice == "(auto)" else st.session_state.topic_choice


def _compute_topic_mastery(student_id: str) -> list[dict]:
    """Aggregate per-topic mastery from the Postgres mastery + misconceptions tables.

    For each topic we look up the topic's concept-type semantic items, then count
    how many of those concepts have a mastery entry for this student and what the
    average score is. Used to power the Profile tab.
    """
    out: list[dict] = []
    topics = _topics()
    conn = _new_conn()
    try:
        semantic = SemanticStore(conn)
        student = StudentStore(conn)
        mastery_by_id = {m.concept_id: m for m in student.mastery_for(student_id)}
        misc_list = student.active_misconceptions(student_id)
        misc_by_concept: dict[str, int] = {}
        for m in misc_list:
            cid = m.get("concept_id")
            if cid:
                misc_by_concept[cid] = misc_by_concept.get(cid, 0) + 1
        for t in topics:
            try:
                items = semantic.by_topic(t.id)
            except Exception:
                items = []
            concepts = [
                it
                for it in items
                if (getattr(it.artifact_type, "value", str(it.artifact_type or "")).lower()
                    == "concept")
            ]
            scores: list[float] = []
            n_assessed = 0
            n_misc = 0
            for c in concepts:
                if c.id in mastery_by_id:
                    scores.append(mastery_by_id[c.id].score)
                    n_assessed += 1
                if c.id in misc_by_concept:
                    n_misc += misc_by_concept[c.id]
            mean = sum(scores) / len(scores) if scores else None
            out.append(
                {
                    "topic_id": t.id,
                    "title": t.title,
                    "area": t.area,
                    "mean_score": mean,
                    "n_assessed": n_assessed,
                    "n_concepts": len(concepts),
                    "n_misc": n_misc,
                }
            )
    finally:
        conn.close()
    return out


def _compute_recommendations(topic_stats: list[dict]) -> list[dict]:
    """Generate 3-5 'work on this next' recommendations.

    Priorities, in order:
      1. Reinforce weak topics — anything assessed below 0.5.
      2. Unlock-ready topics — all prerequisites at >=0.7 mean mastery, but
         the topic itself has no mastery yet.
      3. Quick wins — small topics (few concepts) with zero mastery.

    Each recommendation has: topic_id, title, reason, kind.
    """
    by_id = {t["topic_id"]: t for t in topic_stats}
    topics_cfg = {t.id: t for t in _topics()}
    recs: list[dict] = []

    # 1. Weak topics (reinforce).
    weak = sorted(
        [t for t in topic_stats if t["mean_score"] is not None and t["mean_score"] < 0.5],
        key=lambda t: t["mean_score"],
    )
    for t in weak[:2]:
        recs.append(
            {
                "topic_id": t["topic_id"],
                "title": t["title"],
                "kind": "reinforce",
                "reason": (
                    f"Current mastery is {int(t['mean_score']*100)}% — a few more questions "
                    "here will lock it in."
                ),
            }
        )

    # 2. Unlock-ready topics (prerequisites mastered, this topic untouched).
    unlock_ready: list[dict] = []
    for t in topic_stats:
        if t["n_assessed"] > 0 or t["n_concepts"] == 0:
            continue
        prereqs = topics_cfg.get(t["topic_id"]).prerequisites if topics_cfg.get(t["topic_id"]) else []
        if not prereqs:
            continue
        prereq_scores = [
            by_id[p]["mean_score"]
            for p in prereqs
            if p in by_id and by_id[p]["mean_score"] is not None
        ]
        if prereq_scores and all(s >= 0.7 for s in prereq_scores):
            unlock_ready.append(
                {
                    "topic_id": t["topic_id"],
                    "title": t["title"],
                    "kind": "unlocked",
                    "reason": (
                        "Prerequisites are solid (≥70%). You're ready to move on to this."
                    ),
                }
            )
    for t in unlock_ready[:2]:
        recs.append(t)

    # 3. Quick wins — small topics, no progress, no prereqs (or shallow ones).
    if len(recs) < 3:
        unstarted_small = sorted(
            [
                t
                for t in topic_stats
                if t["n_assessed"] == 0 and 0 < t["n_concepts"] <= 5
            ],
            key=lambda t: t["n_concepts"],
        )
        for t in unstarted_small:
            if any(r["topic_id"] == t["topic_id"] for r in recs):
                continue
            recs.append(
                {
                    "topic_id": t["topic_id"],
                    "title": t["title"],
                    "kind": "quick_win",
                    "reason": (
                        f"Small topic ({t['n_concepts']} concepts) — a great way to "
                        "get a quick win."
                    ),
                }
            )
            if len(recs) >= 4:
                break

    return recs[:4]


def _mastery_band(score: float | None) -> str:
    """Return a CSS-class suffix (good/mid/low) for a mastery score."""
    if score is None:
        return "neutral"
    if score >= 0.7:
        return "good"
    if score >= 0.4:
        return "mid"
    return "low"


def _render_profile_tab(student_id: str):
    """Digital-student view: identity, per-topic mastery, strengths/weaknesses,
    active misconceptions, and long-term memory evidence from XTrace."""

    # --- Pull data ---
    topic_stats = _compute_topic_mastery(student_id)
    conn = _new_conn()
    try:
        student_store = StudentStore(conn)
        all_mastery = student_store.mastery_for(student_id)
        misconceptions = student_store.active_misconceptions(student_id)
    finally:
        conn.close()

    xtrace = _xtrace_client()
    memory_items = []
    if xtrace is not None:
        try:
            memory_items = xtrace.list_memories(student_id=student_id, limit=100)
        except Exception:
            memory_items = []

    episodes = [it for it in memory_items if it.kind == "episode"]
    facts = [it for it in memory_items if it.kind == "fact"]

    n_topics_touched = sum(1 for t in topic_stats if t["n_assessed"] > 0)
    n_mastered_concepts = sum(1 for m in all_mastery if m.score >= 0.8)
    n_misc = len(misconceptions)

    # --- Identity hero ---
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:16px; margin:0 0 18px 0;'>"
        f"<div style='width:48px; height:48px; border-radius:14px; "
        f"background:linear-gradient(135deg, var(--accent) 0%, #8b5cf6 100%); "
        f"display:flex; align-items:center; justify-content:center; color:#fff; "
        f"font-weight:600; font-size:1.2rem; box-shadow:var(--shadow-md);'>"
        f"{(student_id or '?')[0].upper()}</div>"
        f"<div>"
        f"<div style='font-size:1.4rem; font-weight:600; color:var(--text); line-height:1.1;'>"
        f"{student_id}</div>"
        f"<div style='font-size:0.9rem; color:var(--text-muted); margin-top:2px;'>"
        f"Digital learner profile · powered by Memex + long-term memory</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # --- Top-line metrics ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Topics touched", f"{n_topics_touched} / {len(topic_stats)}")
    c2.metric("Concepts mastered", n_mastered_concepts)
    c3.metric("Misconceptions", n_misc)
    c4.metric("Sessions saved", len(episodes))

    # --- Strengths / Weaknesses ---
    assessed = [t for t in topic_stats if t["mean_score"] is not None]
    if assessed:
        sorted_by_mastery = sorted(assessed, key=lambda x: x["mean_score"], reverse=True)
        strengths = [t for t in sorted_by_mastery if t["mean_score"] >= 0.6][:3]
        weaknesses = sorted([t for t in assessed if t["mean_score"] < 0.6], key=lambda x: x["mean_score"])[:3]

        col_s, col_w = st.columns(2)
        with col_s:
            st.markdown("<div class='lm-section-label'>Strengths</div>", unsafe_allow_html=True)
            if strengths:
                chips = "".join(
                    f"<span class='lm-chip lm-chip--success'>"
                    f"<span class='lm-chip-dot'></span>{t['title']} · {int(t['mean_score']*100)}%"
                    f"</span>"
                    for t in strengths
                )
                st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div style='color:var(--text-soft); font-size:0.9rem;'>"
                    "No topics above 60% yet — keep going.</div>",
                    unsafe_allow_html=True,
                )
        with col_w:
            st.markdown("<div class='lm-section-label'>Areas to focus on</div>", unsafe_allow_html=True)
            if weaknesses:
                chips = "".join(
                    f"<span class='lm-chip lm-chip--warn'>"
                    f"<span class='lm-chip-dot'></span>{t['title']} · {int(t['mean_score']*100)}%"
                    f"</span>"
                    for t in weaknesses
                )
                st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div style='color:var(--text-soft); font-size:0.9rem;'>"
                    "Everything assessed so far is solid.</div>",
                    unsafe_allow_html=True,
                )

    # --- Recommendations ---
    recommendations = _compute_recommendations(topic_stats)
    if recommendations:
        st.markdown(
            "<div class='lm-section-label'>Recommended for you</div>",
            unsafe_allow_html=True,
        )
        kind_meta = {
            "reinforce": ("Reinforce", "#fff5e0", "#a05a00"),
            "unlocked":  ("Unlocked", "#e6f7ee", "#1a7f4a"),
            "quick_win": ("Quick win", "#eef0ff", "#3730a3"),
        }
        rec_cols = st.columns(min(len(recommendations), 4))
        for i, rec in enumerate(recommendations):
            label, bg, fg = kind_meta.get(rec["kind"], ("Suggested", "#eef0ff", "#3730a3"))
            with rec_cols[i]:
                with st.container(border=True):
                    st.markdown(
                        f"<div style='display:inline-block; padding:3px 9px; border-radius:999px; "
                        f"background:{bg}; color:{fg}; font-size:0.72rem; font-weight:600; "
                        f"letter-spacing:0.04em; text-transform:uppercase; margin-bottom:8px;'>"
                        f"{label}</div>"
                        f"<div style='font-weight:600; color:var(--text); font-size:0.95rem; "
                        f"margin-bottom:6px; line-height:1.35;'>{rec['title']}</div>"
                        f"<div style='font-size:0.85rem; color:var(--text-muted); line-height:1.5;'>"
                        f"{rec['reason']}</div>",
                        unsafe_allow_html=True,
                    )
                    if st.button(
                        "Study this →",
                        key=f"rec_{rec['topic_id']}_{i}",
                        use_container_width=True,
                    ):
                        st.session_state.topic_choice = rec["topic_id"]
                        st.session_state.pending_prompt = (
                            f"Give me a quick intro to {rec['title']}."
                        )
                        st.session_state.active_tab = "Chat"
                        st.rerun()

    # --- Active misconceptions ---
    if misconceptions:
        st.markdown(
            "<div class='lm-section-label'>Active misconceptions</div>",
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            for m in misconceptions[:8]:
                desc = (m.get("description") or "").strip()
                if not desc:
                    continue
                st.markdown(
                    f"<div style='display:flex; gap:10px; align-items:flex-start; "
                    f"padding:8px 0; border-bottom:1px solid var(--border-soft);'>"
                    f"<span style='flex-shrink:0; width:8px; height:8px; border-radius:50%; "
                    f"background:var(--danger); margin-top:7px;'></span>"
                    f"<div style='color:var(--text); font-size:0.92rem; line-height:1.5;'>{desc}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # --- Per-topic mastery breakdown ---
    st.markdown(
        "<div class='lm-section-label'>Topic-by-topic mastery</div>",
        unsafe_allow_html=True,
    )

    # Sort: assessed-and-strong first, then assessed-and-weak, then unstarted.
    def sort_key(t: dict) -> tuple:
        if t["mean_score"] is None:
            return (1, 0.0)
        return (0, -t["mean_score"])

    sorted_topics = sorted(topic_stats, key=sort_key)

    if not any(t["n_concepts"] > 0 for t in sorted_topics):
        st.markdown(
            "<div style='color:var(--text-soft); padding:12px 0;'>"
            "No semantic items found — run the ingestion script to populate topics.</div>",
            unsafe_allow_html=True,
        )
    else:
        for t in sorted_topics:
            if t["n_concepts"] == 0:
                continue  # Skip topics with no concepts in the DB.
            score = t["mean_score"]
            band = _mastery_band(score)
            fill_class = {
                "good": "lm-meter-fill--good",
                "mid": "lm-meter-fill--mid",
                "low": "lm-meter-fill--low",
                "neutral": "lm-meter-fill--good",
            }[band]
            pct = int((score or 0) * 100)
            score_label = (
                f"{pct}%"
                if score is not None
                else "<span style='color:var(--text-soft);'>not started</span>"
            )
            assessed_label = (
                f"{t['n_assessed']} / {t['n_concepts']} concepts"
                if t["n_assessed"]
                else f"0 / {t['n_concepts']} concepts"
            )
            misc_badge = (
                f"<span class='lm-chip lm-chip--danger' style='margin-left:8px;'>"
                f"<span class='lm-chip-dot'></span>{t['n_misc']} misc</span>"
                if t["n_misc"]
                else ""
            )
            st.markdown(
                f"<div style='padding:10px 0; border-bottom:1px solid var(--border-soft);'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>"
                f"<div style='font-weight:500; color:var(--text); font-size:0.95rem;'>{t['title']}{misc_badge}</div>"
                f"<div style='font-size:0.85rem; color:var(--text-muted);'>{score_label}</div>"
                f"</div>"
                + (
                    f"<div class='lm-meter'>"
                    f"<div class='lm-meter-fill {fill_class}' style='width:{pct}%'></div>"
                    f"</div>"
                    if score is not None
                    else "<div class='lm-meter'><div class='lm-meter-fill' style='width:0%'></div></div>"
                )
                + f"<div style='font-size:0.78rem; color:var(--text-soft); margin-top:2px;'>{assessed_label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # --- Long-term memory evidence (XTrace) ---
    if xtrace is None:
        st.markdown(
            "<div style='margin-top:24px; padding:14px 18px; background:var(--surface-2); "
            "border-radius:var(--radius); font-size:0.88rem; color:var(--text-muted);'>"
            "Set <code>XTRACE_API_KEY</code> and <code>XTRACE_ORG_ID</code> to enable "
            "long-term memory and see session history here."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        "<div class='lm-section-label'>Recent activity</div>",
        unsafe_allow_html=True,
    )

    if not memory_items:
        st.markdown(
            "<div style='margin:8px 0; padding:32px 20px; text-align:center; "
            "background:var(--surface); border:1px dashed var(--border); "
            "border-radius:var(--radius); color:var(--text-muted);'>"
            "<div style='font-size:1.6rem; margin-bottom:6px;'>🧠</div>"
            "<div style='font-size:0.95rem; color:var(--text); font-weight:500;'>"
            "No long-term memory yet.</div>"
            "<div style='font-size:0.88rem; margin-top:2px;'>"
            "Chat in the <strong>Chat</strong> tab and your sessions will land here.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if episodes:
        for i, ep in enumerate(reversed(episodes), 1):
            with st.container(border=True):
                st.markdown(
                    f"<div style='font-weight:600; color:var(--text); font-size:0.95rem; "
                    f"margin-bottom:10px;'>Session #{len(episodes) - i + 1}</div>"
                    f"<div style='color:var(--text); line-height:1.6; font-size:0.95rem;'>{ep.text}</div>",
                    unsafe_allow_html=True,
                )

    if facts:
        st.markdown(
            "<div class='lm-section-label'>Things the tutor remembers about you</div>",
            unsafe_allow_html=True,
        )
        for f in facts:
            with st.container(border=True):
                st.markdown(
                    f"<div style='color:var(--text); line-height:1.55; font-size:0.95rem;'>{f.text}</div>",
                    unsafe_allow_html=True,
                )


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

    # Left sidebar (sidebar no longer owns topic selection).
    student_id, topic_id, budget = _render_left_sidebar()

    # Session-state driven tab nav (st.tabs has no programmatic switch API).
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Chat"

    nav_chat, nav_profile, _nav_spacer = st.columns([1, 1, 6])
    active = st.session_state.active_tab
    if nav_chat.button(
        "Chat",
        key="nav_chat",
        use_container_width=True,
        type="primary" if active == "Chat" else "secondary",
    ):
        st.session_state.active_tab = "Chat"
        st.rerun()
    if nav_profile.button(
        "Profile",
        key="nav_profile",
        use_container_width=True,
        type="primary" if active == "Profile" else "secondary",
    ):
        st.session_state.active_tab = "Profile"
        st.rerun()
    st.markdown(
        "<div style='border-bottom:1px solid var(--border-soft); margin:6px 0 18px 0;'></div>",
        unsafe_allow_html=True,
    )

    if st.session_state.active_tab == "Profile":
        _render_profile_tab(student_id)
        return

    if True:
        # In-content topic chooser. Expanded layout when chat is empty,
        # slim layout once a conversation is going.
        topic_id = _render_topic_header(expanded=not st.session_state.messages)

        # Three-zone layout inside Chat: main chat (3) + right drawer (1)
        main_col, right_col = st.columns([3, 1])

        with main_col:
            if st.session_state.messages:
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
                    placeholder = st.empty()
                    placeholder.markdown("_Selecting context…_")
                    reply = ""
                    for delta in _handle_turn(prompt, student_id, topic_id, budget):
                        reply += delta
                        # Show streaming text with a typing cursor.
                        placeholder.markdown(reply + "▌")
                    # Generator exhausted → post-processing has run, last_decision is set.
                    selected_items = (
                        st.session_state.last_decision.get("selected", [])
                        if st.session_state.last_decision
                        else []
                    )
                    rendered_reply = _render_citations(reply, selected_items)
                    placeholder.empty()
                    _render_with_mermaid(rendered_reply)
                    new_msg_idx = len(st.session_state.messages)
                    _render_quiz_for_message(new_msg_idx, topic_id, student_id)

                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()

            _render_context_analysis()

        with right_col:
            _render_right_pane(right_col, topic_id, student_id)


if __name__ == "__main__":
    main()
