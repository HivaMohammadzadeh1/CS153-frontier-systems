"""FastAPI REST API for Learning Memory OS.

Wraps the existing Python backend (TutorAgent, RoutingEngine, stores, quiz harness)
in HTTP endpoints. Serves the static frontend from /web.
"""

import json
import re
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from learning_memory_os.auth import COOKIE_NAME, AuthStore

from learning_memory_os.config import get_settings
from learning_memory_os.llm import LLM
from learning_memory_os.embeddings import Embedder
from learning_memory_os.memory.store import connect
from learning_memory_os.memory.semantic import SemanticStore
from learning_memory_os.memory.student import StudentStore
from learning_memory_os.memory.episodic import EpisodicStore
from learning_memory_os.memory.conversation import ConversationStore
from learning_memory_os.selector.engine import RoutingEngine
from learning_memory_os.agents.tutor import TutorAgent
from learning_memory_os.agents.profile import build_profile
from learning_memory_os.memory.trace import TraceStore
from learning_memory_os.logging_utils.interactions import InteractionLogger
from learning_memory_os.ingestion.topic_loader import load_topics, resolve_prerequisite_titles
from learning_memory_os.eval.quiz import QuizQuestion, score_answer
from learning_memory_os.memory.decay import effective_score

# Project root: two levels up from src/learning_memory_os/api.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---- Schemas for tool-use ----
QUIZ_QUESTION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "rubric": {"type": "string"},
    },
    "required": ["question", "rubric"],
}

DIAGNOSTIC_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string"},
        "follow_up_question": {"type": "string"},
    },
    "required": ["diagnosis", "follow_up_question"],
}

EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "confirmed_misconception": {"type": "string"},
        "explanation": {"type": "string"},
        "next_action": {"type": "string", "enum": ["explain", "re_test", "wrap_up"]},
        "next_message": {"type": "string"},
    },
    "required": ["confirmed_misconception", "explanation", "next_action", "next_message"],
}


# ---- App ----
app = FastAPI(title="Learning Memory OS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _settings():
    return get_settings()


# ---- Auth gating ----
# Public API paths that never require a session; everything else under /api/ does.
_PUBLIC_API = ("/api/health", "/api/topics", "/api/info", "/api/routers", "/api/auth/", "/api/billing/webhook", "/api/pricing")
_STUDENT_PATH_RE = re.compile(r"^/api/student/([^/]+)")


def _session_user(request: Request) -> Optional[dict]:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    conn = connect(_settings().database_url)
    try:
        return AuthStore(conn).session_user(token)
    finally:
        conn.close()


def _username_from_request(request: Request) -> Optional[str]:
    su = _session_user(request)
    return su["username"] if su else None


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    """Require a valid session for all /api/ data routes (except the public list),
    enforce that /api/student/{id}/... matches the caller, and — when billing is
    enabled — require a paid (is_pro) account for everything except the billing routes."""
    from learning_memory_os import billing
    path = request.url.path
    if path.startswith("/api/") and not path.startswith(_PUBLIC_API):
        su = _session_user(request)
        if not su:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        request.state.username = su["username"]
        request.state.is_pro = bool(su.get("is_pro"))
        m = _STUDENT_PATH_RE.match(path)
        if m and unquote(m.group(1)) != su["username"]:
            return JSONResponse({"detail": "Forbidden"}, status_code=403)
        # Hard paywall: must have paid to use the platform (billing routes exempt).
        if billing.is_enabled() and not su.get("is_pro") and not path.startswith("/api/billing/"):
            return JSONResponse({"detail": "Payment required"}, status_code=402)
    return await call_next(request)


def _llm_and_embedder(model: Optional[str] = None):
    s = _settings()
    llm = LLM(api_key=s.anthropic_api_key, model=model) if model else LLM(api_key=s.anthropic_api_key)
    return llm, Embedder(api_key=s.openai_api_key)


_TOPICS = load_topics(_PROJECT_ROOT / "config" / "topics.yaml")

_AREA_NAMES: dict[str, str] = (
    yaml.safe_load((_PROJECT_ROOT / "config" / "topics.yaml").read_text()) or {}
).get("areas", {})


# ---- Auth endpoints ----
class SignupRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    login: str          # username OR email
    password: str


def _set_session_cookie(response: Response, token: str) -> None:
    s = _settings()
    response.set_cookie(
        COOKIE_NAME, token,
        httponly=True, samesite="lax", secure=s.cookie_secure,
        max_age=s.session_ttl_days * 86400, path="/",
    )


@app.post("/api/auth/signup")
def auth_signup(req: SignupRequest, response: Response):
    conn = connect(_settings().database_url)
    try:
        store = AuthStore(conn)
        try:
            user = store.create_user(req.username, req.email, req.password)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        token = store.create_session(user["id"], user["username"], ttl_days=_settings().session_ttl_days)
        conn.commit()
    finally:
        conn.close()
    _set_session_cookie(response, token)
    return {"username": user["username"]}


@app.post("/api/auth/login")
def auth_login(req: LoginRequest, response: Response):
    conn = connect(_settings().database_url)
    try:
        store = AuthStore(conn)
        user = store.verify_login(req.login, req.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        token = store.create_session(user["id"], user["username"], ttl_days=_settings().session_ttl_days)
        conn.commit()
    finally:
        conn.close()
    _set_session_cookie(response, token)
    return {"username": user["username"]}


@app.post("/api/auth/logout")
def auth_logout(request: Request, response: Response):
    conn = connect(_settings().database_url)
    try:
        AuthStore(conn).delete_session(request.cookies.get(COOKIE_NAME))
        conn.commit()
    finally:
        conn.close()
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def auth_me(request: Request):
    from learning_memory_os import billing
    su = _session_user(request)
    if not su:
        raise HTTPException(status_code=401, detail="Not authenticated")
    billing_enabled = billing.is_enabled()
    # When billing is off (dev/local), everyone has access.
    is_pro = bool(su.get("is_pro")) or not billing_enabled
    return {"username": su["username"], "is_pro": is_pro, "billing_enabled": billing_enabled}


# ---- Request/response models ----

class ChatRequest(BaseModel):
    student_id: str
    conversation_id: Optional[str] = None
    topic_id: Optional[str] = None
    question: str
    budget: int = 3000
    reuse_counts: dict[str, int] = {}
    router: Optional[str] = None   # None/"heuristic" or a finetuned size id e.g. "qwen2_5_7b"


class ChatReference(BaseModel):
    n: int
    id: str
    title: str


class ChatItem(BaseModel):
    id: str
    title: str
    body: str
    token_estimate: int
    score_total: float = 0.0
    score_relevance: float = 0.0
    score_recency: float = 0.0
    score_misconception: float = 0.0
    score_prerequisite: float = 0.0
    score_reuse: float = 0.0


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str                       # tutor text with [n]-style references already substituted
    references: list[ChatReference]  # ordered references
    selected: list[ChatItem]
    dropped: list[ChatItem]
    budget: int
    tokens_used: int
    router: str = "heuristic"        # which router actually produced the selection


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/topics")
def list_topics():
    return [
        {"id": t.id, "title": t.title, "area": t.area,
         "area_title": _AREA_NAMES.get(t.area, "")}
        for t in _TOPICS
    ]


class StoredMessage(BaseModel):
    role: str            # "user" | "assistant"
    content: str
    timestamp: str       # ISO 8601


class StoredMessagesResponse(BaseModel):
    messages: list[StoredMessage]


@app.get("/api/student/{student_id}/messages", response_model=StoredMessagesResponse)
def student_messages(student_id: str, limit: int = 40):
    """Return the most recent question/tutor_reply events for a student, oldest-first,
    formatted as renderable chat messages."""
    conn = connect(_settings().database_url)
    try:
        student = StudentStore(conn)
        student.ensure_student(student_id)
        episodic = EpisodicStore(conn)
        # recent() returns DESC; we want oldest-first for chat replay
        events = list(episodic.recent(student_id, limit=limit))
        events.reverse()

        msgs: list[StoredMessage] = []
        for ev in events:
            if ev.event_type == "question":
                text = (ev.payload or {}).get("text", "")
                if text:
                    msgs.append(StoredMessage(
                        role="user",
                        content=text,
                        timestamp=ev.occurred_at.isoformat() if ev.occurred_at else "",
                    ))
            elif ev.event_type == "tutor_reply":
                text = (ev.payload or {}).get("text", "")
                if text:
                    msgs.append(StoredMessage(
                        role="assistant",
                        content=text,
                        timestamp=ev.occurred_at.isoformat() if ev.occurred_at else "",
                    ))
            # Ignore other event types for now
        return StoredMessagesResponse(messages=msgs)
    finally:
        conn.close()


@app.get("/api/routers")
def list_routers():
    """Available context-router backends: the heuristic engine plus any
    fine-tuned LoRA adapters present on disk."""
    try:
        from learning_memory_os.router.product_adapter import available_sizes
        sizes = available_sizes()
    except Exception:
        sizes = []
    return {"heuristic": True, "finetuned": sizes}


@app.get("/api/info")
def info():
    return {
        "tutor_model": "claude-opus-4-7",
        "embedding_model": "text-embedding-3-small",
    }


class CheckoutRequest(BaseModel):
    source: Optional[str] = None


@app.post("/api/billing/checkout")
def billing_checkout(req: CheckoutRequest, request: Request):
    """Start a Pro upgrade. Logs the click-to-pay signal and returns a Stripe
    checkout/payment URL if billing is configured (else null -> client shows a
    waitlist, still capturing intent)."""
    from learning_memory_os import billing
    username = getattr(request.state, "username", None)
    try:
        InteractionLogger(path=_settings().log_dir / "interactions.jsonl").log(
            {"event": "upgrade_intent", "student_id": username, "source": req.source}
        )
    except Exception:
        pass
    return {"checkout_url": billing.checkout_url(username) if username else None}


@app.post("/api/billing/webhook")
async def billing_webhook(request: Request):
    """Stripe webhook — on successful checkout, grant Pro to the paying user
    (matched by client_reference_id = username). Public route (no session)."""
    from learning_memory_os import billing
    payload = await request.body()
    event = billing.parse_webhook(payload, request.headers.get("stripe-signature"))
    if not event:
        raise HTTPException(status_code=400, detail="invalid webhook")
    username = billing.username_from_event(event)
    if username:
        conn = connect(_settings().database_url)
        try:
            AuthStore(conn).set_pro(username, True)
            conn.commit()
        finally:
            conn.close()
    return {"received": True}


# Calibrated readiness tiers (expert hire-bar mapping).
_READINESS_TIERS = [
    (90, "frontier", "Frontier-ready — would clear the bar at a top inference team"),
    (80, "ready", "Interview-ready — hireable on this evidence"),
    (70, "borderline", "Borderline — close, but tighten the weak areas"),
    (60, "not_ready", "Not ready yet — real gaps to close"),
    (0, "remediation", "Remediation — build the fundamentals first"),
]
_CRITICAL_FAIL_THRESHOLD = 60  # a critical category below this blocks "ready"


def _tier_for(score: float) -> tuple[str, str]:
    for floor, key, label in _READINESS_TIERS:
        if score >= floor:
            return key, label
    return "remediation", _READINESS_TIERS[-1][2]


def _readiness_verdict(conn, student_id: str) -> dict:
    """Hire-bar verdict from the student's interview history. Returns a tier, a
    blended 0-100 score, and the critical-failure gate. Empty/!ready when there
    isn't enough interview signal yet (expert: ready = avg>=80 over >=3 interviews
    AND no critical failure)."""
    import statistics
    from learning_memory_os.agents.interview_prompts import CRITICAL_CATEGORIES

    rows = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT overall_score, evaluation FROM interview_evaluations "
                "WHERE student_id = %s ORDER BY occurred_at DESC LIMIT 5",
                (student_id,),
            )
            rows = cur.fetchall()
    except Exception:
        rows = []  # table not present yet

    n = len(rows)
    if n == 0:
        return {
            "tier": "no_data", "label": "Take a mock interview to get your readiness verdict",
            "score": None, "interview_count": 0, "interview_avg": None,
            "critical_failures": [], "interview_ready": False, "consistency": None,
            "trajectory": None,
        }

    scores = [float(r["overall_score"] or 0) for r in rows]  # most-recent first
    avg = sum(scores) / n

    # trajectory: recent vs older half (chronological); reward improvement
    chrono = list(reversed(scores))
    if n >= 2:
        half = max(1, n // 2)
        older = sum(chrono[:half]) / half
        recent = sum(chrono[-half:]) / half
        trajectory = max(0.0, min(100.0, 70.0 + (recent - older)))
    else:
        trajectory = avg

    # consistency: tight clustering = trustworthy; high variance = risky
    consistency = max(0.0, 100.0 - statistics.pstdev(scores)) if n >= 2 else avg

    blended = 0.6 * avg + 0.2 * trajectory + 0.2 * consistency

    # critical-failure gate from the MOST RECENT interview's category scores
    crit_fail = []
    latest = rows[0]["evaluation"] or {}
    cats = latest.get("category_scores", {}) if isinstance(latest, dict) else {}
    for c in CRITICAL_CATEGORIES:
        if c in cats and float(cats[c] or 0) < _CRITICAL_FAIL_THRESHOLD:
            crit_fail.append({"category": c, "score": int(cats[c] or 0)})

    interview_ready = (n >= 3) and (avg >= 80) and not crit_fail

    tier, label = _tier_for(blended)
    if crit_fail and tier in ("frontier", "ready"):
        tier, label = "not_ready", "Not ready — a load-bearing skill is failing"
    if n < 3 and tier in ("frontier", "ready"):
        label = f"{label} (need {3 - n} more interview{'s' if 3 - n != 1 else ''} to confirm)"

    return {
        "tier": tier, "label": label, "score": round(blended, 1),
        "interview_count": n, "interview_avg": round(avg, 1),
        "consistency": round(consistency, 1), "trajectory": round(trajectory, 1),
        "critical_failures": crit_fail, "interview_ready": interview_ready,
    }


@app.get("/api/student/{student_id}/readiness")
def student_readiness(student_id: str):
    """Interview-readiness report: per-area readiness %, top gaps, and what to
    drill next — computed against the FULL curriculum (untouched topics count as
    0, because an interview covers the whole area). Plus an over-time trend if
    mastery history exists. No LLM calls."""
    from collections import defaultdict
    conn = connect(_settings().database_url)
    try:
        student = StudentStore(conn)
        student.ensure_student(student_id)
        mastery = student.mastery_for(student_id)

        # concept_id -> topic_id
        ids = [m.concept_id for m in mastery]
        concept_topic: dict[str, str] = {}
        if ids:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id::text AS id, topic_id FROM semantic_items WHERE id::text = ANY(%s)",
                    (ids,),
                )
                concept_topic = {r["id"]: r["topic_id"] for r in cur.fetchall()}

        # confidence-weighted mastery per topic
        by_topic: dict[str, list] = defaultdict(list)
        for m in mastery:
            t = concept_topic.get(m.concept_id)
            if t:
                by_topic[t].append((m.score, m.confidence))
        topic_mastery: dict[str, float] = {}
        for t, lst in by_topic.items():
            tot = sum(c for _, c in lst) or 1e-6
            topic_mastery[t] = sum(s * c for s, c in lst) / tot

        # curriculum grouped by area (the full thing an interview covers)
        areas_topics: dict[str, list] = defaultdict(list)
        for tp in _TOPICS:
            areas_topics[tp.area].append(tp.id)

        area_readiness = []
        for area in sorted(areas_topics):
            tids = areas_topics[area]
            r = sum(topic_mastery.get(t, 0.0) for t in tids) / max(1, len(tids))
            area_readiness.append({
                "area": area,
                "area_title": _AREA_NAMES.get(area, ""),
                "readiness": round(r, 3),
                "covered": sum(1 for t in tids if t in topic_mastery),
                "total": len(tids),
            })
        overall = round(sum(a["readiness"] for a in area_readiness) / max(1, len(area_readiness)), 3)

        # gaps: every topic below interview bar (0.6), unstudied counts as 0
        gaps = []
        for tp in _TOPICS:
            score = topic_mastery.get(tp.id, 0.0)
            if score < 0.6:
                gaps.append({
                    "topic_id": tp.id, "title": tp.title, "area": tp.area,
                    "area_title": _AREA_NAMES.get(tp.area, ""),
                    "mastery": round(score, 3), "started": tp.id in topic_mastery,
                })
        gaps.sort(key=lambda g: (g["mastery"], 0 if g["started"] else 1))
        # next-up: prefer an in-progress weak topic, else the weakest gap
        in_progress = [g for g in gaps if g["started"]]
        next_up = (in_progress[0] if in_progress else (gaps[0] if gaps else None))

        # over-time trend (overall readiness), if history is being recorded
        trend = []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT to_char(occurred_at::date,'YYYY-MM-DD') AS d, "
                    "       round(avg(score)::numeric,3) AS m "
                    "FROM mastery_history WHERE student_id = %s "
                    "GROUP BY occurred_at::date ORDER BY occurred_at::date",
                    (student_id,),
                )
                trend = [{"date": r["d"], "avg_mastery": float(r["m"])} for r in cur.fetchall()]
        except Exception:
            trend = []  # mastery_history table not present yet

        # Interview-readiness VERDICT — calibrated to a real hire bar, not raw mastery.
        # Blend = 60% interview avg + 20% trajectory + 20% consistency, with a hard
        # critical-failure gate: a weak score in any of the 4 load-bearing categories
        # blocks "ready" no matter how high the average (expert calibration).
        verdict = _readiness_verdict(conn, student_id)

        # Pro is disabled for now — everything is free. (Gating logic kept below so
        # it can be re-enabled by flipping this back to AuthStore(conn).is_pro(...).)
        pro = True
        return {
            "overall_readiness": overall,
            "areas": area_readiness,
            "verdict": verdict,
            "gaps": gaps[:8] if pro else gaps[:1],
            "gaps_total": len(gaps),
            "next_up": next_up,
            "concepts_mastered": sum(1 for v in topic_mastery.values() if v >= 0.7),
            "topics_covered": len(topic_mastery),
            "topics_total": len(_TOPICS),
            "trend": trend if pro else [],
            "pro": pro,
        }
    finally:
        conn.close()


@app.get("/api/student/{student_id}/interviews")
def student_interviews(student_id: str):
    """History of mock interviews / debugging / forward-deployed sessions, most
    recent first — so students can see what they've done and how they're trending."""
    title_by_id = {t.id: t.title for t in _TOPICS}
    conn = connect(_settings().database_url)
    try:
        rows = []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id::text, topic_id, level, overall_score, evaluation, "
                    "       to_char(occurred_at, 'YYYY-MM-DD\"T\"HH24:MI:SS') AS ts "
                    "FROM interview_evaluations WHERE student_id = %s "
                    "ORDER BY occurred_at DESC LIMIT 50",
                    (student_id,),
                )
                rows = cur.fetchall()
        except Exception:
            rows = []  # table not present yet
    finally:
        conn.close()

    out = []
    for r in rows:
        ev = r["evaluation"] if isinstance(r["evaluation"], dict) else {}
        cats = ev.get("category_scores", {}) if isinstance(ev, dict) else {}
        weakest = min(cats.items(), key=lambda kv: kv[1])[0] if cats else None
        out.append({
            "id": r["id"],
            "topic_id": r["topic_id"],
            "topic_title": title_by_id.get(r["topic_id"], r["topic_id"] or "ML systems"),
            "level": r["level"],
            "kind": ev.get("kind", "interview"),
            "turns": ev.get("turns"),
            "overall_score": r["overall_score"],
            "weakest": weakest,
            "next_topic": ev.get("next_topic"),
            "occurred_at": r["ts"],
        })
    avg = round(sum(o["overall_score"] or 0 for o in out) / len(out), 1) if out else None
    return {"interviews": out, "count": len(out), "avg_score": avg}


@app.get("/api/student/{student_id}/state")
def student_state(student_id: str):
    conn = connect(_settings().database_url)
    try:
        student = StudentStore(conn)
        student.ensure_student(student_id)
        mastery = [
            {"concept_id": m.concept_id, "score": m.score, "confidence": m.confidence}
            for m in student.mastery_for(student_id)
        ]
        misconceptions = [
            {"id": m["id"], "description": m["description"]}
            for m in student.active_misconceptions(student_id)
        ]
        return {"mastery": mastery, "misconceptions": misconceptions}
    finally:
        conn.close()


def _substitute_citations(reply_text: str, id_to_title: dict[str, str]) -> tuple[str, list[ChatReference]]:
    """Find [a1b2c3d4] short ids in reply_text, replace with [1] [2] [3] in order.

    Only ids that resolve to a real title (via ``id_to_title``) are numbered and
    surfaced as references; markers that can't be resolved (e.g. a hallucinated
    id) are stripped so the student never sees a raw hex id as a "source".
    """
    pattern = re.compile(r"\[([a-f0-9]{8})\]")

    ids_in_order: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(reply_text):
        cid = match.group(1)
        if cid in seen:
            continue
        seen.add(cid)
        if cid in id_to_title:           # only number resolvable sources
            ids_in_order.append(cid)

    id_to_n = {cid: i + 1 for i, cid in enumerate(ids_in_order)}

    def _repl(m: re.Match) -> str:
        cid = m.group(1)
        return f"[{id_to_n[cid]}]" if cid in id_to_n else ""

    new_text = pattern.sub(_repl, reply_text)
    # Tidy up any spacing left by a stripped marker (e.g. "foo  ." -> "foo.")
    new_text = re.sub(r"[ \t]{2,}", " ", new_text)
    new_text = re.sub(r"\s+([.,;:])", r"\1", new_text)

    refs = [ChatReference(n=id_to_n[cid], id=cid, title=id_to_title[cid]) for cid in ids_in_order]
    return new_text, refs


TITLE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "A 3-6 word title summarizing the topic."},
    },
    "required": ["title"],
}


def _prepare_turn(conn, req: ChatRequest, llm, embedder, engine, logger) -> dict:
    """Everything a chat turn needs *before* generation: conversation, candidate
    pool, context selection (heuristic or fine-tuned router), profile, and the
    built prompt. Shared by /api/chat and /api/chat/stream."""
    student = StudentStore(conn)
    student.ensure_student(req.student_id)
    semantic = SemanticStore(conn)
    episodic = EpisodicStore(conn)
    convs = ConversationStore(conn)

    conv_was_created = False
    conversation_id = req.conversation_id
    if conversation_id is None:
        conversation_id = convs.create(req.student_id, title="New chat")
        conv_was_created = True

    # Embed the question ONCE and reuse it for both retrieval and routing (was being
    # embedded twice per turn — an extra OpenAI round-trip on every message).
    q_emb = embedder.embed_one(req.question)
    if req.topic_id:
        candidates = semantic.by_topic(req.topic_id)
    else:
        candidates = semantic.vector_search(query=q_emb, k=20)

    active_misc = student.active_misconceptions(req.student_id)
    misc_concept_ids = {m["concept_id"] for m in active_misc if m.get("concept_id")}
    misc_topics: set[str] = {m["topic_id"] for m in active_misc if m.get("topic_id")}
    if misc_concept_ids:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT topic_id FROM semantic_items WHERE id::text = ANY(%s)",
                (list(misc_concept_ids),),
            )
            misc_topics |= {r["topic_id"] for r in cur.fetchall()}
    due_concept_ids = set(student.due_for_review(req.student_id))
    prereq_titles = (
        resolve_prerequisite_titles(conn, topic_id=req.topic_id, topics=_TOPICS)
        if req.topic_id else set()
    )
    recent = episodic.recent(req.student_id, limit=10)
    recent_ids = {e.id for e in recent if e.id}

    profile = build_profile(conn, req.student_id)
    mastery_records = student.mastery_for(req.student_id)

    router_used = "heuristic"
    preselected = None
    if req.router and req.router != "heuristic":
        from learning_memory_os.router.product_adapter import finetuned_select, available_sizes
        if req.router in available_sizes():
            try:
                preselected = finetuned_select(
                    size_id=req.router, student=student, student_id=req.student_id,
                    question=req.question, candidates=candidates, budget=req.budget,
                )
                router_used = req.router
            except Exception as e:  # noqa: BLE001
                logger.log({"event": "finetuned_router_failed", "router": req.router, "error": str(e)})
                preselected = None
        else:
            logger.log({"event": "finetuned_router_unavailable", "router": req.router})

    decision = engine.route(
        candidates=candidates, task_embedding=q_emb,
        misconception_concept_ids=misc_concept_ids, misconception_topics=misc_topics,
        prerequisites=prereq_titles, recent_ids=recent_ids,
        reuse_counts=dict(req.reuse_counts), due_concept_ids=due_concept_ids, budget=req.budget,
    )
    selected_items = preselected if preselected is not None else decision.selected
    system, user_prompt = TutorAgent.build_prompt(
        req.question, selected_items,
        weak_concepts=profile.weaknesses or None, strong_concepts=profile.strengths or None,
        active_misconception_texts=profile.misconceptions or None,
        due_concepts=profile.due_for_review or None,
        learning_style=profile.learning_style or None,
    )
    return {
        "conversation_id": conversation_id, "conv_was_created": conv_was_created,
        "candidates": candidates, "decision": decision, "preselected": preselected,
        "selected_items": selected_items, "router_used": router_used,
        "profile": profile, "mastery_records": mastery_records,
        "recent_ids": recent_ids, "due_concept_ids": due_concept_ids,
        "system": system, "user_prompt": user_prompt,
    }


def _finalize_turn(conn, req: ChatRequest, llm, logger, p: dict, reply_text: str) -> dict:
    """Everything *after* generation: capture the trace, log episodic events,
    auto-title, soft mastery bump, and resolve citations. Returns ChatResponse fields."""
    student = StudentStore(conn)
    episodic = EpisodicStore(conn)
    convs = ConversationStore(conn)
    decision = p["decision"]
    candidates = p["candidates"]
    selected_items = p["selected_items"]
    preselected = p["preselected"]
    conversation_id = p["conversation_id"]
    profile = p["profile"]
    mastery_records = p["mastery_records"]
    tokens_used = sum((getattr(it, "token_estimate", 0) or 0) for it in selected_items)

    try:
        TraceStore(conn).record_turn(
            student_id=req.student_id, conversation_id=conversation_id,
            task_text=req.question, budget=req.budget,
            student_state={
                "mastery": {m.concept_id: round(m.score, 3) for m in mastery_records},
                "active_misconceptions": profile.misconceptions,
                "recent_episodic_ids": list(p["recent_ids"]),
                "due_concept_ids": list(p["due_concept_ids"]),
            },
            candidate_pool=[
                {"id": it.id, "title": it.title,
                 "body_excerpt": (it.body or "")[:200], "token_estimate": it.token_estimate}
                for it in candidates
            ],
            selected_ids=[it.id for it in decision.selected],
            dropped_ids=[it.id for it in decision.dropped],
            scores={k: round(v.total, 4) for k, v in decision.scores.items()},
            reply=reply_text, model=getattr(llm, "model", None),
        )
    except Exception as e:  # noqa: BLE001 — capture is best-effort
        logger.log({"event": "trace_capture_failed", "student_id": req.student_id, "error": str(e)})

    episodic.append(
        student_id=req.student_id, event_type="question",
        payload={"text": req.question, "topic_id": req.topic_id, "source": "api"},
        conversation_id=conversation_id,
    )
    episodic.append(
        student_id=req.student_id, event_type="tutor_reply",
        payload={"text": reply_text, "selected_ids": [it.id for it in selected_items],
                 "tokens_used": tokens_used},
        conversation_id=conversation_id,
    )
    convs.touch(conversation_id)

    current_title = convs.get_title(conversation_id)
    if p["conv_was_created"] or current_title in ("", "New chat"):
        try:
            title_data = llm.complete_with_schema(
                system="Summarize a student tutoring question into a SHORT title (3-6 words). Output only the title field.",
                user=f"STUDENT QUESTION: {req.question}",
                schema=TITLE_SCHEMA, tool_name="submit_title",
                tool_description="Submit a short conversation title.", max_tokens=128,
            )
            title = (title_data.get("title") or "").strip()
            if title:
                convs.set_title(conversation_id, title[:80])
        except Exception:
            pass

    for it in selected_items:
        if it.artifact_type is not None and it.artifact_type.value == "concept":
            current = next((m for m in student.mastery_for(req.student_id) if m.concept_id == it.id), None)
            if current:
                new_score, new_conf = min(1.0, current.score + 0.02), min(1.0, current.confidence + 0.05)
            else:
                new_score, new_conf = 0.3, 0.1
            student.update_mastery(req.student_id, it.id, new_score, new_conf)

    conn.commit()

    id_to_title: dict[str, str] = {}
    for it in [*candidates, *selected_items, *decision.selected]:
        id_to_title.setdefault(it.id, it.title)
        id_to_title.setdefault(it.id[:8], it.title)
    new_text, refs = _substitute_citations(reply_text, id_to_title)

    def _item(it, scores):
        sc = scores.get(it.id)
        return ChatItem(
            id=it.id, title=it.title, body=it.body, token_estimate=it.token_estimate,
            score_total=sc.total if sc else 0.0, score_relevance=sc.relevance if sc else 0.0,
            score_recency=sc.recency if sc else 0.0, score_misconception=sc.misconception if sc else 0.0,
            score_prerequisite=sc.prerequisite if sc else 0.0, score_reuse=sc.reuse if sc else 0.0,
        )

    if preselected is not None:
        sel_ids = {it.id for it in selected_items}
        dropped_items = [it for it in candidates if it.id not in sel_ids][:8]
        resp_selected = [_item(it, {}) for it in selected_items]
        resp_dropped = [_item(it, {}) for it in dropped_items]
        resp_budget, resp_tokens = req.budget, tokens_used
    else:
        resp_selected = [_item(it, decision.scores) for it in decision.selected]
        resp_dropped = [_item(it, decision.scores) for it in decision.dropped[:8]]
        resp_budget, resp_tokens = decision.budget, decision.tokens_used

    return {
        "conversation_id": conversation_id, "reply": new_text, "references": refs,
        "selected": resp_selected, "dropped": resp_dropped,
        "budget": resp_budget, "tokens_used": resp_tokens, "router": p["router_used"],
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    req.student_id = request.state.username   # never trust a client-supplied identity
    llm, embedder = _llm_and_embedder(_settings().tutor_model)
    engine = RoutingEngine()
    s = _settings()
    logger = InteractionLogger(path=s.log_dir / "interactions.jsonl")
    conn = connect(s.database_url)
    try:
        p = _prepare_turn(conn, req, llm, embedder, engine, logger)
        text = llm.complete(system=p["system"], user=p["user_prompt"], max_tokens=1024)
        return ChatResponse(**_finalize_turn(conn, req, llm, logger, p, text))
    finally:
        conn.close()


_REASONING_INSTRUCTION = (
    "\n\nBefore answering, show your reasoning. Output EXACTLY in this format:\n"
    "---THINKING---\n"
    "<concise step-by-step reasoning: what the question is really asking, which "
    "context items are relevant, and how you'll structure the answer>\n"
    "---ANSWER---\n"
    "<the student-facing answer, following ALL the style rules above>\n"
    "The student only sees the answer; keep the reasoning brief (a few sentences)."
)


def _split_thinking(deltas):
    """Split a raw text stream into ('thinking'|'text', chunk) around the
    ---THINKING--- / ---ANSWER--- markers. Falls back to all-text if the model
    doesn't emit the markers, so the answer is never lost."""
    THINK, ANS = "---THINKING---", "---ANSWER---"
    phase, buf = "pre", ""
    for delta in deltas:
        buf += delta
        moved = True
        while moved:
            moved = False
            if phase == "pre":
                i = buf.find(THINK)
                if i != -1:
                    buf = buf[i + len(THINK):]
                    phase = "think"
                    moved = True
                else:
                    stripped = buf.lstrip()
                    if stripped and not THINK.startswith(stripped[:len(THINK)]):
                        phase = "answer"   # model skipped the marker
                        moved = True
            elif phase == "think":
                j = buf.find(ANS)
                if j != -1:
                    if j > 0:
                        yield ("thinking", buf[:j])
                    buf = buf[j + len(ANS):]
                    phase = "answer"
                    moved = True
                else:
                    hold = len(ANS) - 1
                    if len(buf) > hold:
                        yield ("thinking", buf[:-hold])
                        buf = buf[-hold:]
            elif phase == "answer":
                if buf:
                    yield ("text", buf)
                    buf = ""
    if buf:
        yield ("thinking" if phase == "think" else "text", buf)


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest, request: Request):
    """Same as /api/chat but streams the reply token-by-token via SSE, then sends
    a final `{"done": true, ...}` event with references/selection/conversation_id."""
    req.student_id = request.state.username
    llm, embedder = _llm_and_embedder(_settings().tutor_model)
    engine = RoutingEngine()
    s = _settings()
    logger = InteractionLogger(path=s.log_dir / "interactions.jsonl")

    conn = connect(s.database_url)
    try:
        p = _prepare_turn(conn, req, llm, embedder, engine, logger)
    except Exception:
        conn.close()
        raise

    def event_stream():
        try:
            chunks: list[str] = []
            raw = llm.stream(
                system=p["system"] + _REASONING_INSTRUCTION,
                user=p["user_prompt"], max_tokens=2048,
            )
            for kind, piece in _split_thinking(raw):
                if kind == "thinking":
                    yield f"data: {json.dumps({'thinking': piece})}\n\n"
                else:
                    chunks.append(piece)
                    yield f"data: {json.dumps({'delta': piece})}\n\n"
            fin = _finalize_turn(conn, req, llm, logger, p, "".join(chunks))
            done = ChatResponse(**fin).model_dump()
            done["done"] = True
            yield f"data: {json.dumps(done)}\n\n"
        except Exception as e:  # noqa: BLE001
            logger.log({"event": "chat_stream_failed", "student_id": req.student_id, "error": str(e)})
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            conn.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class QuizGenRequest(BaseModel):
    topic_id: Optional[str] = None      # optional: inferred from `question` if absent
    student_id: Optional[str] = None
    question: Optional[str] = None       # the chat question the student wants tested on


class QuizGenResponse(BaseModel):
    question: str
    rubric: str
    difficulty: str = "Easy"
    topic_id: str = ""                    # the resolved topic (so the client can score)
    topic_title: str = ""


# Adaptive difficulty bands. `lo` is the inclusive lower bound on the blended
# proficiency score (0..1). Questions get harder as the student improves.
_QUIZ_BANDS = [
    (0.0, "Easy",
     "Make this an EASY warm-up: test recall and basic intuition of ONE core "
     "concept. Answerable in 2-3 sentences. No math derivations, multi-part "
     "questions, or obscure edge cases."),
    (0.4, "Medium",
     "Make this MODERATE: ask the student to apply one concept to a concrete, "
     "realistic scenario. Light reasoning, answerable in 3-5 sentences."),
    (0.65, "Hard",
     "Make this CHALLENGING: require comparing two approaches or analyzing a "
     "tradeoff / failure mode. Answerable in a short paragraph."),
    (0.85, "Expert",
     "Make this EXPERT-level: probe edge cases, quantitative reasoning, or "
     "synthesis across multiple concepts. Assume strong mastery."),
]


def _quiz_difficulty(mastery_avg, recent_scores) -> tuple[str, str, float]:
    """Blend topic mastery with recent quiz performance into a difficulty band.
    New/unknown topics start gentle so students aren't thrown in the deep end."""
    base = 0.2 if mastery_avg is None else float(mastery_avg)
    if recent_scores:
        ra = sum(recent_scores) / len(recent_scores)
        base = 0.6 * base + 0.4 * ra
    label, guidance = _QUIZ_BANDS[0][1], _QUIZ_BANDS[0][2]
    for lo, lbl, g in _QUIZ_BANDS:
        if base >= lo:
            label, guidance = lbl, g
    return label, guidance, base


@app.post("/api/quiz/generate", response_model=QuizGenResponse)
def quiz_generate(req: QuizGenRequest, request: Request):
    req.student_id = request.state.username
    llm, embedder = _llm_and_embedder()
    title_by_id = {t.id: t.title for t in _TOPICS}
    conn = connect(_settings().database_url)
    try:
        topic_id = req.topic_id
        # No topic chosen? Infer it from the question the student wants tested on —
        # they shouldn't have to pick a topic to quiz themselves on what they just asked.
        if not topic_id and req.question:
            try:
                qvec = embedder.embed_one(req.question)
                hits = SemanticStore(conn).vector_search(query=qvec, k=5)
                topic_id = next((h.topic_id for h in hits if h.topic_id), None)
            except Exception:
                topic_id = None
        if not topic_id:
            raise HTTPException(status_code=400, detail="Could not determine a topic — pick one or ask a more specific question.")

        with conn.cursor() as cur:
            cur.execute(
                "SELECT body FROM semantic_items "
                "WHERE topic_id = %s AND artifact_type IN ('concept','example','paper_claim') "
                "ORDER BY random() LIMIT 6",
                (topic_id,),
            )
            excerpt = "\n\n".join(r["body"] for r in cur.fetchall())

            mastery_avg = None
            recent_scores: list[float] = []
            if req.student_id:
                cur.execute(
                    "SELECT avg(m.score) AS a FROM mastery m "
                    "JOIN semantic_items s ON s.id = m.concept_id "
                    "WHERE m.student_id = %s AND s.topic_id = %s",
                    (req.student_id, topic_id),
                )
                row = cur.fetchone()
                mastery_avg = row["a"] if row and row["a"] is not None else None

                cur.execute(
                    "SELECT (payload->>'score')::float AS sc FROM episodic_events "
                    "WHERE student_id = %s AND event_type = 'quiz_attempt' "
                    "  AND payload->>'topic_id' = %s AND payload ? 'score' "
                    "ORDER BY occurred_at DESC LIMIT 3",
                    (req.student_id, topic_id),
                )
                recent_scores = [r["sc"] for r in cur.fetchall() if r["sc"] is not None]

        # What the student already knows / gets wrong on this topic — so the quiz
        # targets their actual gaps instead of asking something generic.
        misc_lines = []
        if req.student_id:
            for m in StudentStore(conn).active_misconceptions(req.student_id):
                if m.get("topic_id") == topic_id and m.get("description"):
                    misc_lines.append(m["description"])
    finally:
        conn.close()
    if not excerpt:
        raise HTTPException(status_code=400, detail=f"No material for topic '{topic_id}'")

    label, guidance, _eff = _quiz_difficulty(mastery_avg, recent_scores)
    topic_title = title_by_id.get(topic_id, topic_id)
    know = ("the student is new to this topic" if mastery_avg is None
            else f"the student's current mastery here is ~{round(float(mastery_avg) * 100)}%")
    misc_block = ("\n\nThe student has shown these MISCONCEPTIONS on this topic — "
                  "prefer a question that probes or corrects one of them:\n- "
                  + "\n- ".join(misc_lines[:3])) if misc_lines else ""
    relevance = (f"\n\nThe student just asked: \"{req.question.strip()[:300]}\". Make the quiz "
                 "question directly relevant to what they were exploring." if req.question else "")
    data = llm.complete_with_schema(
        system=(
            "Generate ONE quiz question on the given ML systems engineering topic, "
            "calibrated to the student's current level. Keep it focused and clearly "
            "worded; prefer testing understanding over trickiness. The rubric should "
            "list the key points a correct answer must cover.\n\n"
            f"DIFFICULTY — {label}: {guidance}\n\n"
            f"Personalize: {know}.{misc_block}{relevance}"
        ),
        user=f"TOPIC: {topic_title}\n\nMATERIAL EXCERPT:\n{excerpt[:3000]}",
        schema=QUIZ_QUESTION_SCHEMA,
        tool_name="submit_quiz_question",
        tool_description="Submit the generated quiz question and rubric.",
    )
    return QuizGenResponse(question=data["question"], rubric=data["rubric"], difficulty=label,
                           topic_id=topic_id, topic_title=topic_title)


class QuizScoreRequest(BaseModel):
    student_id: str
    topic_id: str
    question: str
    rubric: str
    answer: str


class QuizScoreResponse(BaseModel):
    score: float
    rationale: str


@app.post("/api/quiz/score", response_model=QuizScoreResponse)
def quiz_score(req: QuizScoreRequest, request: Request):
    req.student_id = request.state.username
    llm, _ = _llm_and_embedder()
    q = QuizQuestion(question=req.question, rubric=req.rubric)
    result = score_answer(question=q, student_answer=req.answer, judge_llm=llm)
    # Log as episodic event AND update mastery
    conn = connect(_settings().database_url)
    try:
        episodic = EpisodicStore(conn)
        episodic.append(
            student_id=req.student_id, event_type="quiz_attempt",
            payload={"topic_id": req.topic_id, "question": req.question,
                     "answer": req.answer, "score": result.score,
                     "rationale": result.rationale},
        )

        # Bump mastery: exponential moving average toward the quiz score
        student = StudentStore(conn)
        student.ensure_student(req.student_id)

        with conn.cursor() as cur:
            cur.execute(
                "SELECT id::text FROM semantic_items WHERE topic_id = %s AND artifact_type = 'concept' LIMIT 5",
                (req.topic_id,),
            )
            concept_ids = [r["id"] for r in cur.fetchall()]

        # Pass the raw quiz score as evidence; update_mastery does the
        # confidence-weighted blend with the prior (no manual EMA here).
        for cid in concept_ids:
            student.update_mastery(req.student_id, cid, result.score, 0.3)

        # Quiz score is the outcome reward for the turn that taught this material.
        TraceStore(conn).attach_reward(req.student_id, result.score)

        conn.commit()
    finally:
        conn.close()
    return QuizScoreResponse(score=result.score, rationale=result.rationale)


# ── Design-interview AI Judge (AI Frontiers Lab) ───────────────────────────────
class InterviewQuestionRequest(BaseModel):
    student_id: str
    topic_id: Optional[str] = None
    level: str = "intermediate"
    goal: Optional[str] = None


class InterviewEvalRequest(BaseModel):
    student_id: str
    topic_id: Optional[str] = None
    level: str = "intermediate"
    question: str = ""
    answer: str = ""
    # Multi-turn: ordered [{"q","a"}, ...]. When present, the judge grades the whole
    # conversation and we persist the formatted transcript instead of a single Q/A.
    transcript: Optional[list[dict]] = None


class InterviewFollowupRequest(BaseModel):
    student_id: str
    topic_id: Optional[str] = None
    level: str = "intermediate"
    transcript: list[dict]  # [{"q","a"}, ...] so far


@app.post("/api/interview/question")
def interview_question(req: InterviewQuestionRequest):
    """Generate an open-ended ML-systems design question, targeting the student's
    weakest studied topic by default."""
    from learning_memory_os.agents.interview import InterviewAgent
    llm, _ = _llm_and_embedder()
    title_by_id = {t.id: t.title for t in _TOPICS}
    topic_id = req.topic_id
    known: Optional[str] = None
    conn = connect(_settings().database_url)
    try:
        with conn.cursor() as cur:
            if not topic_id:
                cur.execute(
                    "SELECT s.topic_id AS t, avg(m.score) AS a FROM mastery m "
                    "JOIN semantic_items s ON s.id = m.concept_id "
                    "WHERE m.student_id = %s GROUP BY s.topic_id ORDER BY a ASC LIMIT 1",
                    (req.student_id,),
                )
                row = cur.fetchone()
                topic_id = row["t"] if row else (_TOPICS[0].id if _TOPICS else None)
            # what the student already knows on this topic (mastery + misconceptions)
            mastery_avg = None
            if topic_id:
                cur.execute(
                    "SELECT avg(m.score) AS a FROM mastery m JOIN semantic_items s ON s.id = m.concept_id "
                    "WHERE m.student_id = %s AND s.topic_id = %s",
                    (req.student_id, topic_id),
                )
                row = cur.fetchone()
                mastery_avg = row["a"] if row and row["a"] is not None else None
        misc = [m["description"] for m in StudentStore(conn).active_misconceptions(req.student_id)
                if m.get("topic_id") == topic_id and m.get("description")][:3]
    finally:
        conn.close()
    if mastery_avg is not None or misc:
        parts = []
        if mastery_avg is not None:
            parts.append(f"current mastery ~{round(float(mastery_avg) * 100)}%")
        if misc:
            parts.append("known misconceptions: " + "; ".join(misc))
        known = "; ".join(parts)
    topic_title = title_by_id.get(topic_id, topic_id or "ML systems")
    q = InterviewAgent(llm).generate_question(topic_title=topic_title, level=req.level, goal=req.goal, known=known)
    return {"topic_id": topic_id, "topic_title": topic_title, "level": req.level, "question": q}


@app.post("/api/interview/followup")
def interview_followup(req: InterviewFollowupRequest):
    """Mid-interview probe: given the transcript so far, ask the next follow-up that
    drills into the weakest part of the candidate's latest answer."""
    from learning_memory_os.agents.interview import InterviewAgent
    llm, _ = _llm_and_embedder()
    title_by_id = {t.id: t.title for t in _TOPICS}
    topic_title = title_by_id.get(req.topic_id, req.topic_id or "ML systems")
    q = InterviewAgent(llm).followup(
        topic_title=topic_title, level=req.level, transcript=req.transcript,
    )
    return {"topic_id": req.topic_id, "topic_title": topic_title, "question": q}


@app.post("/api/interview/evaluate")
def interview_evaluate(req: InterviewEvalRequest):
    """Judge a design answer (structured rubric), persist it, update the skill
    model from the score, record misconceptions, and store a tutor reflection."""
    from learning_memory_os.agents.interview import InterviewAgent
    llm, _ = _llm_and_embedder()
    title_by_id = {t.id: t.title for t in _TOPICS}
    topic_title = title_by_id.get(req.topic_id, req.topic_id or "ML systems")
    conn = connect(_settings().database_url)
    try:
        student = StudentStore(conn)
        student.ensure_student(req.student_id)
        style = build_profile(conn, req.student_id).learning_style or None
        try:
            ev = InterviewAgent(llm).evaluate(
                question=req.question, answer=req.answer, topic_title=topic_title,
                level=req.level, profile_summary=style, transcript=req.transcript,
            )
        except ValueError:
            raise HTTPException(status_code=502, detail="The AI judge did not return an evaluation. Please try again.")
        overall = int(ev.get("overall_score") or 0)
        # Persist the conversation: for a multi-turn interview store the flattened
        # transcript so the readiness verdict + history reflect the full exchange.
        if req.transcript:
            ev["turns"] = len(req.transcript)
            store_q = "\n\n".join((t.get("q") or "") for t in req.transcript)
            store_a = "\n\n".join((t.get("a") or "") for t in req.transcript)
        else:
            store_q, store_a = req.question, req.answer

        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO interview_evaluations "
                "(student_id, topic_id, level, question, answer, overall_score, evaluation) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)",
                (req.student_id, req.topic_id, req.level, store_q, store_a,
                 overall, json.dumps(ev)),
            )
            concept_ids = []
            if req.topic_id:
                cur.execute(
                    "SELECT id::text FROM semantic_items WHERE topic_id = %s "
                    "AND artifact_type = 'concept' LIMIT 6",
                    (req.topic_id,),
                )
                concept_ids = [r["id"] for r in cur.fetchall()]
        # Skill-model update: blend the interview score into topic mastery.
        for cid in concept_ids:
            student.update_mastery(req.student_id, cid, overall / 100.0, 0.4)
        # Record detected misconceptions as durable, evidence-tagged notes.
        for m in (ev.get("misconceptions") or [])[:5]:
            desc = (m.get("description") or "").strip()
            if desc:
                student.record_misconception(
                    req.student_id, concept_id=None, description=desc[:300],
                    evidence=(m.get("concept") or None), topic_id=req.topic_id,
                )
        EpisodicStore(conn).append(
            student_id=req.student_id, event_type="interview_attempt",
            payload={"topic_id": req.topic_id, "overall_score": overall},
        )
        # Self-improvement: persist a structured tutor reflection.
        weak = sorted((ev.get("category_scores") or {}).items(), key=lambda kv: kv[1])[:3]
        reflection = (
            f"Interview on {topic_title}: scored {overall}/100. "
            f"Weakest: {', '.join(k for k, _ in weak)}. "
            f"Next: {ev.get('recommended_exercise_type', 'mock_interview')} on {ev.get('next_topic', '')}."
        )
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tutor_reflections (student_id, summary, payload) VALUES (%s, %s, %s::jsonb)",
                (req.student_id, reflection,
                 json.dumps({"topic_id": req.topic_id, "overall": overall, "next_topic": ev.get("next_topic")})),
            )
        conn.commit()
    finally:
        conn.close()
    return ev


# ── Production Debugging Mode ──────────────────────────────────────────────────
class DebugIncidentRequest(BaseModel):
    student_id: str
    topic_id: Optional[str] = None
    level: str = "intermediate"


class DebugEvalRequest(BaseModel):
    student_id: str
    topic_id: Optional[str] = None
    level: str = "intermediate"
    incident: str
    diagnosis: str


def _weakest_topic(student_id: str) -> Optional[str]:
    conn = connect(_settings().database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT s.topic_id AS t FROM mastery m JOIN semantic_items s ON s.id = m.concept_id "
                "WHERE m.student_id = %s GROUP BY s.topic_id ORDER BY avg(m.score) ASC LIMIT 1",
                (student_id,),
            )
            row = cur.fetchone()
            return row["t"] if row else (_TOPICS[0].id if _TOPICS else None)
    finally:
        conn.close()


@app.post("/api/debug/incident")
def debug_incident(req: DebugIncidentRequest):
    """Generate a realistic ML-systems production incident (with simulated logs/metrics)."""
    from learning_memory_os.agents.interview import DebuggingAgent
    llm, _ = _llm_and_embedder()
    title_by_id = {t.id: t.title for t in _TOPICS}
    topic_id = req.topic_id or _weakest_topic(req.student_id)
    topic_title = title_by_id.get(topic_id, topic_id or "ML systems")
    inc = DebuggingAgent(llm).generate_incident(topic_title=topic_title, level=req.level)
    return {"topic_id": topic_id, "topic_title": topic_title, "level": req.level, "incident": inc}


@app.post("/api/debug/evaluate")
def debug_evaluate(req: DebugEvalRequest):
    """Grade the student's debugging process; persist + update skill model + reflect."""
    from learning_memory_os.agents.interview import DebuggingAgent
    llm, _ = _llm_and_embedder()
    title_by_id = {t.id: t.title for t in _TOPICS}
    topic_title = title_by_id.get(req.topic_id, req.topic_id or "ML systems")
    conn = connect(_settings().database_url)
    try:
        student = StudentStore(conn)
        student.ensure_student(req.student_id)
        style = build_profile(conn, req.student_id).learning_style or None
        try:
            ev = DebuggingAgent(llm).evaluate(
                incident=req.incident, diagnosis=req.diagnosis, topic_title=topic_title,
                level=req.level, profile_summary=style,
            )
        except ValueError:
            raise HTTPException(status_code=502, detail="The AI judge did not return an evaluation. Please try again.")
        ev["kind"] = "debug"
        overall = int(ev.get("overall_score") or 0)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO interview_evaluations "
                "(student_id, topic_id, level, question, answer, overall_score, evaluation) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)",
                (req.student_id, req.topic_id, req.level, req.incident, req.diagnosis,
                 overall, json.dumps(ev)),
            )
            concept_ids = []
            if req.topic_id:
                cur.execute(
                    "SELECT id::text FROM semantic_items WHERE topic_id = %s "
                    "AND artifact_type = 'concept' LIMIT 6",
                    (req.topic_id,),
                )
                concept_ids = [r["id"] for r in cur.fetchall()]
        for cid in concept_ids:
            student.update_mastery(req.student_id, cid, overall / 100.0, 0.4)
        for m in (ev.get("misconceptions") or [])[:5]:
            desc = (m.get("description") or "").strip()
            if desc:
                student.record_misconception(
                    req.student_id, concept_id=None, description=desc[:300],
                    evidence=(m.get("concept") or None), topic_id=req.topic_id,
                )
        EpisodicStore(conn).append(
            student_id=req.student_id, event_type="debug_attempt",
            payload={"topic_id": req.topic_id, "overall_score": overall},
        )
        weak = sorted((ev.get("category_scores") or {}).items(), key=lambda kv: kv[1])[:3]
        reflection = (
            f"Debugging incident on {topic_title}: scored {overall}/100. "
            f"Weakest: {', '.join(k for k, _ in weak)}. Next: {ev.get('next_topic', '')}."
        )
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tutor_reflections (student_id, summary, payload) VALUES (%s, %s, %s::jsonb)",
                (req.student_id, reflection, json.dumps({"kind": "debug", "topic_id": req.topic_id, "overall": overall})),
            )
        conn.commit()
    finally:
        conn.close()
    return ev


# ── Forward-deployed engineer mode ─────────────────────────────────────────────
class ForwardScenarioRequest(BaseModel):
    student_id: str
    topic_id: Optional[str] = None
    level: str = "intermediate"


class ForwardEvalRequest(BaseModel):
    student_id: str
    topic_id: Optional[str] = None
    level: str = "intermediate"
    scenario: str
    response: str


@app.post("/api/forward/scenario")
def forward_scenario(req: ForwardScenarioRequest):
    """Generate a vague customer scenario ('our AI agent feels slow') for the
    forward-deployed engineer exercise."""
    from learning_memory_os.agents.interview import ForwardDeployedAgent
    llm, _ = _llm_and_embedder()
    title_by_id = {t.id: t.title for t in _TOPICS}
    topic_id = req.topic_id or _weakest_topic(req.student_id)
    topic_title = title_by_id.get(topic_id, topic_id or "ML systems")
    sc = ForwardDeployedAgent(llm).generate_scenario(topic_title=topic_title, level=req.level)
    return {"topic_id": topic_id, "topic_title": topic_title, "level": req.level, "scenario": sc}


@app.post("/api/forward/evaluate")
def forward_evaluate(req: ForwardEvalRequest):
    """Grade how the candidate handled the customer problem across the 7
    forward-deployed sub-skills; persist + update skill model + reflect."""
    from learning_memory_os.agents.interview import ForwardDeployedAgent
    llm, _ = _llm_and_embedder()
    title_by_id = {t.id: t.title for t in _TOPICS}
    topic_title = title_by_id.get(req.topic_id, req.topic_id or "ML systems")
    conn = connect(_settings().database_url)
    try:
        student = StudentStore(conn)
        student.ensure_student(req.student_id)
        style = build_profile(conn, req.student_id).learning_style or None
        try:
            ev = ForwardDeployedAgent(llm).evaluate(
                scenario=req.scenario, response=req.response, topic_title=topic_title,
                level=req.level, profile_summary=style,
            )
        except ValueError:
            raise HTTPException(status_code=502, detail="The AI judge did not return an evaluation. Please try again.")
        ev["kind"] = "forward"
        overall = int(ev.get("overall_score") or 0)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO interview_evaluations "
                "(student_id, topic_id, level, question, answer, overall_score, evaluation) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)",
                (req.student_id, req.topic_id, req.level, req.scenario, req.response,
                 overall, json.dumps(ev)),
            )
            concept_ids = []
            if req.topic_id:
                cur.execute(
                    "SELECT id::text FROM semantic_items WHERE topic_id = %s "
                    "AND artifact_type = 'concept' LIMIT 6",
                    (req.topic_id,),
                )
                concept_ids = [r["id"] for r in cur.fetchall()]
        for cid in concept_ids:
            student.update_mastery(req.student_id, cid, overall / 100.0, 0.4)
        for m in (ev.get("misconceptions") or [])[:5]:
            desc = (m.get("description") or "").strip()
            if desc:
                student.record_misconception(
                    req.student_id, concept_id=None, description=desc[:300],
                    evidence=(m.get("concept") or None), topic_id=req.topic_id,
                )
        EpisodicStore(conn).append(
            student_id=req.student_id, event_type="forward_attempt",
            payload={"topic_id": req.topic_id, "overall_score": overall},
        )
        weak = sorted((ev.get("category_scores") or {}).items(), key=lambda kv: kv[1])[:3]
        reflection = (
            f"Forward-deployed scenario on {topic_title}: scored {overall}/100. "
            f"Weakest: {', '.join(k for k, _ in weak)}. Next: {ev.get('next_topic', '')}."
        )
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tutor_reflections (student_id, summary, payload) VALUES (%s, %s, %s::jsonb)",
                (req.student_id, reflection, json.dumps({"kind": "forward", "topic_id": req.topic_id, "overall": overall})),
            )
        conn.commit()
    finally:
        conn.close()
    return ev


class DiagnosticStartRequest(BaseModel):
    original_question: str
    rubric: str
    student_answer: str
    score: float


class DiagnosticStartResponse(BaseModel):
    diagnosis: str
    follow_up_question: str


@app.post("/api/diagnostic/start", response_model=DiagnosticStartResponse)
def diagnostic_start(req: DiagnosticStartRequest):
    llm, _ = _llm_and_embedder()
    data = llm.complete_with_schema(
        system=("You are a tutor diagnosing a student's misunderstanding from a wrong quiz answer. "
                "Return a one-sentence diagnosis (your guess at what they misunderstand) and a probing "
                "follow-up question that would help confirm or refute the diagnosis."),
        user=(f"ORIGINAL QUESTION: {req.original_question}\n\n"
              f"RUBRIC: {req.rubric}\n\n"
              f"STUDENT'S ANSWER (scored {req.score:.2f}):\n{req.student_answer}"),
        schema=DIAGNOSTIC_SCHEMA,
        tool_name="submit_diagnosis",
        tool_description="Submit the diagnosis and follow-up question.",
    )
    return DiagnosticStartResponse(
        diagnosis=data["diagnosis"],
        follow_up_question=data["follow_up_question"],
    )


class DiagnosticTurnRequest(BaseModel):
    student_id: str
    original_question: str
    diagnosis: str
    follow_up_question: str
    student_answer: str
    turn_index: int = 1
    topic_id: str | None = None


class DiagnosticTurnResponse(BaseModel):
    confirmed_misconception: str
    explanation: str
    next_action: str
    next_message: str


@app.post("/api/diagnostic/turn", response_model=DiagnosticTurnResponse)
def diagnostic_turn(req: DiagnosticTurnRequest, request: Request):
    req.student_id = request.state.username
    llm, _ = _llm_and_embedder()
    data = llm.complete_with_schema(
        system=("You are a tutor walking a student through a misunderstanding. Given the student's "
                "answer to your follow-up question, either confirm the misconception and explain "
                "the correct mental model, request a re-test on the concept, or wrap up if they "
                "now understand. Set next_action accordingly."),
        user=(f"ORIGINAL QUESTION: {req.original_question}\n\n"
              f"YOUR DIAGNOSIS: {req.diagnosis}\n\n"
              f"YOUR FOLLOW-UP: {req.follow_up_question}\n\n"
              f"STUDENT'S ANSWER (turn {req.turn_index}):\n{req.student_answer}"),
        schema=EXPLAIN_SCHEMA,
        tool_name="submit_explanation",
        tool_description="Submit the next step of the diagnostic conversation.",
    )

    # If wrapping up, record the misconception
    if data["next_action"] == "wrap_up":
        conn = connect(_settings().database_url)
        try:
            student = StudentStore(conn)
            student.ensure_student(req.student_id)
            student.record_misconception(
                req.student_id,
                concept_id=None,
                description=data["confirmed_misconception"][:500],
                evidence=req.original_question[:500],
                topic_id=req.topic_id,
            )
            conn.commit()
        finally:
            conn.close()

    return DiagnosticTurnResponse(**data)


class FeedbackRequest(BaseModel):
    student_id: str
    message_idx: int            # client-assigned ordinal of the assistant message
    rating: int                 # +1 (helpful), -1 (not helpful)
    selected_item_ids: list[str] = []   # which items contributed to this reply


@app.post("/api/feedback")
def feedback(req: FeedbackRequest, request: Request):
    req.student_id = request.state.username
    conn = connect(_settings().database_url)
    try:
        student = StudentStore(conn)
        student.ensure_student(req.student_id)
        episodic = EpisodicStore(conn)
        episodic.append(
            student_id=req.student_id,
            event_type="feedback",
            payload={
                "message_idx": req.message_idx,
                "rating": req.rating,
                "selected_item_ids": req.selected_item_ids,
            },
        )
        # Thumbs feedback is an outcome reward (+1 / -1) on the captured turn.
        TraceStore(conn).attach_reward(req.student_id, float(req.rating))
        conn.commit()
    finally:
        conn.close()
    return {"ok": True}


@app.get("/api/student/{student_id}/progress")
def student_progress(student_id: str):
    conn = connect(_settings().database_url)
    try:
        student = StudentStore(conn)
        student.ensure_student(student_id)
        mastery = student.mastery_for(student_id)

        # Join with concept titles and topic_ids
        ids = [m.concept_id for m in mastery]
        id_to_meta: dict[str, dict] = {}
        if ids:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id::text AS id, title, topic_id FROM semantic_items WHERE id::text = ANY(%s)",
                    (ids,),
                )
                for r in cur.fetchall():
                    id_to_meta[r["id"]] = {"title": r["title"], "topic_id": r["topic_id"]}

        # Group by topic
        by_topic: dict[str, list] = {}
        for m in mastery:
            meta = id_to_meta.get(m.concept_id, {"title": m.concept_id, "topic_id": "?"})
            # Decay-on-read: knowledge fades since it was last reinforced.
            eff = effective_score(m.score, m.confidence, m.last_updated)
            by_topic.setdefault(meta["topic_id"], []).append({
                "concept_title": meta["title"],
                "score": round(eff, 3),
                "confidence": m.confidence,
            })

        # Confidence-weighted average mastery per topic
        topic_summaries = []
        for topic_id, items in by_topic.items():
            total_conf = sum(i["confidence"] for i in items) or 0.001
            avg = sum(i["score"] * i["confidence"] for i in items) / total_conf
            topic_summaries.append({
                "topic_id": topic_id,
                "concepts": items[:5],
                "avg_mastery": round(avg, 3),
            })
        topic_summaries.sort(key=lambda x: -x["avg_mastery"])

        misconceptions = [
            {
                "id": m["id"],
                "description": m["description"],
                "detected_at": str(m.get("detected_at", "")),
            }
            for m in student.active_misconceptions(student_id)
        ]
        return {"topics": topic_summaries, "misconceptions": misconceptions}
    finally:
        conn.close()


@app.get("/api/student/{student_id}/review")
def student_review(student_id: str):
    """Concepts due for spaced-repetition review, with title + topic for display."""
    conn = connect(_settings().database_url)
    try:
        student = StudentStore(conn)
        student.ensure_student(student_id)
        due_ids = student.due_for_review(student_id)
        if not due_ids:
            return {"due": []}
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id::text AS id, title, topic_id FROM semantic_items WHERE id::text = ANY(%s)",
                (due_ids,),
            )
            meta = {r["id"]: r for r in cur.fetchall()}
        due = [
            {
                "concept_id": cid,
                "title": meta.get(cid, {}).get("title", cid),
                "topic_id": meta.get(cid, {}).get("topic_id"),
            }
            for cid in due_ids
        ]
        return {"due": due}
    finally:
        conn.close()


@app.get("/api/student/{student_id}/profile")
def student_profile_view(student_id: str):
    """The always-on learner profile: what the tutor has adapted to for this user."""
    conn = connect(_settings().database_url)
    try:
        return build_profile(conn, student_id).to_dict()
    finally:
        conn.close()


@app.get("/api/student/{student_id}/traces/summary")
def traces_summary(student_id: str):
    """How much personalization data has been captured for this user."""
    conn = connect(_settings().database_url)
    try:
        store = TraceStore(conn)
        recent = [
            {
                "task_text": r["task_text"],
                "task_type": r["task_type"],
                "n_selected": r["n_selected"],
                "reward": r["reward"],
                "occurred_at": str(r["occurred_at"]),
            }
            for r in store.recent(student_id, limit=20)
        ]
        return {"count": store.count(student_id), "recent": recent}
    finally:
        conn.close()


@app.get("/api/student/{student_id}/traces/export")
def traces_export(student_id: str, min_reward: Optional[float] = None, format: str = "router"):
    """Export this user's captured turns as fine-tune-ready JSONL.

    format=router → Trajectory selection-training records;
    format=tutor  → rich records keeping the reply + reward.
    """
    conn = connect(_settings().database_url)
    try:
        store = TraceStore(conn)
        if format == "tutor":
            rows = [json.dumps(r, default=str) for r in store.export_records(student_id, min_reward=min_reward)]
        else:
            rows = [json.dumps(t.model_dump(), default=str) for t in store.export_trajectories(student_id, min_reward=min_reward)]
        body = "\n".join(rows)
        return PlainTextResponse(
            body,
            media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{student_id}-trajectories.jsonl"'},
        )
    finally:
        conn.close()


@app.delete("/api/student/{student_id}/traces")
def traces_delete(student_id: str):
    """Consent control: erase all captured personalization data for this user."""
    conn = connect(_settings().database_url)
    try:
        n = TraceStore(conn).delete_for_student(student_id)
        conn.commit()
        return {"deleted": n}
    finally:
        conn.close()


@app.get("/api/student/{student_id}/activity")
def student_activity(student_id: str):
    """Aggregate everything a student has done: lifetime stats, quiz-score
    history, and a recent activity timeline — all from episodic_events + mastery."""
    conn = connect(_settings().database_url)
    try:
        student = StudentStore(conn)
        student.ensure_student(student_id)

        mastery = student.mastery_for(student_id)
        concepts_assessed = len(mastery)
        concepts_mastered = sum(1 for m in mastery if m.score >= 0.7)
        misconceptions = student.active_misconceptions(student_id)

        with conn.cursor() as cur:
            # Event-type counts
            cur.execute(
                "SELECT event_type, count(*) AS n FROM episodic_events "
                "WHERE student_id = %s GROUP BY event_type",
                (student_id,),
            )
            counts = {r["event_type"]: r["n"] for r in cur.fetchall()}

            # Quiz score history (chronological)
            cur.execute(
                "SELECT (payload->>'topic_id') AS topic_id, "
                "       (payload->>'score')::float AS score, occurred_at "
                "FROM episodic_events "
                "WHERE student_id = %s AND event_type = 'quiz_attempt' "
                "  AND payload ? 'score' "
                "ORDER BY occurred_at ASC",
                (student_id,),
            )
            quiz_rows = cur.fetchall()

            # Recent timeline (questions, quizzes, feedback)
            cur.execute(
                "SELECT event_type, payload, occurred_at FROM episodic_events "
                "WHERE student_id = %s AND event_type IN ('question','quiz_attempt','feedback') "
                "ORDER BY occurred_at DESC LIMIT 40",
                (student_id,),
            )
            tl_rows = cur.fetchall()

            # Active span + distinct active days
            cur.execute(
                "SELECT min(occurred_at) AS first, max(occurred_at) AS last, "
                "       count(DISTINCT occurred_at::date) AS days "
                "FROM episodic_events WHERE student_id = %s",
                (student_id,),
            )
            span = cur.fetchone() or {}

            # Topics touched (distinct topic_id over assessed concepts)
            topics_touched = 0
            ids = [m.concept_id for m in mastery]
            if ids:
                cur.execute(
                    "SELECT count(DISTINCT topic_id) AS n FROM semantic_items WHERE id::text = ANY(%s)",
                    (ids,),
                )
                topics_touched = cur.fetchone()["n"]

            # Conversation count
            cur.execute(
                "SELECT count(*) AS n FROM conversations WHERE student_id = %s",
                (student_id,),
            )
            conv_count = cur.fetchone()["n"]

        quiz_scores = [q["score"] for q in quiz_rows if q["score"] is not None]
        avg_quiz = round(sum(quiz_scores) / len(quiz_scores), 3) if quiz_scores else None

        def _label(row):
            p = row["payload"] or {}
            if row["event_type"] == "question":
                return (p.get("text") or p.get("question") or p.get("task") or "Asked a question")[:120]
            if row["event_type"] == "quiz_attempt":
                t = p.get("topic_id") or ""
                return f"Quiz · {t}" if t else "Took a quiz"
            if row["event_type"] == "feedback":
                return "Rated a tutor response"
            return row["event_type"]

        timeline = [
            {
                "type": r["event_type"],
                "label": _label(r),
                "topic_id": (r["payload"] or {}).get("topic_id"),
                "score": (r["payload"] or {}).get("score"),
                "occurred_at": r["occurred_at"].isoformat() if r["occurred_at"] else None,
            }
            for r in tl_rows
        ]

        return {
            "stats": {
                "questions": counts.get("question", 0),
                "quizzes": counts.get("quiz_attempt", 0),
                "feedback": counts.get("feedback", 0),
                "avg_quiz_score": avg_quiz,
                "concepts_assessed": concepts_assessed,
                "concepts_mastered": concepts_mastered,
                "topics_touched": topics_touched,
                "misconceptions": len(misconceptions),
                "conversations": conv_count,
                "active_days": span.get("days", 0) or 0,
                "first_active": span["first"].isoformat() if span.get("first") else None,
                "last_active": span["last"].isoformat() if span.get("last") else None,
            },
            "quiz_history": [
                {
                    "topic_id": q["topic_id"],
                    "score": q["score"],
                    "occurred_at": q["occurred_at"].isoformat() if q["occurred_at"] else None,
                }
                for q in quiz_rows
            ],
            "timeline": timeline,
        }
    finally:
        conn.close()


# ── Conversation endpoints ────────────────────────────────────────────────────

class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    last_message_at: str


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]


@app.get("/api/student/{student_id}/conversations", response_model=ConversationListResponse)
def list_conversations(student_id: str, limit: int = 100):
    conn = connect(_settings().database_url)
    try:
        StudentStore(conn).ensure_student(student_id)
        convs = ConversationStore(conn).list_for_student(student_id, limit=limit)
        return ConversationListResponse(conversations=[
            ConversationSummary(
                id=c["id"],
                title=c["title"] or "New chat",
                created_at=c["created_at"].isoformat() if c["created_at"] else "",
                last_message_at=c["last_message_at"].isoformat() if c["last_message_at"] else "",
            )
            for c in convs
        ])
    finally:
        conn.close()


class ConversationMessagesResponse(BaseModel):
    conversation_id: str
    messages: list[StoredMessage]


@app.get("/api/conversations/{conversation_id}/messages", response_model=ConversationMessagesResponse)
def conversation_messages(conversation_id: str, request: Request):
    conn = connect(_settings().database_url)
    try:
        store = ConversationStore(conn)
        owner = store.owner(conversation_id)
        if owner is not None and owner != request.state.username:
            raise HTTPException(status_code=403, detail="Forbidden")
        rows = store.messages(conversation_id)
        msgs: list[StoredMessage] = []
        for r in rows:
            payload = r["payload"] or {}
            text = payload.get("text", "")
            if not text:
                continue
            role = "user" if r["event_type"] == "question" else "assistant"
            msgs.append(StoredMessage(
                role=role,
                content=text,
                timestamp=r["occurred_at"].isoformat() if r["occurred_at"] else "",
            ))
        return ConversationMessagesResponse(conversation_id=conversation_id, messages=msgs)
    finally:
        conn.close()


class CreateConversationRequest(BaseModel):
    student_id: str
    title: str = "New chat"


class CreateConversationResponse(BaseModel):
    id: str
    title: str


@app.post("/api/conversations", response_model=CreateConversationResponse)
def create_conversation(req: CreateConversationRequest, request: Request):
    req.student_id = request.state.username
    conn = connect(_settings().database_url)
    try:
        StudentStore(conn).ensure_student(req.student_id)
        cid = ConversationStore(conn).create(req.student_id, title=req.title)
        conn.commit()
        return CreateConversationResponse(id=cid, title=req.title)
    finally:
        conn.close()


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request):
    conn = connect(_settings().database_url)
    try:
        store = ConversationStore(conn)
        owner = store.owner(conversation_id)
        if owner is not None and owner != request.state.username:
            raise HTTPException(status_code=403, detail="Forbidden")
        store.delete(conversation_id)
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


# ── Onboarding intake (goal / level / target / learning style) ────────────────
class OnboardingRequest(BaseModel):
    goal: Optional[str] = None
    level: Optional[str] = None
    target: Optional[str] = None
    learning_style: Optional[str] = None


@app.get("/api/student/{student_id}/onboarding")
def get_onboarding(student_id: str):
    conn = connect(_settings().database_url)
    try:
        StudentStore(conn).ensure_student(student_id)
        with conn.cursor() as cur:
            cur.execute("SELECT profile FROM students WHERE id = %s", (student_id,))
            row = cur.fetchone()
        prof = (row["profile"] if row else {}) or {}
        return {"onboarding": prof.get("onboarding") or {}}
    finally:
        conn.close()


@app.post("/api/student/{student_id}/onboarding")
def set_onboarding(student_id: str, req: OnboardingRequest):
    data = {k: v for k, v in {
        "goal": req.goal, "level": req.level, "target": req.target,
        "learning_style": req.learning_style,
    }.items() if v}
    conn = connect(_settings().database_url)
    try:
        StudentStore(conn).ensure_student(student_id)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE students SET profile = COALESCE(profile, '{}'::jsonb) "
                "|| jsonb_build_object('onboarding', %s::jsonb) WHERE id = %s",
                (json.dumps(data), student_id),
            )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "onboarding": data}


# ── Pricing tiers (placeholder; payments via /api/billing) ────────────────────
@app.get("/api/pricing")
def pricing():
    return {"tiers": [
        {"id": "free", "name": "Free", "price": "$0",
         "features": ["Readiness score + 1 gap", "Limited mock interviews", "Concept chat"]},
        {"id": "pro", "name": "Pro", "price": "$29/mo",
         "features": ["Full gap analysis + drill plan", "Unlimited mock interviews & debugging", "Readiness trend over time", "Adaptive curriculum"]},
        {"id": "bootcamp", "name": "Interview Bootcamp", "price": "$199",
         "features": ["Everything in Pro", "Targeted FAANG/infra interview track", "Daily drills + weekly mock loop", "Company-specific prep"]},
        {"id": "premium", "name": "Premium Human + AI", "price": "Contact",
         "features": ["Everything in Bootcamp", "Human staff-engineer review", "1:1 mock interviews", "Offer-negotiation guidance"]},
    ]}


# Mount static frontend last so /api routes take precedence
web_dir = _PROJECT_ROOT / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
