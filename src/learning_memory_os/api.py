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
_PUBLIC_API = ("/api/health", "/api/topics", "/api/info", "/api/routers", "/api/auth/", "/api/billing/webhook")
_STUDENT_PATH_RE = re.compile(r"^/api/student/([^/]+)")


def _username_from_request(request: Request) -> Optional[str]:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    conn = connect(_settings().database_url)
    try:
        return AuthStore(conn).username_for_session(token)
    finally:
        conn.close()


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    """Require a valid session for all /api/ data routes (except the public list),
    and enforce that /api/student/{id}/... matches the logged-in user."""
    path = request.url.path
    if path.startswith("/api/") and not path.startswith(_PUBLIC_API):
        username = _username_from_request(request)
        if not username:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)
        request.state.username = username
        m = _STUDENT_PATH_RE.match(path)
        if m and unquote(m.group(1)) != username:
            return JSONResponse({"detail": "Forbidden"}, status_code=403)
    return await call_next(request)


def _llm_and_embedder():
    s = _settings()
    return LLM(api_key=s.anthropic_api_key), Embedder(api_key=s.openai_api_key)


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
    username = _username_from_request(request)
    if not username:
        raise HTTPException(status_code=401, detail="Not authenticated")
    conn = connect(_settings().database_url)
    try:
        pro = AuthStore(conn).is_pro(username)
    finally:
        conn.close()
    return {"username": username, "is_pro": pro}


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

        # Pro gating (server-enforced): free users get overall + areas + ONE gap
        # teaser; the full gap analysis and over-time trend are the paid product.
        pro = AuthStore(conn).is_pro(student_id)
        return {
            "overall_readiness": overall,
            "areas": area_readiness,
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

    if req.topic_id:
        candidates = semantic.by_topic(req.topic_id)
    else:
        q_emb = embedder.embed_one(req.question)
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

    task_emb = embedder.embed_one(req.question)
    decision = engine.route(
        candidates=candidates, task_embedding=task_emb,
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
    llm, embedder = _llm_and_embedder()
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
    llm, embedder = _llm_and_embedder()
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
    topic_id: str
    student_id: Optional[str] = None


class QuizGenResponse(BaseModel):
    question: str
    rubric: str
    difficulty: str = "Easy"


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
    llm, _ = _llm_and_embedder()
    conn = connect(_settings().database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT body FROM semantic_items "
                "WHERE topic_id = %s AND artifact_type IN ('concept','example','paper_claim') "
                "ORDER BY random() LIMIT 6",
                (req.topic_id,),
            )
            excerpt = "\n\n".join(r["body"] for r in cur.fetchall())

            mastery_avg = None
            recent_scores: list[float] = []
            if req.student_id:
                cur.execute(
                    "SELECT avg(m.score) AS a FROM mastery m "
                    "JOIN semantic_items s ON s.id = m.concept_id "
                    "WHERE m.student_id = %s AND s.topic_id = %s",
                    (req.student_id, req.topic_id),
                )
                row = cur.fetchone()
                mastery_avg = row["a"] if row and row["a"] is not None else None

                cur.execute(
                    "SELECT (payload->>'score')::float AS sc FROM episodic_events "
                    "WHERE student_id = %s AND event_type = 'quiz_attempt' "
                    "  AND payload->>'topic_id' = %s AND payload ? 'score' "
                    "ORDER BY occurred_at DESC LIMIT 3",
                    (req.student_id, req.topic_id),
                )
                recent_scores = [r["sc"] for r in cur.fetchall() if r["sc"] is not None]
    finally:
        conn.close()
    if not excerpt:
        raise HTTPException(status_code=400, detail=f"No material for topic '{req.topic_id}'")

    label, guidance, _eff = _quiz_difficulty(mastery_avg, recent_scores)
    data = llm.complete_with_schema(
        system=(
            "Generate ONE quiz question on the given ML systems engineering topic, "
            "calibrated to the student's current level. Keep it focused and clearly "
            "worded; prefer testing understanding over trickiness. The rubric should "
            "list the key points a correct answer must cover.\n\n"
            f"DIFFICULTY — {label}: {guidance}"
        ),
        user=f"TOPIC: {req.topic_id}\n\nMATERIAL EXCERPT:\n{excerpt[:3000]}",
        schema=QUIZ_QUESTION_SCHEMA,
        tool_name="submit_quiz_question",
        tool_description="Submit the generated quiz question and rubric.",
    )
    return QuizGenResponse(question=data["question"], rubric=data["rubric"], difficulty=label)


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


# Mount static frontend last so /api routes take precedence
web_dir = _PROJECT_ROOT / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
