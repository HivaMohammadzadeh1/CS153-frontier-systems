"""FastAPI REST API for Learning Memory OS.

Wraps the existing Python backend (TutorAgent, RoutingEngine, stores, quiz harness)
in HTTP endpoints. Serves the static frontend from /web.
"""

import re
from pathlib import Path
from typing import Optional

# Project root: two levels up from src/learning_memory_os/api.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
from learning_memory_os.logging_utils.interactions import InteractionLogger
from learning_memory_os.ingestion.topic_loader import load_topics, resolve_prerequisite_titles
from learning_memory_os.eval.quiz import QuizQuestion, score_answer


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


def _llm_and_embedder():
    s = _settings()
    return LLM(api_key=s.anthropic_api_key), Embedder(api_key=s.openai_api_key)


_TOPICS = load_topics(_PROJECT_ROOT / "config" / "topics.yaml")


# ---- Request/response models ----

class ChatRequest(BaseModel):
    student_id: str
    conversation_id: Optional[str] = None
    topic_id: Optional[str] = None
    question: str
    budget: int = 3000
    reuse_counts: dict[str, int] = {}


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


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/topics")
def list_topics():
    return [{"id": t.id, "title": t.title, "area": t.area} for t in _TOPICS]


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


@app.get("/api/info")
def info():
    return {
        "tutor_model": "claude-opus-4-7",
        "embedding_model": "text-embedding-3-small",
    }


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


def _substitute_citations(reply_text: str, decision_selected) -> tuple[str, list[ChatReference]]:
    """Find [a1b2c3d4] short ids in reply_text, replace with [1] [2] [3] in order.
    Return the rewritten text + ordered references list."""
    pattern = re.compile(r"\[([a-f0-9]{8})\]")
    ids_in_order: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(reply_text):
        cid = match.group(1)
        if cid not in seen:
            seen.add(cid)
            ids_in_order.append(cid)
    # Map id -> number
    id_to_n = {cid: i + 1 for i, cid in enumerate(ids_in_order)}
    new_text = pattern.sub(lambda m: f"[{id_to_n[m.group(1)]}]", reply_text)

    # Build references list using titles from selected items
    sel_by_id = {it.id: it for it in decision_selected}
    refs: list[ChatReference] = []
    for cid in ids_in_order:
        title = sel_by_id[cid].title if cid in sel_by_id else cid
        refs.append(ChatReference(n=id_to_n[cid], id=cid, title=title))
    return new_text, refs


TITLE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "A 3-6 word title summarizing the topic."},
    },
    "required": ["title"],
}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    llm, embedder = _llm_and_embedder()
    engine = RoutingEngine()
    s = _settings()
    log_path = s.log_dir / "interactions.jsonl"
    logger = InteractionLogger(path=log_path)

    conn = connect(s.database_url)
    try:
        student = StudentStore(conn)
        student.ensure_student(req.student_id)
        semantic = SemanticStore(conn)
        episodic = EpisodicStore(conn)
        convs = ConversationStore(conn)

        # Resolve or create a conversation
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

        misconceptions = {m["id"] for m in student.active_misconceptions(req.student_id)}
        prereq_titles = (
            resolve_prerequisite_titles(conn, topic_id=req.topic_id, topics=_TOPICS)
            if req.topic_id else set()
        )
        recent = episodic.recent(req.student_id, limit=10)
        recent_ids = {e.id for e in recent if e.id}

        # Build student profile for adaptive prompting
        mastery_records = student.mastery_for(req.student_id)
        weak_ids = [m.concept_id for m in mastery_records if m.score < 0.4 and m.confidence > 0.2]
        strong_ids = [m.concept_id for m in mastery_records if m.score > 0.7 and m.confidence > 0.3]

        weak_titles: list[str] = []
        strong_titles: list[str] = []
        if weak_ids or strong_ids:
            ids_needed = list(set(weak_ids + strong_ids))
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id::text AS id, title FROM semantic_items WHERE id::text = ANY(%s)",
                    (ids_needed,),
                )
                id_to_title = {r["id"]: r["title"] for r in cur.fetchall()}
            weak_titles = [id_to_title[i] for i in weak_ids if i in id_to_title][:5]
            strong_titles = [id_to_title[i] for i in strong_ids if i in id_to_title][:5]

        active_misc_texts = [
            m["description"][:120] for m in student.active_misconceptions(req.student_id)
        ][:3]

        tutor = TutorAgent(llm=llm, engine=engine, embedder=embedder, logger=logger)
        response = tutor.answer(
            student_id=req.student_id,
            question=req.question,
            candidates=candidates,
            active_misconceptions=misconceptions,
            prerequisites=prereq_titles,
            recent_ids=recent_ids,
            reuse_counts=dict(req.reuse_counts),
            budget=req.budget,
            weak_concepts=weak_titles or None,
            strong_concepts=strong_titles or None,
            active_misconception_texts=active_misc_texts or None,
        )

        # Re-run engine to recover decision details
        task_emb = embedder.embed_one(req.question)
        decision = engine.route(
            candidates=candidates,
            task_embedding=task_emb,
            active_misconceptions=misconceptions,
            prerequisites=prereq_titles,
            recent_ids=recent_ids,
            reuse_counts=dict(req.reuse_counts),
            budget=req.budget,
        )

        episodic.append(
            student_id=req.student_id, event_type="question",
            payload={"text": req.question, "topic_id": req.topic_id, "source": "api"},
            conversation_id=conversation_id,
        )
        episodic.append(
            student_id=req.student_id, event_type="tutor_reply",
            payload={"text": response.text,
                     "selected_ids": [it.id for it in response.selected_items],
                     "tokens_used": response.tokens_used},
            conversation_id=conversation_id,
        )

        # Bump last_message_at so this conversation rises to the top
        convs.touch(conversation_id)

        # Auto-title on first user message (if title is still default)
        current_title = convs.get_title(conversation_id)
        if conv_was_created or current_title in ("", "New chat"):
            try:
                title_data = llm.complete_with_schema(
                    system="Summarize a student tutoring question into a SHORT title (3-6 words). Output only the title field.",
                    user=f"STUDENT QUESTION: {req.question}",
                    schema=TITLE_SCHEMA,
                    tool_name="submit_title",
                    tool_description="Submit a short conversation title.",
                    max_tokens=128,
                )
                title = (title_data.get("title") or "").strip()
                if title:
                    convs.set_title(conversation_id, title[:80])
            except Exception:
                pass  # silently ignore title-gen failures

        # Soft mastery bump for engaged concepts
        from learning_memory_os.schemas.artifacts import ArtifactType
        for it in response.selected_items:
            # artifact_type is ArtifactType enum (str subclass); compare .value
            if it.artifact_type is not None and it.artifact_type.value == "concept":
                current = next(
                    (m for m in student.mastery_for(req.student_id) if m.concept_id == it.id),
                    None,
                )
                if current:
                    new_score = min(1.0, current.score + 0.02)
                    new_conf = min(1.0, current.confidence + 0.05)
                else:
                    new_score = 0.3  # low initial mastery on first encounter
                    new_conf = 0.1
                student.update_mastery(req.student_id, it.id, new_score, new_conf)

        conn.commit()

        new_text, refs = _substitute_citations(response.text, decision.selected)

        def _item(it, scores):
            sc = scores.get(it.id)
            return ChatItem(
                id=it.id, title=it.title, body=it.body, token_estimate=it.token_estimate,
                score_total=sc.total if sc else 0.0,
                score_relevance=sc.relevance if sc else 0.0,
                score_recency=sc.recency if sc else 0.0,
                score_misconception=sc.misconception if sc else 0.0,
                score_prerequisite=sc.prerequisite if sc else 0.0,
                score_reuse=sc.reuse if sc else 0.0,
            )

        return ChatResponse(
            conversation_id=conversation_id,
            reply=new_text,
            references=refs,
            selected=[_item(it, decision.scores) for it in decision.selected],
            dropped=[_item(it, decision.scores) for it in decision.dropped[:8]],
            budget=decision.budget,
            tokens_used=decision.tokens_used,
        )
    finally:
        conn.close()


class QuizGenRequest(BaseModel):
    topic_id: str


class QuizGenResponse(BaseModel):
    question: str
    rubric: str


@app.post("/api/quiz/generate", response_model=QuizGenResponse)
def quiz_generate(req: QuizGenRequest):
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
    finally:
        conn.close()
    if not excerpt:
        raise HTTPException(status_code=400, detail=f"No material for topic '{req.topic_id}'")
    data = llm.complete_with_schema(
        system="Generate ONE substantive quiz question on the given ML systems engineering topic.",
        user=f"TOPIC: {req.topic_id}\n\nMATERIAL EXCERPT:\n{excerpt[:3000]}",
        schema=QUIZ_QUESTION_SCHEMA,
        tool_name="submit_quiz_question",
        tool_description="Submit the generated quiz question and rubric.",
    )
    return QuizGenResponse(question=data["question"], rubric=data["rubric"])


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
def quiz_score(req: QuizScoreRequest):
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

        for cid in concept_ids:
            current = next(
                (m for m in student.mastery_for(req.student_id) if m.concept_id == cid),
                None,
            )
            if current:
                # Exponential moving average: 70% old, 30% new
                new_score = 0.7 * current.score + 0.3 * result.score
                new_conf = min(1.0, current.confidence + 0.1)
            else:
                new_score = result.score
                new_conf = 0.3  # low confidence on first observation
            student.update_mastery(req.student_id, cid, new_score, new_conf)

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


class DiagnosticTurnResponse(BaseModel):
    confirmed_misconception: str
    explanation: str
    next_action: str
    next_message: str


@app.post("/api/diagnostic/turn", response_model=DiagnosticTurnResponse)
def diagnostic_turn(req: DiagnosticTurnRequest):
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
def feedback(req: FeedbackRequest):
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
            by_topic.setdefault(meta["topic_id"], []).append({
                "concept_title": meta["title"],
                "score": m.score,
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
def conversation_messages(conversation_id: str):
    conn = connect(_settings().database_url)
    try:
        rows = ConversationStore(conn).messages(conversation_id)
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
def create_conversation(req: CreateConversationRequest):
    conn = connect(_settings().database_url)
    try:
        StudentStore(conn).ensure_student(req.student_id)
        cid = ConversationStore(conn).create(req.student_id, title=req.title)
        conn.commit()
        return CreateConversationResponse(id=cid, title=req.title)
    finally:
        conn.close()


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    conn = connect(_settings().database_url)
    try:
        ConversationStore(conn).delete(conversation_id)
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


# Mount static frontend last so /api routes take precedence
web_dir = _PROJECT_ROOT / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
