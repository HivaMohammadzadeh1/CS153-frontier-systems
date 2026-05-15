from ..llm import LLM
from ..embeddings import Embedder
from ..logging_utils.interactions import InteractionLogger
from ..schemas.memory import MemoryItem
from ..selector.engine import RoutingEngine
from .base import AgentResponse


TUTOR_SYSTEM = """You are a friendly, focused ML systems engineering tutor.

Style rules (these matter):
1. START with a one-sentence intuitive answer to the student's question.
2. THEN give a short structured explanation: use markdown headings (##) and short bullet lists. Keep each section tight; the student can ask follow-ups for depth.
3. WHEN a diagram would clarify the answer, emit a fenced mermaid block (```mermaid ... ```). Use simple flowchart, sequenceDiagram, or graph TD syntax. Don't force diagrams; only include if they actually help.
4. END with one specific follow-up question: "Want me to go deeper on X or Y?" — invite the next turn.
5. Cite supporting context items by their short id, e.g., [a1b2c3d4]. The UI converts these to numbered references; the student will see [1], [2], [3], not the raw ids.
6. NEVER dump a textbook. If you can't fit the answer in ~250 words, pick the most important angle and offer to expand.
7. If the provided context doesn't fully answer the question, say so plainly and suggest what additional material would help.

You are talking to a student who wants to learn ML systems engineering. Be warm, concrete, and curious-leaning. Avoid jargon dumps; explain terms when you first use them."""


class TutorAgent:
    def __init__(
        self,
        *,
        llm: LLM,
        engine: RoutingEngine,
        embedder: Embedder,
        logger: InteractionLogger,
    ):
        self.llm = llm
        self.engine = engine
        self.embedder = embedder
        self.logger = logger

    def answer(
        self,
        *,
        student_id: str,
        question: str,
        candidates: list[MemoryItem],
        active_misconceptions: set[str],
        prerequisites: set[str],
        recent_ids: set[str],
        reuse_counts: dict[str, int],
        budget: int,
    ) -> AgentResponse:
        task_emb = self.embedder.embed_one(question)
        decision = self.engine.route(
            candidates=candidates,
            task_embedding=task_emb,
            active_misconceptions=active_misconceptions,
            prerequisites=prerequisites,
            recent_ids=recent_ids,
            reuse_counts=reuse_counts,
            budget=budget,
        )
        self.logger.log(
            {
                "event": "routing_decision",
                "agent": "tutor",
                "student_id": student_id,
                "task": question,
                "selected_ids": [it.id for it in decision.selected],
                "dropped_ids": [it.id for it in decision.dropped],
                "tokens_used": decision.tokens_used,
                "budget": decision.budget,
            }
        )

        context_block = "\n\n".join(
            f"[{it.id}] {it.title}\n{it.body}" for it in decision.selected
        )
        user_prompt = f"CONTEXT ITEMS:\n{context_block}\n\nSTUDENT QUESTION:\n{question}"
        text = self.llm.complete(
            system=TUTOR_SYSTEM, user=user_prompt, max_tokens=1024
        )

        self.logger.log(
            {
                "event": "tutor_reply",
                "agent": "tutor",
                "student_id": student_id,
                "text": text,
            }
        )
        return AgentResponse(
            text=text, selected_items=decision.selected, tokens_used=decision.tokens_used
        )
