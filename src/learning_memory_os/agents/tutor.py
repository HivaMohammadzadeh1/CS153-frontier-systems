from ..llm import LLM
from ..embeddings import Embedder
from ..logging_utils.interactions import InteractionLogger
from ..schemas.memory import MemoryItem
from ..selector.engine import RoutingEngine
from .base import AgentResponse


TUTOR_SYSTEM = """You are a tutor for ML systems engineering students.
Use ONLY the provided context items as evidence. Cite them by [item-id] inline.
Keep answers tight and concrete. If the context does not answer the question, say so
and suggest what additional material would help."""


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
