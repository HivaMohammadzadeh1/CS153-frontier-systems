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

You are talking to a student who wants to learn ML systems engineering. Be warm, concrete, and curious-leaning. Avoid jargon dumps; explain terms when you first use them.

When a "STUDENT PROFILE" section is provided:
- Calibrate depth: skip basics on topics where mastery is high; spend more time on weak areas.
- If the question touches an active misconception, address it explicitly and gently correct it.
- Don't be condescending. Don't say "as I mentioned before" — the student may not remember."""


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

    @staticmethod
    def build_prompt(
        question: str,
        selected: list[MemoryItem],
        *,
        weak_concepts: list[str] | None = None,
        strong_concepts: list[str] | None = None,
        active_misconception_texts: list[str] | None = None,
        due_concepts: list[str] | None = None,
    ) -> tuple[str, str]:
        """Build (system, user) prompts from selected context + the student profile.

        Shared by answer() and the streaming endpoint so both produce an identical
        prompt for a given selection.
        """
        context_block = "\n\n".join(f"[{it.id}] {it.title}\n{it.body}" for it in selected)
        profile_parts = []
        if weak_concepts:
            profile_parts.append(f"- Mastery is LOW for: {', '.join(weak_concepts)}")
        if strong_concepts:
            profile_parts.append(f"- Mastery is HIGH for: {', '.join(strong_concepts)}")
        if active_misconception_texts:
            profile_parts.append(
                f"- Active misconceptions to address: {'; '.join(active_misconception_texts)}"
            )
        if due_concepts:
            profile_parts.append(
                f"- Due for review (refresh gently if relevant): {', '.join(due_concepts)}"
            )
        profile_block = ("STUDENT PROFILE:\n" + "\n".join(profile_parts) + "\n\n") if profile_parts else ""
        user_prompt = f"{profile_block}CONTEXT ITEMS:\n{context_block}\n\nSTUDENT QUESTION:\n{question}"
        return TUTOR_SYSTEM, user_prompt

    def answer(
        self,
        *,
        student_id: str,
        question: str,
        candidates: list[MemoryItem],
        misconception_concept_ids: set[str] | None = None,
        misconception_topics: set[str] | None = None,
        prerequisites: set[str],
        recent_ids: set[str],
        reuse_counts: dict[str, int],
        due_concept_ids: set[str] | None = None,
        budget: int,
        weak_concepts: list[str] | None = None,
        strong_concepts: list[str] | None = None,
        active_misconception_texts: list[str] | None = None,
        due_concepts: list[str] | None = None,
        preselected_items: list[MemoryItem] | None = None,
    ) -> AgentResponse:
        # When a context selection is supplied (e.g. by a fine-tuned router),
        # use it directly and skip the heuristic engine. Otherwise route.
        if preselected_items is not None:
            selected = preselected_items
            tokens_used = sum((getattr(it, "token_estimate", 0) or 0) for it in selected)
            self.logger.log(
                {
                    "event": "routing_decision",
                    "agent": "tutor",
                    "router": "preselected",
                    "student_id": student_id,
                    "task": question,
                    "selected_ids": [it.id for it in selected],
                    "tokens_used": tokens_used,
                    "budget": budget,
                }
            )
        else:
            task_emb = self.embedder.embed_one(question)
            decision = self.engine.route(
                candidates=candidates,
                task_embedding=task_emb,
                misconception_concept_ids=misconception_concept_ids,
                misconception_topics=misconception_topics,
                prerequisites=prerequisites,
                recent_ids=recent_ids,
                reuse_counts=reuse_counts,
                due_concept_ids=due_concept_ids,
                budget=budget,
            )
            selected = decision.selected
            tokens_used = decision.tokens_used
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

        system, user_prompt = self.build_prompt(
            question, selected,
            weak_concepts=weak_concepts, strong_concepts=strong_concepts,
            active_misconception_texts=active_misconception_texts, due_concepts=due_concepts,
        )
        text = self.llm.complete(system=system, user=user_prompt, max_tokens=1024)

        self.logger.log(
            {
                "event": "tutor_reply",
                "agent": "tutor",
                "student_id": student_id,
                "text": text,
            }
        )
        return AgentResponse(
            text=text, selected_items=selected, tokens_used=tokens_used
        )
