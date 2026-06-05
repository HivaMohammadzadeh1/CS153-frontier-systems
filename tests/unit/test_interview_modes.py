"""Tests for multi-turn interview + forward-deployed mode (no live LLM)."""
from learning_memory_os.agents.interview import (
    InterviewAgent, ForwardDeployedAgent, _format_transcript,
)
from learning_memory_os.agents.interview_prompts import FORWARD_CATEGORIES, CATEGORIES


class _FakeLLM:
    """Captures the last (system,user) and returns canned output."""
    def __init__(self, schema_return=None, text_return="NEXT PROBE?"):
        self.schema_return = schema_return or {}
        self.text_return = text_return
        self.last_user = None
        self.last_system = None

    def complete(self, *, system, user, max_tokens=0):
        self.last_system, self.last_user = system, user
        return self.text_return

    def complete_with_schema(self, *, system, user, schema, tool_name,
                             tool_description, max_tokens=0):
        self.last_system, self.last_user = system, user
        return dict(self.schema_return)


def test_format_transcript_labels_followups():
    out = _format_transcript([
        {"q": "Design a serving stack", "a": "I'd use vLLM"},
        {"q": "Why PagedAttention?", "a": "Fragmentation"},
    ])
    assert "QUESTION:" in out
    assert "FOLLOW-UP 1:" in out
    assert "Fragmentation" in out


def test_followup_sees_the_transcript():
    llm = _FakeLLM(text_return="What's your p99 budget?")
    q = InterviewAgent(llm).followup(
        topic_title="KV cache", level="intermediate",
        transcript=[{"q": "Design it", "a": "batching helps"}],
    )
    assert q == "What's your p99 budget?"
    assert "batching helps" in llm.last_user  # the probe is conditioned on the answer


def test_multiturn_evaluate_uses_weighted_overall():
    # category_scores high on communication, low on technicals → weighted overall low
    scores = {c: 40 for c in CATEGORIES}
    scores["communication"] = 100
    llm = _FakeLLM(schema_return={"category_scores": scores})
    ev = InterviewAgent(llm).evaluate(
        topic_title="Inference", transcript=[{"q": "q1", "a": "a1"}, {"q": "q2", "a": "a2"}],
    )
    assert "MULTI-TURN" in llm.last_user
    assert ev["overall_score"] < 50  # communication is a multiplier, not a driver


def test_forward_deployed_normalizes_seven_subskills():
    llm = _FakeLLM(schema_return={"overall_score": 72, "category_scores": {"problem_framing": 80}})
    ev = ForwardDeployedAgent(llm).evaluate(
        scenario="our agent feels slow", response="I'd ask for p99 and tokens/s first",
        topic_title="Latency",
    )
    # every forward sub-skill present after normalization
    for c in FORWARD_CATEGORIES:
        assert c in ev["category_scores"]
    assert "p99" in llm.last_user


def test_forward_respond_conditions_on_transcript_and_message():
    llm = _FakeLLM(text_return="p99 TTFT is 9s on long chats.")
    reply = ForwardDeployedAgent(llm).respond(
        scenario="agent feels slow", topic_title="Latency",
        transcript=[{"student": "is it TTFT or decode?", "customer": "not sure"}],
        message="give me p50/p99 TTFT",
    )
    assert reply == "p99 TTFT is 9s on long chats."
    # the customer-sim prompt must see both the prior exchange and the new question
    assert "is it TTFT or decode?" in llm.last_user
    assert "give me p50/p99 TTFT" in llm.last_user


def test_forward_interactive_evaluate_uses_transcript():
    llm = _FakeLLM(schema_return={"overall_score": 70, "category_scores": {"metric_selection": 85}})
    ev = ForwardDeployedAgent(llm).evaluate(
        scenario="agent feels slow", topic_title="Latency",
        transcript=[{"student": "what are p99 TTFT numbers?", "customer": "9s"}],
        response="root cause: prefill blowup",
    )
    assert "INTERACTIVE" in llm.last_user
    assert "what are p99 TTFT numbers?" in llm.last_user
    for c in FORWARD_CATEGORIES:
        assert c in ev["category_scores"]
