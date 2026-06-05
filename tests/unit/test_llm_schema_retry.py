"""complete_with_schema retries once when the model skips the tool call."""
from learning_memory_os.llm import LLM


class _Block:
    def __init__(self, type_, name=None, input=None, text=None):
        self.type = type_
        self.name = name
        self.input = input
        self.text = text


class _Resp:
    def __init__(self, content):
        self.content = content


class _FakeMessages:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = 0

    def create(self, **kwargs):
        resp = self._scripted[min(self.calls, len(self._scripted) - 1)]
        self.calls += 1
        return resp


class _FakeClient:
    def __init__(self, scripted):
        self.messages = _FakeMessages(scripted)


def _llm(scripted):
    llm = LLM.__new__(LLM)          # bypass real Anthropic() construction
    llm.model = "test"
    llm._client = _FakeClient(scripted)
    return llm


SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


def test_retries_then_succeeds():
    # First call: no tool_use (just text). Second call: proper tool_use.
    llm = _llm([
        _Resp([_Block("text", text="oops, forgot the tool")]),
        _Resp([_Block("tool_use", name="submit", input={"x": 7})]),
    ])
    out = llm.complete_with_schema(system="s", user="u", schema=SCHEMA, tool_name="submit")
    assert out == {"x": 7}
    assert llm._client.messages.calls == 2   # it retried


def test_first_try_succeeds_no_retry():
    llm = _llm([_Resp([_Block("tool_use", name="submit", input={"x": 1})])])
    assert llm.complete_with_schema(system="s", user="u", schema=SCHEMA, tool_name="submit") == {"x": 1}
    assert llm._client.messages.calls == 1


def test_raises_after_two_failures():
    llm = _llm([_Resp([_Block("text", text="nope")])])  # always toolless
    try:
        llm.complete_with_schema(system="s", user="u", schema=SCHEMA, tool_name="submit")
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "after retry" in str(e)
    assert llm._client.messages.calls == 2
