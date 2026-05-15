from unittest.mock import MagicMock, patch
from learning_memory_os.llm import LLM


def test_llm_complete_returns_text():
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text="hello world")]

    with patch("learning_memory_os.llm.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        client.messages.create.return_value = fake_response

        llm = LLM(api_key="sk-test", model="claude-opus-4-7")
        out = llm.complete(system="be terse", user="hi")

        assert out == "hello world"
        client.messages.create.assert_called_once()
        kwargs = client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-opus-4-7"
        assert kwargs["system"] == "be terse"
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_llm_complete_json_parses_response():
    fake_response = MagicMock()
    fake_response.content = [MagicMock(text='{"k": 1}')]

    with patch("learning_memory_os.llm.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        client.messages.create.return_value = fake_response

        llm = LLM(api_key="sk-test", model="claude-opus-4-7")
        out = llm.complete_json(system="emit json", user="hi")
        assert out == {"k": 1}


def test_llm_complete_json_handles_trailing_text():
    """Regression: LLM sometimes appends commentary after the JSON object."""
    from unittest.mock import MagicMock, patch
    from learning_memory_os.llm import LLM

    fake_response = MagicMock()
    fake_response.content = [
        MagicMock(text='Here is the JSON:\n{"question": "Q?", "rubric": "R"}\nLet me know if you need more.')
    ]
    with patch("learning_memory_os.llm.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        client.messages.create.return_value = fake_response
        llm = LLM(api_key="sk-test", model="claude-opus-4-7")
        out = llm.complete_json(system="x", user="y")
        assert out == {"question": "Q?", "rubric": "R"}


def test_llm_complete_json_handles_code_fence():
    """Regression: LLM sometimes wraps JSON in ```json ... ``` fences."""
    from unittest.mock import MagicMock, patch
    from learning_memory_os.llm import LLM

    fake_response = MagicMock()
    fake_response.content = [
        MagicMock(text='```json\n{"a": 1, "b": [2, 3]}\n```')
    ]
    with patch("learning_memory_os.llm.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        client.messages.create.return_value = fake_response
        llm = LLM(api_key="sk-test", model="claude-opus-4-7")
        out = llm.complete_json(system="x", user="y")
        assert out == {"a": 1, "b": [2, 3]}


def test_llm_complete_json_handles_raw_newlines_in_strings():
    """Regression: LLM sometimes puts literal newlines inside string values."""
    from unittest.mock import MagicMock, patch
    from learning_memory_os.llm import LLM

    fake_response = MagicMock()
    fake_response.content = [
        MagicMock(text='{"question": "Line one\nline two", "rubric": "ok"}')
    ]
    with patch("learning_memory_os.llm.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        client.messages.create.return_value = fake_response
        llm = LLM(api_key="sk-test", model="claude-opus-4-7")
        out = llm.complete_json(system="x", user="y")
        # Result should preserve the newline (now properly decoded as a real newline)
        assert out == {"question": "Line one\nline two", "rubric": "ok"}


def test_llm_complete_json_handles_tab_in_string():
    from unittest.mock import MagicMock, patch
    from learning_memory_os.llm import LLM

    fake_response = MagicMock()
    fake_response.content = [
        MagicMock(text='{"a": "hello\tworld"}')
    ]
    with patch("learning_memory_os.llm.Anthropic") as MockAnthropic:
        client = MockAnthropic.return_value
        client.messages.create.return_value = fake_response
        llm = LLM(api_key="sk-test", model="claude-opus-4-7")
        out = llm.complete_json(system="x", user="y")
        assert out == {"a": "hello\tworld"}
