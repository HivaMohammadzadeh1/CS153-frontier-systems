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
