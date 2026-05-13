from unittest.mock import MagicMock, patch
from learning_memory_os.embeddings import Embedder


def test_embedder_returns_vectors():
    fake_response = MagicMock()
    fake_response.data = [MagicMock(embedding=[0.1] * 1536)]

    with patch("learning_memory_os.embeddings.OpenAI") as MockOpenAI:
        client = MockOpenAI.return_value
        client.embeddings.create.return_value = fake_response

        e = Embedder(api_key="sk-test")
        out = e.embed_one("hello")
        assert len(out) == 1536
        assert out[0] == 0.1
