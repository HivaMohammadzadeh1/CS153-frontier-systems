import json
import re
from anthropic import Anthropic


class LLM:
    def __init__(self, api_key: str, model: str = "claude-opus-4-7"):
        self._client = Anthropic(api_key=api_key)
        self.model = model

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> str:
        resp = self._client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> dict:
        text = self.complete(
            system=system, user=user, max_tokens=max_tokens, temperature=0.0
        )
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        payload = match.group(0) if match else text
        return json.loads(payload)
