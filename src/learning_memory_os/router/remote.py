"""Call a remotely-hosted router over an OpenAI-compatible API (e.g. vLLM on a
DigitalOcean GPU Droplet — see deploy/digitalocean/).

Uses the /v1/completions endpoint with the RAW router prompt (no chat template),
matching how the adapter was trained and how router/infer.py runs it locally.

Config via env:
  LMOS_ROUTER_ENDPOINT   base url, e.g. http://203.0.113.10:8000/v1   (required)
  LMOS_ROUTER_MODEL      served-model-name, default "lmos-router-7b"
  LMOS_ROUTER_API_KEY    bearer token if the server requires one
"""
import os

from ..trajectories.schemas import StudentState, PoolItem, TaskType
from .prompt import format_router_input, parse_router_output


def endpoint() -> str | None:
    return os.environ.get("LMOS_ROUTER_ENDPOINT") or None


class RemoteAPIRouter:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None):
        from openai import OpenAI
        self.base_url = base_url or os.environ["LMOS_ROUTER_ENDPOINT"]
        self.model = model or os.environ.get("LMOS_ROUTER_MODEL", "lmos-router-7b")
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key or os.environ.get("LMOS_ROUTER_API_KEY") or "none",
        )

    def route(
        self,
        *,
        student_state: StudentState,
        task_type: TaskType,
        task_text: str,
        budget: int,
        candidate_pool: list[PoolItem],
        max_new_tokens: int = 128,
    ) -> list[str]:
        prompt = format_router_input(
            student_state=student_state,
            task_type=task_type,
            task_text=task_text,
            budget=budget,
            candidate_pool=candidate_pool,
        )
        resp = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        return parse_router_output(resp.choices[0].text)
