"""Run a fine-tuned LoRA router against live product data.

Bridges the product's MemoryItem candidates + DB student state into the exact
PoolItem/StudentState representation the router was trained on (see
trajectories/sampler.py), runs the adapter in-process, and maps the selected
short-ids back to the original MemoryItem candidates.

Models load lazily and are cached, so the first finetuned request pays the
load cost and subsequent ones are fast. Intended for local/dev testing of the
fine-tuned routers in the product; the 7B is better served on a GPU host.
"""
from functools import lru_cache
from pathlib import Path

import yaml

from ..trajectories.schemas import PoolItem, StudentState, TaskType
from .infer import FineTunedRouter

_ROOT = Path(__file__).resolve().parents[3]
_SIZES = {
    s["id"]: s
    for s in (yaml.safe_load((_ROOT / "config" / "router_sizes.yaml").read_text()) or {}).get("sizes", [])
}
_CKPT = _ROOT / "data" / "router_checkpoints"


def available_sizes() -> list[str]:
    """Routers usable right now: local adapters present on disk, plus "remote"
    if a hosted endpoint (LMOS_ROUTER_ENDPOINT) is configured."""
    sizes = [sid for sid in _SIZES if (_CKPT / sid / "adapter" / "adapter_config.json").exists()]
    from .remote import endpoint as _remote_endpoint
    if _remote_endpoint():
        sizes.append("remote")
    return sizes


@lru_cache(maxsize=3)
def get_router(size_id: str):
    """Return a router object for a local size id or the hosted "remote" backend."""
    if size_id == "remote":
        from .remote import RemoteAPIRouter
        return RemoteAPIRouter()
    if size_id not in _SIZES:
        raise ValueError(f"unknown router size: {size_id}")
    return FineTunedRouter(adapter_dir=_CKPT / size_id / "adapter", base_model=_SIZES[size_id]["hf_model"])


def _short(uuid_id: str) -> str:
    # Matches sampler.py: short id = first 8 hex chars of the UUID.
    return uuid_id.replace("-", "")[:8]


def pool_from_candidates(candidates) -> tuple[list[PoolItem], dict]:
    """MemoryItem candidates -> PoolItem list (+ short-id -> MemoryItem map)."""
    pool: list[PoolItem] = []
    by_short: dict = {}
    for it in candidates:
        sid = _short(it.id)
        by_short[sid] = it
        body = getattr(it, "body", "") or ""
        te = getattr(it, "token_estimate", None) or max(1, len(body) // 4)
        pool.append(PoolItem(id=sid, title=it.title, body_excerpt=body[:300], token_estimate=te))
    return pool, by_short


def _student_state(student, student_id: str) -> StudentState:
    mastery = {m.concept_id.replace("-", "")[:8]: round(m.score, 2) for m in student.mastery_for(student_id)[:8]}
    misc = [m["description"][:200] for m in student.active_misconceptions(student_id)[:2]]
    return StudentState(student_id=student_id, mastery=mastery, active_misconceptions=misc, recent_episodic_ids=[])


def finetuned_select(
    *,
    size_id: str,
    student,
    student_id: str,
    question: str,
    candidates: list,
    budget: int,
    task_type: TaskType = TaskType.EXPLAIN,
) -> list:
    """Select context items with the fine-tuned router; returns MemoryItem objects
    (a subset of `candidates`, preserving their order)."""
    pool, by_short = pool_from_candidates(candidates)
    state = _student_state(student, student_id)
    router = get_router(size_id)
    selected_ids = set(
        router.route(
            student_state=state,
            task_type=task_type,
            task_text=question,
            budget=budget,
            candidate_pool=pool,
        )
    )
    return [it for it in candidates if _short(it.id) in selected_ids]
