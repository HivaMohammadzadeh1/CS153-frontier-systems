from dataclasses import dataclass
from ..schemas.memory import MemoryItem


@dataclass
class AgentResponse:
    text: str
    selected_items: list[MemoryItem]
    tokens_used: int
