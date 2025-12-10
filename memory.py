from typing import List, Dict, Any
from collections import deque

from config import MEMORY_MAX_TURNS
from logging_utils import get_logger

logger = get_logger()


class ConversationMemory:
    """
    Simple in-memory conversation buffer:
    stores last N user/assistant turns.
    """

    def __init__(self, max_turns: int = MEMORY_MAX_TURNS):
        self.max_turns = max_turns
        self._history = deque(maxlen=max_turns)

    def add_turn(self, role: str, content: str) -> None:
        self._history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        return list(self._history)

    def clear(self):
        self._history.clear()

    def as_chatml(self) -> List[Dict[str, str]]:
        """
        Format for OpenAI chat-style models.
        """
        return self.get_history()


# Simple global if you don't care about multi-user separation
_global_memory = ConversationMemory()


def get_global_memory() -> ConversationMemory:
    return _global_memory
