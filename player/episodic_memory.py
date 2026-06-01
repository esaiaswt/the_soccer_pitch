"""Episodic memory module for storing past game states, actions, and outcomes.

Provides a fixed-capacity ring buffer of Episode records that the Player_Agent
uses to inform decisions based on recent history. Uses collections.deque for
O(1) amortized append with automatic eviction of the oldest entries.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass


@dataclass
class Episode:
    """A single episode recording a game state, action taken, and outcome."""

    cycle: int
    game_state: dict  # snapshot at decision time
    action: dict  # {"dx": float, "dy": float, "kick": bool}
    next_state_delta: dict  # computed diff of relevant state fields
    effectiveness: float | None  # filled in by Reflection Engine


class EpisodicMemory:
    """Fixed-capacity episodic memory with chronological ordering.

    Stores episodes in a ring buffer (deque) that automatically evicts the
    oldest episode when capacity is reached, ensuring O(1) amortized insertion.
    """

    def __init__(self, max_capacity: int = 100) -> None:
        self._buffer: collections.deque[Episode] = collections.deque(
            maxlen=max_capacity
        )

    def add(self, episode: Episode) -> None:
        """Add an episode to memory. Evicts oldest if at capacity."""
        self._buffer.append(episode)

    def get_all(self) -> list[Episode]:
        """Return all episodes in chronological order (oldest to newest)."""
        return list(self._buffer)

    def get_recent(self, n: int) -> list[Episode]:
        """Return the n most recent episodes in chronological order.

        If n exceeds the number of stored episodes, returns all episodes.
        """
        if n >= len(self._buffer):
            return list(self._buffer)
        return list(collections.deque(self._buffer, maxlen=n))

    def __len__(self) -> int:
        """Return the number of episodes currently stored."""
        return len(self._buffer)
