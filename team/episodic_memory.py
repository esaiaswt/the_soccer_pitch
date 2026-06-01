"""Episodic Memory module for the team/ application.

Stores past game states, actions, and outcomes as timestamped episodes.
Uses a ring buffer (collections.deque) for O(1) append with automatic
eviction of oldest entries when capacity is reached.

This is an independent implementation with no imports from the player/ package.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class Episode:
    """A single episode recording a game state, action taken, and outcome."""

    cycle: int
    game_state: dict  # snapshot at decision time
    action: dict  # {"dx": float, "dy": float, "kick": bool}
    next_state_delta: dict  # computed diff of relevant state fields
    effectiveness: float | None = None  # filled in by Reflection Engine


class EpisodicMemory:
    """Ring-buffer episodic memory with configurable capacity.

    Stores episodes in chronological order and automatically evicts
    the oldest episode when the configured maximum capacity is reached.
    """

    def __init__(self, max_capacity: int = 100) -> None:
        """Initialize episodic memory with a maximum capacity.

        Args:
            max_capacity: Maximum number of episodes to retain. Defaults to 100.
        """
        self._buffer: deque[Episode] = deque(maxlen=max_capacity)

    def add(self, episode: Episode) -> None:
        """Add an episode to memory. Evicts oldest if at capacity.

        Args:
            episode: The episode to store.
        """
        self._buffer.append(episode)

    def get_all(self) -> list[Episode]:
        """Return all episodes in chronological order (oldest to newest).

        Returns:
            List of all stored episodes ordered by insertion time.
        """
        return list(self._buffer)

    def get_recent(self, n: int) -> list[Episode]:
        """Return the n most recent episodes in chronological order.

        Args:
            n: Number of recent episodes to retrieve.

        Returns:
            List of up to n most recent episodes, oldest first.
        """
        if n <= 0:
            return []
        items = list(self._buffer)
        return items[-n:]

    def __len__(self) -> int:
        """Return the current number of stored episodes."""
        return len(self._buffer)
