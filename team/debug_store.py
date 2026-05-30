"""Thread-safe debug store for dashboard consumption.

Provides per-player debug info and coach observation/instruction history
that the Streamlit dashboard reads on each rerun cycle. Player threads
and the Coach thread write data; the dashboard thread reads it.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass


@dataclass
class PlayerDebugInfo:
    """Debug snapshot for a single player agent.

    Attributes
    ----------
    latest_state : dict | None
        The last game state snapshot seen by this player.
    latest_action : dict | None
        The last action submitted by this player (dx, dy, kick).
    latest_instruction : str | None
        The last coach instruction received by this player.
    last_update : float
        Timestamp (time.time()) of the most recent update.
    """

    latest_state: dict | None
    latest_action: dict | None
    latest_instruction: str | None
    last_update: float


class DebugStore:
    """Thread-safe container for per-agent debug data.

    Used by the Streamlit dashboard to display live debug panels for each
    Player agent and the Coach agent's observation/instruction history.
    Multiple writer threads (Coach + 4 Players) and one reader thread
    (Streamlit dashboard) access this concurrently.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._players: dict[str, PlayerDebugInfo] = {}
        self._coach: dict = {"observations": [], "instructions": {}}

    def update_player(self, position: str, info: PlayerDebugInfo) -> None:
        """Store debug info for a player position.

        Parameters
        ----------
        position : str
            The player position key (e.g. "Goalkeeper", "Defender").
        info : PlayerDebugInfo
            The debug snapshot to store for this player.
        """
        with self._lock:
            self._players[position] = info

    def get_player(self, position: str) -> PlayerDebugInfo | None:
        """Return the latest debug info for a player position.

        Parameters
        ----------
        position : str
            The player position key to look up.

        Returns
        -------
        PlayerDebugInfo | None
            The stored debug info, or None if no info has been set
            for that position.
        """
        with self._lock:
            return self._players.get(position)

    def update_coach(self, observations: list[dict], instructions: dict) -> None:
        """Store the coach's most recent observations and instructions.

        Parameters
        ----------
        observations : list[dict]
            The most recent game state observations from the coach.
        instructions : dict
            The most recent instructions issued by the coach,
            keyed by player position.
        """
        with self._lock:
            self._coach = {
                "observations": observations,
                "instructions": instructions,
            }

    def get_coach(self) -> dict:
        """Return the coach's stored observations and instructions.

        Returns a deep copy so that callers cannot mutate the internal state.

        Returns
        -------
        dict
            A dict with 'observations' (list[dict]) and 'instructions' (dict)
            keys. Returns empty defaults if no coach data has been stored.
        """
        with self._lock:
            return copy.deepcopy(self._coach)
