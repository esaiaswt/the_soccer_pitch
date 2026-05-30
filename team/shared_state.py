"""Thread-safe shared state container for game state snapshots.

Provides atomic get/set access to the latest game state snapshot received
from the State Poller. Multiple reader threads (Coach + 4 Players) and one
writer thread (State Poller) access this concurrently.
"""

from __future__ import annotations

import threading
import time


class SharedState:
    """Thread-safe container holding the latest game state snapshot.

    Uses threading.Lock for atomic reads and writes, ensuring that the
    State Poller can update the snapshot while Coach and Player agents
    read it concurrently without data races.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: dict | None = None
        self._last_update_time: float | None = None

    def get_snapshot(self) -> dict | None:
        """Return the most recent game state snapshot, or None if not yet set.

        Returns
        -------
        dict | None
            The latest game state snapshot, or None if no snapshot has been
            stored yet.
        """
        with self._lock:
            return self._snapshot

    def set_snapshot(self, snapshot: dict) -> None:
        """Store a new game state snapshot and record the update timestamp.

        Parameters
        ----------
        snapshot : dict
            The game state snapshot received from the Pitch server.
        """
        with self._lock:
            self._snapshot = snapshot
            self._last_update_time = time.time()

    def get_last_update_time(self) -> float | None:
        """Return the timestamp of the last successful snapshot update.

        Returns
        -------
        float | None
            The time.time() value when set_snapshot() was last called,
            or None if no snapshot has been stored yet.
        """
        with self._lock:
            return self._last_update_time
