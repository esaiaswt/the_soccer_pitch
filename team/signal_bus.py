"""Signal Bus module for the team/ application.

Provides a thread-safe communication channel for Player agents to broadcast
short intention signals to teammates without requiring LLM calls.

Uses threading.Semaphore(4) for both read and write concurrency limits,
with a threading.Lock protecting the internal signal dict. Retains only
the most recent signal per sender position.

This module operates exclusively within the team/ application.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A short intention signal broadcast by a Player agent.

    Attributes:
        sender_position: The player position that sent the signal (e.g., "Striker").
        signal_type: The type of signal (e.g., "requesting_pass", "making_run").
        payload: Brief additional context, max 50 characters.
        timestamp: Time when the signal was published (time.time()).
    """

    sender_position: str
    signal_type: str
    payload: str
    timestamp: float


class SignalBus:
    """Thread-safe signal bus for inter-player communication.

    Allows Player agents to publish short intention signals visible to
    all teammates. Retains only the most recent signal from each sender
    position. Supports concurrent access from up to 4 readers and 4 writers.
    """

    def __init__(self, max_readers: int = 4, max_writers: int = 4) -> None:
        """Initialize the Signal Bus with concurrency limits.

        Args:
            max_readers: Maximum concurrent readers. Defaults to 4.
            max_writers: Maximum concurrent writers. Defaults to 4.
        """
        self._signals: dict[str, Signal] = {}
        self._lock = threading.Lock()
        self._read_semaphore = threading.Semaphore(max_readers)
        self._write_semaphore = threading.Semaphore(max_writers)

    def publish(self, signal: Signal) -> None:
        """Publish a signal to the bus.

        Replaces any existing signal from the same sender position.
        Rejects signals with payload exceeding 50 characters.

        Args:
            signal: The signal to publish.
        """
        if len(signal.payload) > 50:
            logger.warning(
                "Signal from %s rejected: payload exceeds 50 characters (%d chars)",
                signal.sender_position,
                len(signal.payload),
            )
            return

        self._write_semaphore.acquire()
        try:
            with self._lock:
                self._signals[signal.sender_position] = signal
        finally:
            self._write_semaphore.release()

    def read_all(self, exclude_position: str | None = None) -> list[Signal]:
        """Read all active signals, optionally excluding a sender position.

        Args:
            exclude_position: If provided, signals from this position are
                excluded from the result (typically the reader's own position).

        Returns:
            List of active signals from other players.
        """
        self._read_semaphore.acquire()
        try:
            with self._lock:
                if exclude_position is None:
                    return list(self._signals.values())
                return [
                    signal
                    for signal in self._signals.values()
                    if signal.sender_position != exclude_position
                ]
        finally:
            self._read_semaphore.release()

    def clear(self) -> None:
        """Clear all active signals (e.g., on dead ball detection)."""
        with self._lock:
            self._signals.clear()
