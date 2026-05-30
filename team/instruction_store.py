"""Instruction store for Coach-to-Player communication.

Thread-safe container for Coach instructions, one slot per player position.
The Coach writes instructions and Player threads read them concurrently.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class CoachInstruction:
    """A tactical instruction from the Coach to a specific Player.

    Attributes
    ----------
    content : str
        Natural-language tactical guidance. Never truncated regardless of length.
    timestamp : float
        time.time() when the instruction was generated.
    target_position : str
        Target player position: "Goalkeeper", "Defender", "Midfielder", or "Striker".
    """

    content: str
    timestamp: float
    target_position: str


class InstructionStore:
    """Thread-safe store for Coach instructions, one slot per player position.

    The Coach agent writes instructions via set_instruction(), and Player agent
    threads read them via get_instruction() or get_all_instructions(). A
    threading.Lock ensures concurrent access safety.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._instructions: dict[str, CoachInstruction] = {}

    def set_instruction(self, position: str, instruction: CoachInstruction) -> None:
        """Store a Coach instruction for the given player position.

        Overwrites any previously stored instruction for that position.
        Content is stored without truncation regardless of length.

        Parameters
        ----------
        position : str
            The target player position (e.g. "Goalkeeper", "Striker").
        instruction : CoachInstruction
            The instruction to store.
        """
        with self._lock:
            self._instructions[position] = instruction

    def get_instruction(self, position: str) -> CoachInstruction | None:
        """Retrieve the latest Coach instruction for a given position.

        Parameters
        ----------
        position : str
            The player position to look up.

        Returns
        -------
        CoachInstruction | None
            The stored instruction, or None if no instruction has been set
            for that position.
        """
        with self._lock:
            return self._instructions.get(position)

    def get_all_instructions(self) -> dict[str, CoachInstruction]:
        """Retrieve all stored Coach instructions.

        Returns
        -------
        dict[str, CoachInstruction]
            A copy of the internal mapping from position to instruction.
        """
        with self._lock:
            return dict(self._instructions)
