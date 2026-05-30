"""Unit tests for team/instruction_store.py."""

import threading
import time

from team.instruction_store import CoachInstruction, InstructionStore


class TestCoachInstruction:
    """Tests for the CoachInstruction dataclass."""

    def test_creation(self):
        """CoachInstruction stores content, timestamp, and target_position."""
        instr = CoachInstruction(
            content="Push up the left flank",
            timestamp=1000.0,
            target_position="Midfielder",
        )
        assert instr.content == "Push up the left flank"
        assert instr.timestamp == 1000.0
        assert instr.target_position == "Midfielder"

    def test_long_content_not_truncated(self):
        """Instructions longer than 500 chars are stored without truncation."""
        long_content = "x" * 2000
        instr = CoachInstruction(
            content=long_content,
            timestamp=1.0,
            target_position="Striker",
        )
        assert len(instr.content) == 2000
        assert instr.content == long_content


class TestInstructionStore:
    """Tests for the InstructionStore class."""

    def test_get_instruction_returns_none_when_empty(self):
        """get_instruction returns None for positions with no instruction."""
        store = InstructionStore()
        assert store.get_instruction("Goalkeeper") is None

    def test_set_and_get_instruction(self):
        """set_instruction stores and get_instruction retrieves correctly."""
        store = InstructionStore()
        instr = CoachInstruction(
            content="Stay on the line",
            timestamp=100.0,
            target_position="Goalkeeper",
        )
        store.set_instruction("Goalkeeper", instr)
        result = store.get_instruction("Goalkeeper")
        assert result is not None
        assert result.content == "Stay on the line"
        assert result.timestamp == 100.0
        assert result.target_position == "Goalkeeper"

    def test_set_instruction_overwrites_previous(self):
        """Setting a new instruction for the same position overwrites the old one."""
        store = InstructionStore()
        instr1 = CoachInstruction(content="First", timestamp=1.0, target_position="Defender")
        instr2 = CoachInstruction(content="Second", timestamp=2.0, target_position="Defender")
        store.set_instruction("Defender", instr1)
        store.set_instruction("Defender", instr2)
        result = store.get_instruction("Defender")
        assert result is not None
        assert result.content == "Second"
        assert result.timestamp == 2.0

    def test_get_all_instructions_empty(self):
        """get_all_instructions returns empty dict when no instructions set."""
        store = InstructionStore()
        assert store.get_all_instructions() == {}

    def test_get_all_instructions_returns_all(self):
        """get_all_instructions returns all stored instructions."""
        store = InstructionStore()
        positions = ["Goalkeeper", "Defender", "Midfielder", "Striker"]
        for i, pos in enumerate(positions):
            instr = CoachInstruction(
                content=f"Instruction for {pos}",
                timestamp=float(i),
                target_position=pos,
            )
            store.set_instruction(pos, instr)

        all_instructions = store.get_all_instructions()
        assert len(all_instructions) == 4
        for pos in positions:
            assert pos in all_instructions
            assert all_instructions[pos].target_position == pos

    def test_get_all_instructions_returns_copy(self):
        """get_all_instructions returns a copy, not a reference to internal state."""
        store = InstructionStore()
        instr = CoachInstruction(content="Test", timestamp=1.0, target_position="Striker")
        store.set_instruction("Striker", instr)

        result = store.get_all_instructions()
        result.pop("Striker")
        # Internal state should be unaffected
        assert store.get_instruction("Striker") is not None

    def test_concurrent_access(self):
        """Multiple threads can read and write without errors."""
        store = InstructionStore()
        errors: list[Exception] = []

        def writer(position: str, count: int):
            try:
                for i in range(count):
                    instr = CoachInstruction(
                        content=f"Instruction {i}",
                        timestamp=float(i),
                        target_position=position,
                    )
                    store.set_instruction(position, instr)
            except Exception as e:
                errors.append(e)

        def reader(position: str, count: int):
            try:
                for _ in range(count):
                    store.get_instruction(position)
                    store.get_all_instructions()
            except Exception as e:
                errors.append(e)

        threads = []
        positions = ["Goalkeeper", "Defender", "Midfielder", "Striker"]
        for pos in positions:
            threads.append(threading.Thread(target=writer, args=(pos, 100)))
            threads.append(threading.Thread(target=reader, args=(pos, 100)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent access errors: {errors}"
