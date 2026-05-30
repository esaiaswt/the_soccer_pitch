"""Unit tests for CoachMemory class.

Tests the rolling buffer, validation, and retrieval methods.
"""

from __future__ import annotations

import time

from team.coach_agent import CoachMemory, REQUIRED_SNAPSHOT_FIELDS


def _valid_snapshot(**overrides) -> dict:
    """Create a valid snapshot with all required fields."""
    base = {
        "ball": {"x": 600.0, "y": 425.0},
        "players": {"Red_Goalkeeper": {"x": 100.0, "y": 425.0}},
        "score": {"Red": 0, "Blue": 0},
        "time_left": 90.0,
        "match_state": "Playing",
    }
    base.update(overrides)
    return base


class TestCoachMemoryInit:
    """Test CoachMemory initialization."""

    def test_default_max_size(self):
        memory = CoachMemory()
        assert memory._max_size == 50

    def test_custom_max_size(self):
        memory = CoachMemory(max_size=10)
        assert memory._max_size == 10

    def test_starts_empty(self):
        memory = CoachMemory()
        assert memory.get_history() == []


class TestCoachMemoryAddSnapshot:
    """Test add_snapshot validation and storage."""

    def test_valid_snapshot_added(self):
        memory = CoachMemory()
        memory.add_snapshot(_valid_snapshot())
        assert len(memory.get_history()) == 1

    def test_received_at_timestamp_added(self):
        memory = CoachMemory()
        before = time.time()
        memory.add_snapshot(_valid_snapshot())
        after = time.time()

        entry = memory.get_history()[0]
        assert "received_at" in entry
        assert before <= entry["received_at"] <= after

    def test_original_fields_preserved(self):
        memory = CoachMemory()
        snapshot = _valid_snapshot(time_left=45.0)
        memory.add_snapshot(snapshot)

        entry = memory.get_history()[0]
        assert entry["time_left"] == 45.0
        assert entry["match_state"] == "Playing"
        assert entry["ball"] == {"x": 600.0, "y": 425.0}

    def test_missing_ball_rejected(self):
        memory = CoachMemory()
        snapshot = _valid_snapshot()
        del snapshot["ball"]
        memory.add_snapshot(snapshot)
        assert len(memory.get_history()) == 0

    def test_missing_players_rejected(self):
        memory = CoachMemory()
        snapshot = _valid_snapshot()
        del snapshot["players"]
        memory.add_snapshot(snapshot)
        assert len(memory.get_history()) == 0

    def test_missing_score_rejected(self):
        memory = CoachMemory()
        snapshot = _valid_snapshot()
        del snapshot["score"]
        memory.add_snapshot(snapshot)
        assert len(memory.get_history()) == 0

    def test_missing_time_left_rejected(self):
        memory = CoachMemory()
        snapshot = _valid_snapshot()
        del snapshot["time_left"]
        memory.add_snapshot(snapshot)
        assert len(memory.get_history()) == 0

    def test_missing_match_state_rejected(self):
        memory = CoachMemory()
        snapshot = _valid_snapshot()
        del snapshot["match_state"]
        memory.add_snapshot(snapshot)
        assert len(memory.get_history()) == 0

    def test_missing_multiple_fields_rejected(self):
        memory = CoachMemory()
        memory.add_snapshot({"ball": {"x": 0, "y": 0}})
        assert len(memory.get_history()) == 0

    def test_empty_dict_rejected(self):
        memory = CoachMemory()
        memory.add_snapshot({})
        assert len(memory.get_history()) == 0


class TestCoachMemoryBufferOverflow:
    """Test rolling buffer behavior when full."""

    def test_discards_oldest_when_full(self):
        memory = CoachMemory(max_size=3)
        for i in range(5):
            memory.add_snapshot(_valid_snapshot(time_left=float(i)))

        history = memory.get_history()
        assert len(history) == 3
        # Should contain the 3 most recent (ids 2, 3, 4)
        assert history[0]["time_left"] == 2.0
        assert history[1]["time_left"] == 3.0
        assert history[2]["time_left"] == 4.0

    def test_never_exceeds_max_size(self):
        memory = CoachMemory(max_size=5)
        for i in range(100):
            memory.add_snapshot(_valid_snapshot(time_left=float(i)))
        assert len(memory.get_history()) == 5

    def test_chronological_order_maintained(self):
        memory = CoachMemory(max_size=10)
        for i in range(7):
            memory.add_snapshot(_valid_snapshot(time_left=float(i)))

        history = memory.get_history()
        for j in range(len(history) - 1):
            assert history[j]["received_at"] <= history[j + 1]["received_at"]


class TestCoachMemoryGetHistory:
    """Test get_history() method."""

    def test_returns_all_snapshots(self):
        memory = CoachMemory(max_size=10)
        for i in range(5):
            memory.add_snapshot(_valid_snapshot(time_left=float(i)))
        assert len(memory.get_history()) == 5

    def test_oldest_first_order(self):
        memory = CoachMemory(max_size=10)
        for i in range(5):
            memory.add_snapshot(_valid_snapshot(time_left=float(i)))

        history = memory.get_history()
        assert history[0]["time_left"] == 0.0
        assert history[-1]["time_left"] == 4.0

    def test_returns_copy_not_reference(self):
        memory = CoachMemory()
        memory.add_snapshot(_valid_snapshot())
        h1 = memory.get_history()
        h2 = memory.get_history()
        assert h1 is not h2


class TestCoachMemoryGetRecent:
    """Test get_recent(n) method."""

    def test_returns_n_most_recent(self):
        memory = CoachMemory(max_size=10)
        for i in range(5):
            memory.add_snapshot(_valid_snapshot(time_left=float(i)))

        recent = memory.get_recent(2)
        assert len(recent) == 2
        assert recent[0]["time_left"] == 3.0
        assert recent[1]["time_left"] == 4.0

    def test_oldest_first_within_returned_list(self):
        memory = CoachMemory(max_size=10)
        for i in range(5):
            memory.add_snapshot(_valid_snapshot(time_left=float(i)))

        recent = memory.get_recent(3)
        assert recent[0]["time_left"] == 2.0
        assert recent[1]["time_left"] == 3.0
        assert recent[2]["time_left"] == 4.0

    def test_n_exceeds_buffer_returns_all(self):
        memory = CoachMemory(max_size=10)
        for i in range(3):
            memory.add_snapshot(_valid_snapshot(time_left=float(i)))

        recent = memory.get_recent(10)
        assert len(recent) == 3

    def test_n_zero_returns_empty(self):
        memory = CoachMemory(max_size=10)
        memory.add_snapshot(_valid_snapshot())
        assert memory.get_recent(0) == []

    def test_n_negative_returns_empty(self):
        memory = CoachMemory(max_size=10)
        memory.add_snapshot(_valid_snapshot())
        assert memory.get_recent(-1) == []

    def test_empty_buffer_returns_empty(self):
        memory = CoachMemory()
        assert memory.get_recent(5) == []
